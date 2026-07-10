from __future__ import annotations

import json
import time
import uuid
from typing import Any

from .openai_compat import (
    CompletionAccumulator,
    normalize_chat_messages,
)



def _repair_tool_call_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure every tool message has a preceding assistant with tool_calls."""
    result: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "tool":
            pending.append(msg)
            continue
        if pending:
            _flush_pending(result, pending)
        result.append(msg)
    if pending:
        _flush_pending(result, pending)
    return result


def _flush_pending(result: list[dict[str, Any]], pending: list[dict[str, Any]]) -> None:
    """Flush pending tool messages, inserting a placeholder assistant if needed."""
    call_ids = [m.get("tool_call_id", "") for m in pending]
    if not _preceding_assistant_covers(result, call_ids):
        result.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": "", "arguments": "{}"}}
                for cid in call_ids
            ],
        })
    result.extend(pending)
    pending.clear()


def _preceding_assistant_covers(result: list[dict[str, Any]], call_ids: list[str]) -> bool:
    """Check if the last message is an assistant whose tool_calls cover all call_ids."""
    if not result:
        return False
    last = result[-1]
    if last.get("role") != "assistant":
        return False
    tc = last.get("tool_calls") or []
    tc_ids = {t.get("id") for t in tc}
    return tc_ids >= set(call_ids)

def _input_item_to_messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one Responses API input item to zero or more Chat messages.

    Handles ``message``, ``function_call`` and ``function_call_output``.
    Unknown item types are ignored so we never crash on client quirks.
    """
    if not isinstance(item, dict):
        return []
    item_type = item.get("type", "message")

    if item_type == "message":
        role = item.get("role", "user")
        content = item.get("content", "")
        if isinstance(content, list):
            text_parts = [
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            content = "\n".join(text_parts) if text_parts else str(content)
        if not content:
            return []
        return [{"role": role, "content": str(content)}]

    if item_type == "function_call":
        name = item.get("name", "")
        if not name:
            return []
        arguments = item.get("arguments", "")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        call_id = (
            item.get("call_id")
            or item.get("id")
            or f"call_{uuid.uuid4().hex[:24]}"
        )
        return [{
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": str(arguments or "")},
            }],
        }]

    if item_type == "function_call_output":
        call_id = (
            item.get("call_id")
            or item.get("id")
            or f"call_{uuid.uuid4().hex[:24]}"
        )
        output = item.get("output", "")
        if isinstance(output, list):
            text_parts = [
                b.get("text", "") for b in output
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            output = "\n".join(text_parts) if text_parts else str(output)
        return [{
            "role": "tool",
            "tool_call_id": call_id,
            "content": str(output),
        }]

    return []


def _input_to_messages(
    input_data: str | list[dict[str, Any]],
    instructions: str | None = None,
) -> list[dict[str, Any]]:
    """Convert Responses API input + instructions to OpenAI-style messages."""
    result: list[dict[str, Any]] = []
    if instructions:
        result.append({"role": "system", "content": instructions})
    if isinstance(input_data, str):
        result.append({"role": "user", "content": input_data})
    elif isinstance(input_data, list):
        for msg in input_data:
            result.extend(_input_item_to_messages(msg))
    result = _repair_tool_call_messages(result)
    return normalize_chat_messages(result)


def build_upstream_payload(
    body: dict[str, Any],
    *,
    session: Any,
    run_id: str,
    client_id: str,
    trace_session_id: str | None = None,
    upstream_model_id: str | None = None,
) -> dict[str, Any]:
    """Build Codebuff upstream payload from a Responses API request."""
    from .openai_compat import _UPSTREAM_CHAT_KEYS, model_id as oai_model_id

    messages = _input_to_messages(body.get("input"), body.get("instructions"))
    payload = {
        key: body[key] for key in _UPSTREAM_CHAT_KEYS
        if key in body and body[key] is not None
    }
    # Responses API tools differ from Chat Completions:
    #   Responses: {"type": "function", "name": "x", "parameters": {...}}
    #   Chat:      {"type": "function", "function": {"name": "x", "parameters": {...}}}
    # Upstream only accepts type="function" tools; drop others (custom, file_search, etc).
    if "tools" in payload:
        converted = []
        for tool in payload["tools"]:
            if tool.get("type") != "function":
                continue
            if "function" not in tool:
                fn_fields = {
                    k: tool[k]
                    for k in ("name", "description", "parameters", "strict")
                    if k in tool
                }
                if not isinstance(fn_fields.get("parameters"), dict) or "type" not in fn_fields["parameters"]:
                    fn_fields["parameters"] = {"type": "object", "properties": {}}
                converted.append({"type": "function", "function": fn_fields})
            else:
                fn = dict(tool["function"])
                if not isinstance(fn.get("parameters"), dict) or "type" not in fn["parameters"]:
                    fn["parameters"] = {"type": "object", "properties": {}}
                    tool["function"] = fn
                converted.append(tool)
        payload["tools"] = converted
    payload["model"] = upstream_model_id or oai_model_id(body.get("model"))
    payload["messages"] = messages
    payload["stream"] = True
    payload.setdefault("stop", [])
    payload["provider"] = {"data_collection": "deny"}
    payload["codebuff_metadata"] = {
        "freebuff_instance_id": session.instance_id,
        "trace_session_id": trace_session_id or str(uuid.uuid4()),
        "run_id": run_id,
        "client_id": client_id,
        "cost_mode": "free",
    }
    return payload


def encode_responses_event(event: str, data: dict[str, Any]) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
    )


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _ensure_text_item(state: dict[str, Any]) -> dict[str, Any]:
    if "text_item" not in state:
        state["text_item"] = {
            "id": f"msg_{uuid.uuid4().hex}",
            "output_index": state["next_output_index"],
            "added": False,
            "full_text": "",
        }
        state["next_output_index"] += 1
    return state["text_item"]


def _ensure_reasoning_item(state: dict[str, Any]) -> dict[str, Any]:
    if "reasoning_item" not in state:
        state["reasoning_item"] = {
            "id": f"rs_{uuid.uuid4().hex}",
            "output_index": state["next_output_index"],
            "added": False,
            "full_text": "",
        }
        state["next_output_index"] += 1
    return state["reasoning_item"]


def _ensure_call_item(
    state: dict[str, Any], index: int,
) -> dict[str, Any]:
    state.setdefault("tool_items", {})
    items = state["tool_items"]
    if index not in items:
        items[index] = {
            "id": f"fc_{uuid.uuid4().hex}",
            "output_index": state["next_output_index"],
            "added": False,
            "name": "",
            "arguments": "",
        }
        state["next_output_index"] += 1
    return items[index]


def responses_stream_events(
    openai_chunk: dict[str, Any],
    *,
    response_id: str,
    model: str,
    state: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Convert one OpenAI chunk into Responses API SSE events.

    Mirrors OpenAI's Responses API event ordering so conformant clients
    (Open Design, SDKs) can parse incrementally:

    - ``response.created`` (once)
    - For each output item: ``response.output_item.added``,
      per-block deltas (``response.output_text.delta`` /
      ``response.reasoning_text.delta`` /
      ``response.function_call_arguments.delta``) then
      ``*.done``, then ``response.output_item.done``.
    - ``response.completed`` (once, on finish_reason).
    """
    events: list[tuple[str, dict[str, Any]]] = []

    state.setdefault("next_output_index", 0)
    state.setdefault("final_emitted", False)

    if not state.get("created"):
        state["created"] = True
        events.append(("response.created", {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": int(time.time()),
                "model": model,
                "output": [],
                "status": "in_progress",
                "usage": None,
            },
        }))
        events.append(("response.in_progress", {
            "type": "response.in_progress",
            "response": {
                "id": response_id,
                "status": "in_progress",
            },
        }))

    finish_reason = None
    usage: dict[str, Any] | None = None
    for choice in openai_chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        finish_reason = finish_reason or choice.get("finish_reason")

        content_piece = delta.get("content")
        if content_piece:
            item = _ensure_text_item(state)
            item["full_text"] += content_piece
            if not item["added"]:
                item["added"] = True
                events.append(("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": item["output_index"],
                    "item": {
                        "id": item["id"],
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                }))
                events.append(("response.content_part.added", {
                    "type": "response.content_part.added",
                    "item_id": item["id"],
                    "output_index": item["output_index"],
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                }))
            events.append(("response.output_text.delta", {
                "type": "response.output_text.delta",
                "item_id": item["id"],
                "output_index": item["output_index"],
                "content_index": 0,
                "delta": content_piece,
            }))

        reasoning_piece = delta.get("reasoning_content")
        if reasoning_piece:
            item = _ensure_reasoning_item(state)
            item["full_text"] += reasoning_piece
            if not item["added"]:
                item["added"] = True
                events.append(("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": item["output_index"],
                    "item": {
                        "id": item["id"],
                        "type": "reasoning",
                        "summary": [],
                    },
                }))
            events.append(("response.reasoning_text.delta", {
                "type": "response.reasoning_text.delta",
                "item_id": item["id"],
                "output_index": item["output_index"],
                "content_index": 0,
                "delta": reasoning_piece,
            }))

        for tool_call in delta.get("tool_calls") or []:
            tc_index = int(tool_call.get("index", 0))
            item = _ensure_call_item(state, tc_index)
            if tool_call.get("id"):
                item["id"] = tool_call["id"]
            function = tool_call.get("function") or {}
            if function.get("name"):
                item["name"] += function["name"]
            arguments_chunk = function.get("arguments")
            if arguments_chunk:
                item["arguments"] += arguments_chunk
            if not item["added"]:
                item["added"] = True
                events.append(("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": item["output_index"],
                    "item": {
                        "id": item["id"],
                        "type": "function_call",
                        "status": "in_progress",
                        "call_id": item["id"],
                        "name": item["name"],
                        "arguments": item["arguments"],
                    },
                }))
            if arguments_chunk:
                events.append(("response.function_call_arguments.delta", {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item["id"],
                    "output_index": item["output_index"],
                    "delta": arguments_chunk,
                }))

    usage = openai_chunk.get("usage") or usage

    if finish_reason and not state.get("final_emitted"):
        state["final_emitted"] = True
        completed_output: list[dict[str, Any]] = []

        reasoning = state.get("reasoning_item")
        if reasoning and reasoning["added"]:
            full_reasoning = reasoning["full_text"]
            events.append(("response.reasoning_text.done", {
                "type": "response.reasoning_text.done",
                "item_id": reasoning["id"],
                "output_index": reasoning["output_index"],
                "content_index": 0,
                "text": full_reasoning,
            }))
            events.append(("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": reasoning["output_index"],
                "item": {
                    "id": reasoning["id"],
                    "type": "reasoning",
                    "summary": [{
                        "type": "summary_text",
                        "text": full_reasoning,
                    }],
                },
            }))
            completed_output.append({
                "type": "reasoning",
                "id": reasoning["id"],
                "summary": [{
                    "type": "summary_text",
                    "text": full_reasoning,
                }],
            })

        for tc_index in sorted(state.get("tool_items") or {}):
            item = state["tool_items"][tc_index]
            if not item["added"]:
                continue
            events.append(("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "item_id": item["id"],
                "output_index": item["output_index"],
                "arguments": item["arguments"],
            }))
            events.append(("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": item["output_index"],
                "item": {
                    "id": item["id"],
                    "type": "function_call",
                    "status": "completed",
                    "call_id": item["id"],
                    "name": item["name"],
                    "arguments": item["arguments"],
                },
            }))
            completed_output.append({
                "type": "function_call",
                "id": item["id"],
                "call_id": item["id"],
                "name": item["name"],
                "arguments": item["arguments"],
            })

        text_item = state.get("text_item")
        if text_item and (text_item["added"] or text_item["full_text"]):
            full_text = text_item["full_text"]
            if not text_item["added"]:
                text_item["added"] = True
                events.append(("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": text_item["output_index"],
                    "item": {
                        "id": text_item["id"],
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                }))
                events.append(("response.content_part.added", {
                    "type": "response.content_part.added",
                    "item_id": text_item["id"],
                    "output_index": text_item["output_index"],
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                }))
            events.append(("response.output_text.done", {
                "type": "response.output_text.done",
                "item_id": text_item["id"],
                "output_index": text_item["output_index"],
                "content_index": 0,
                "text": full_text,
            }))
            events.append(("response.content_part.done", {
                "type": "response.content_part.done",
                "item_id": text_item["id"],
                "output_index": text_item["output_index"],
                "content_index": 0,
                "part": {"type": "output_text", "text": full_text, "annotations": []},
            }))
            events.append(("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": text_item["output_index"],
                "item": {
                    "id": text_item["id"],
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": full_text,
                        "annotations": [],
                    }],
                },
            }))
            completed_output.append({
                "type": "message",
                "id": text_item["id"],
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": full_text,
                    "annotations": [],
                }],
            })

        if not completed_output:
            empty_msg_id = f"msg_{uuid.uuid4().hex}"
            events.append(("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": empty_msg_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
            }))
            events.append(("response.content_part.added", {
                "type": "response.content_part.added",
                "item_id": empty_msg_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            }))
            events.append(("response.output_text.done", {
                "type": "response.output_text.done",
                "item_id": empty_msg_id,
                "output_index": 0,
                "content_index": 0,
                "text": "",
            }))
            events.append(("response.content_part.done", {
                "type": "response.content_part.done",
                "item_id": empty_msg_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            }))
            events.append(("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": empty_msg_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [],
                },
            }))

        events.append(("response.completed", {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": int(time.time()),
                "model": model,
                "status": "completed",
                "output": state.get("_final_output") or _snapshot_output(state),
                "usage": {
                    "input_tokens": (usage or {}).get("prompt_tokens", 0),
                    "output_tokens": (usage or {}).get("completion_tokens", 0),
                    "total_tokens": (usage or {}).get("total_tokens", 0),
                },
            },
        }))

    return events


def _snapshot_output(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Re-derive completed output items from accumulated state.

    Used as a safety net so clients always see the full output in
    ``response.completed`` regardless of add/done event ordering issues.
    """
    output: list[dict[str, Any]] = []
    if state.get("reasoning_item", {}).get("added"):
        rs = state["reasoning_item"]
        output.append({
            "type": "reasoning",
            "id": rs["id"],
            "summary": [{"type": "summary_text", "text": rs["full_text"]}],
        })
    for index in sorted(state.get("tool_items") or {}):
        item = state["tool_items"][index]
        if not item["added"]:
            continue
        output.append({
            "type": "function_call",
            "id": item["id"],
            "call_id": item["id"],
            "name": item["name"],
            "arguments": item["arguments"],
        })
    text = state.get("text_item")
    if text and (text["added"] or text["full_text"]):
        output.append({
            "type": "message",
            "id": text["id"],
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": text["full_text"],
                "annotations": [],
            }],
        })
    if not output:
        output.append({
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex}",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "", "annotations": []}],
        })
    return output


def build_non_streaming_response(
    accumulator: dict[str, Any],
    *,
    response_id: str,
    model: str,
) -> dict[str, Any]:
    """Build Responses API format from accumulated OpenAI response."""
    usage = accumulator.get("usage") or {}
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "status": "completed",
        "output": _build_output_items(accumulator),
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


def _build_output_items(accumulator: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    reasoning = accumulator.get("reasoning_content") or ""
    if reasoning:
        reasoning_id = f"rs_{uuid.uuid4().hex}"
        output.append({
            "type": "reasoning",
            "id": reasoning_id,
            "summary": [{"type": "summary_text", "text": reasoning}],
        })
    tool_calls = accumulator.get("tool_calls") or []
    for tool_call in tool_calls:
        call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}"
        function = tool_call.get("function") or {}
        output.append({
            "type": "function_call",
            "id": call_id,
            "call_id": call_id,
            "name": function.get("name", ""),
            "arguments": function.get("arguments", ""),
        })
    content = accumulator.get("content") or ""
    if content or not output:
        output.append({
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex}",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content, "annotations": []}],
        })
    return output


def collect_response_payload(
    *,
    response_id: str,
    model: str,
    request_body: dict[str, Any],
) -> dict[str, Any]:
    """Convenience helper that builds a Responses API payload
    directly from a CompletionAccumulator's final state.
    """
    acc = CompletionAccumulator(model)
    return build_non_streaming_response(
        {
            "content": "",
            "usage": None,
            "tool_calls": [],
            "reasoning_content": "",
            "finish_reason": None,
        },
        response_id=response_id,
        model=model,
    )
