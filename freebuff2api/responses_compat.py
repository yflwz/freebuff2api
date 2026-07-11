from __future__ import annotations

import dataclasses
import json
import logging
import time
import uuid
from typing import Any

from .openai_compat import (
    CompletionAccumulator,
    normalize_chat_messages,
)


logger = logging.getLogger("freebuff2api.responses_compat")


def _input_item_to_messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one Responses API input item to zero or more Chat messages."""
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
        call_id = item.get("call_id") or item.get("id") or _new_call_id()
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
        call_id = item.get("call_id") or item.get("id") or _new_call_id()
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
    """Convert Responses API input + instructions to OpenAI-style messages.

    Processes items sequentially. Dedup duplicates using tracking set.
    Tool outputs that arrive before their matching function_call are deferred
    until the matching call is seen, so the assistant/tool message ordering
    required by OpenAI Chat Completions is preserved.
    """
    result: list[dict[str, Any]] = []
    if instructions:
        result.append({"role": "system", "content": instructions})
    if isinstance(input_data, str):
        result.append({"role": "user", "content": input_data})
        return normalize_chat_messages(result)

    input_list = input_data if isinstance(input_data, list) else []
    seen: set[str] = set()
    seen_calls: set[str] = set()
    deferred_tools: dict[str, dict[str, Any]] = {}

    for item in input_list:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "message")
        call_id = item.get("call_id") or item.get("id")

        if call_id and item_type in ("function_call", "function_call_output"):
            dedup_key = f"{item_type}_{call_id}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

        if item_type == "function_call_output":
            output_call_id = item.get("call_id") or item.get("id")
            if output_call_id and output_call_id not in seen_calls:
                deferred_tools[output_call_id] = item
                continue

        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id")
            if call_id:
                seen_calls.add(call_id)

        result.extend(_input_item_to_messages(item))

        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id")
            if call_id and call_id in deferred_tools:
                result.extend(_input_item_to_messages(deferred_tools.pop(call_id)))

    # Any tool outputs whose function_call never appeared are invalid input;
    # skip them rather than emitting a placeholder that upstream may reject.
    if deferred_tools:
        logger.warning(
            "responses input contains function_call_output without matching function_call: %s",
            list(deferred_tools.keys()),
        )

    return normalize_chat_messages(result)


def normalize_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Responses API tools to strict Chat Completions format."""
    converted = []
    for tool in tools:
        tool_type = tool.get("type", "")
        if tool_type not in ("function", "custom"):
            continue

        fn = tool.get("function") or tool
        parameters = fn.get("parameters")

        if not isinstance(parameters, dict) or "type" not in parameters:
            parameters = {"type": "object", "properties": {}}

        fn_fields = {
            "name": fn.get("name", ""),
            "parameters": parameters,
        }
        for k in ("description", "strict"):
            if k in fn:
                fn_fields[k] = fn[k]

        converted.append({"type": "function", "function": fn_fields})
    return converted


def build_upstream_payload(
    body: dict[str, Any],
    *,
    session: Any,
    run_id: str,
    client_id: str,
    trace_session_id: str | None = None,
    upstream_model_id: str | None = None,
) -> dict[str, Any]:
    from .openai_compat import _UPSTREAM_CHAT_KEYS, model_id as oai_model_id

    messages = _input_to_messages(body.get("input"), body.get("instructions"))
    payload = {
        key: body[key] for key in _UPSTREAM_CHAT_KEYS
        if key in body and body[key] is not None
    }

    if "tools" in payload:
        payload["tools"] = normalize_tools(payload["tools"])

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


@dataclasses.dataclass
class StreamItemState:
    id: str
    output_index: int
    added: bool = False
    full_text: str = ""
    name: str = ""
    arguments: str = ""


@dataclasses.dataclass
class ResponsesStreamState:
    created: bool = False
    final_emitted: bool = False
    next_output_index: int = 0
    text_item: StreamItemState | None = None
    reasoning_item: StreamItemState | None = None
    tool_items: dict[int, StreamItemState] = dataclasses.field(default_factory=dict)

    def ensure_text_item(self) -> StreamItemState:
        if not self.text_item:
            self.text_item = StreamItemState(
                id=f"msg_{uuid.uuid4().hex}", output_index=self.next_output_index
            )
            self.next_output_index += 1
        return self.text_item

    def ensure_reasoning_item(self) -> StreamItemState:
        if not self.reasoning_item:
            self.reasoning_item = StreamItemState(
                id=f"rs_{uuid.uuid4().hex}", output_index=self.next_output_index
            )
            self.next_output_index += 1
        return self.reasoning_item

    def ensure_call_item(self, index: int) -> StreamItemState:
        if index not in self.tool_items:
            self.tool_items[index] = StreamItemState(
                id=f"fc_{uuid.uuid4().hex}", output_index=self.next_output_index
            )
            self.next_output_index += 1
        return self.tool_items[index]

    def snapshot_output(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        if self.reasoning_item and self.reasoning_item.added:
            output.append({
                "type": "reasoning",
                "id": self.reasoning_item.id,
                "summary": [{"type": "summary_text", "text": self.reasoning_item.full_text}],
            })
        for index in sorted(self.tool_items):
            item = self.tool_items[index]
            if not item.added:
                continue
            output.append({
                "type": "function_call",
                "id": item.id,
                "call_id": item.id,
                "name": item.name,
                "arguments": item.arguments,
            })
        if self.text_item and (self.text_item.added or self.text_item.full_text):
            output.append({
                "type": "message",
                "id": self.text_item.id,
                "role": "assistant",
                "content": [{"type": "output_text", "text": self.text_item.full_text, "annotations": []}],
            })
        if not output:
            output.append({
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "", "annotations": []}],
            })
        return output


def responses_stream_events(
    openai_chunk: dict[str, Any],
    *,
    response_id: str,
    model: str,
    state: ResponsesStreamState,
) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []

    if not state.created:
        state.created = True
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
            item = state.ensure_text_item()
            item.full_text += content_piece
            if not item.added:
                item.added = True
                events.append(("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": item.output_index,
                    "item": {
                        "id": item.id,
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                }))
                events.append(("response.content_part.added", {
                    "type": "response.content_part.added",
                    "item_id": item.id,
                    "output_index": item.output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                }))
            events.append(("response.output_text.delta", {
                "type": "response.output_text.delta",
                "item_id": item.id,
                "output_index": item.output_index,
                "content_index": 0,
                "delta": content_piece,
            }))

        reasoning_piece = delta.get("reasoning_content")
        if reasoning_piece:
            item = state.ensure_reasoning_item()
            item.full_text += reasoning_piece
            if not item.added:
                item.added = True
                events.append(("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": item.output_index,
                    "item": {
                        "id": item.id,
                        "type": "reasoning",
                        "summary": [],
                    },
                }))
            events.append(("response.reasoning_text.delta", {
                "type": "response.reasoning_text.delta",
                "item_id": item.id,
                "output_index": item.output_index,
                "content_index": 0,
                "delta": reasoning_piece,
            }))

        for tool_call in delta.get("tool_calls") or []:
            tc_index = int(tool_call.get("index", 0))
            item = state.ensure_call_item(tc_index)
            if tool_call.get("id"):
                item.id = tool_call["id"]
            function = tool_call.get("function") or {}
            if function.get("name"):
                item.name += function["name"]
            arguments_chunk = function.get("arguments")
            if arguments_chunk:
                item.arguments += arguments_chunk
            if not item.added:
                item.added = True
                events.append(("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": item.output_index,
                    "item": {
                        "id": item.id,
                        "type": "function_call",
                        "status": "in_progress",
                        "call_id": item.id,
                        "name": item.name,
                        "arguments": item.arguments,
                    },
                }))
            if arguments_chunk:
                events.append(("response.function_call_arguments.delta", {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item.id,
                    "output_index": item.output_index,
                    "delta": arguments_chunk,
                }))

    usage = openai_chunk.get("usage") or usage

    if finish_reason and not state.final_emitted:
        state.final_emitted = True
        completed_output: list[dict[str, Any]] = []

        reasoning = state.reasoning_item
        if reasoning and reasoning.added:
            events.append(("response.reasoning_text.done", {
                "type": "response.reasoning_text.done",
                "item_id": reasoning.id,
                "output_index": reasoning.output_index,
                "content_index": 0,
                "text": reasoning.full_text,
            }))
            events.append(("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": reasoning.output_index,
                "item": {
                    "id": reasoning.id,
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": reasoning.full_text}],
                },
            }))
            completed_output.append({
                "type": "reasoning",
                "id": reasoning.id,
                "summary": [{"type": "summary_text", "text": reasoning.full_text}],
            })

        for tc_index in sorted(state.tool_items):
            item = state.tool_items[tc_index]
            if not item.added:
                continue
            events.append(("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "item_id": item.id,
                "output_index": item.output_index,
                "arguments": item.arguments,
            }))
            events.append(("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": item.output_index,
                "item": {
                    "id": item.id,
                    "type": "function_call",
                    "status": "completed",
                    "call_id": item.id,
                    "name": item.name,
                    "arguments": item.arguments,
                },
            }))
            completed_output.append({
                "type": "function_call",
                "id": item.id,
                "call_id": item.id,
                "name": item.name,
                "arguments": item.arguments,
            })

        text_item = state.text_item
        if text_item and (text_item.added or text_item.full_text):
            if not text_item.added:
                text_item.added = True
                events.append(("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": text_item.output_index,
                    "item": {
                        "id": text_item.id,
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                }))
                events.append(("response.content_part.added", {
                    "type": "response.content_part.added",
                    "item_id": text_item.id,
                    "output_index": text_item.output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                }))
            events.append(("response.output_text.done", {
                "type": "response.output_text.done",
                "item_id": text_item.id,
                "output_index": text_item.output_index,
                "content_index": 0,
                "text": text_item.full_text,
            }))
            events.append(("response.content_part.done", {
                "type": "response.content_part.done",
                "item_id": text_item.id,
                "output_index": text_item.output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": text_item.full_text, "annotations": []},
            }))
            events.append(("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": text_item.output_index,
                "item": {
                    "id": text_item.id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text_item.full_text, "annotations": []}],
                },
            }))
            completed_output.append({
                "type": "message",
                "id": text_item.id,
                "role": "assistant",
                "content": [{"type": "output_text", "text": text_item.full_text, "annotations": []}],
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
                "output": completed_output or state.snapshot_output(),
                "usage": {
                    "input_tokens": (usage or {}).get("prompt_tokens", 0),
                    "output_tokens": (usage or {}).get("completion_tokens", 0),
                    "total_tokens": (usage or {}).get("total_tokens", 0),
                },
            },
        }))

    return events


def build_non_streaming_response(
    accumulator: dict[str, Any],
    *,
    response_id: str,
    model: str,
) -> dict[str, Any]:
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
        call_id = tool_call.get("id") or _new_call_id()
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
    """Convenience helper that builds a Responses API payload directly."""
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
