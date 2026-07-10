from __future__ import annotations

import time
import uuid
from typing import Any

from .codebuff import FreebuffSession
from .models import resolve_model
from .openai_compat import normalize_chat_messages

ANTHROPIC_VERSION = "2023-06-01"

_STOP_REASON_MAP: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "content_filter",
}



_ANTHROPIC_TOOL_KEYS = frozenset({"name", "description", "input_schema", "type"})

def translate_anthropic_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Convert Anthropic tool schema to OpenAI Chat Completions tool format.

    Anthropic: {"name": "x", "description": "...", "input_schema": {"type": "object", ...}}
    OpenAI:    {"type": "function", "function": {"name": "x", "description": "...", "parameters": {...}}}
    """
    if not tools:
        return tools
    converted = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and "function" in tool and isinstance(tool["function"], dict):
            converted.append(tool)
            continue
        tool_type = tool.get("type", "")
        if tool_type and tool_type != "function":
            continue
        name = tool.get("name", "")
        if not name:
            continue
        description = tool.get("description", "")
        input_schema = tool.get("input_schema", {})
        if not isinstance(input_schema, dict):
            input_schema = {"type": "object", "properties": {}}
        if not input_schema.get("type"):
            input_schema = dict(input_schema)
            input_schema["type"] = "object"
        converted.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": input_schema,
            },
        })
    return converted if converted else None

def _anthropic_msg_role(role: str) -> str:
    return "assistant" if role == "assistant" else "user"


def _anthropic_content_to_str(raw: Any) -> str:
    """Convert Anthropic content (string or list of blocks) to a plain string."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = []
        for block in raw:
            if not isinstance(block, dict):
                continue
            t = block.get("type")
            if t == "text":
                parts.append(block.get("text", ""))
            elif t == "tool_use":
                parts.append(
                    f"[tool_use: {block.get('name', '')} "
                    f"id={block.get('id', '')} "
                    f"input={block.get('input', {})}]"
                )
            elif t == "tool_result":
                tool_content = block.get("content", "")
                if isinstance(tool_content, list):
                    tool_texts = [
                        b.get("text", "") for b in tool_content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    tool_content = "\n".join(tool_texts)
                parts.append(f"[tool_result: {tool_content}]")
            elif t == "image":
                parts.append("[image]")
        return "\n".join(parts) if parts else ""
    return str(raw) if raw else ""


def normalize_messages(
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert Anthropic messages + system to OpenAI-style messages.

    Reuses openai_compat.normalize_chat_messages for the final pass
    so the Buffy system override and cache_control are applied consistently.
    """
    result: list[dict[str, Any]] = []

    if system:
        if isinstance(system, str):
            result.append({"role": "system", "content": system})
        elif isinstance(system, list):
            texts = [
                b.get("text", "") for b in system
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            result.append({"role": "system", "content": "\n".join(texts)})

    for msg in messages or []:
        role = _anthropic_msg_role(msg.get("role", "user"))
        content = _anthropic_content_to_str(msg.get("content", ""))
        result.append({"role": role, "content": content})

    # Reuse OpenAI compat's normalize — adds Buffy prefix + ephemeral cache_control
    return normalize_chat_messages(result)


def model_id(requested: str | None = None) -> str:
    return resolve_model(requested).upstream_id


def encode_anthropic_event(event: str, data: dict[str, Any]) -> str:
    import json
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(",", ":"))}\n\n"


def anthropic_stream_events(
    openai_chunk: dict[str, Any],
    *,
    message_id: str,
    model: str,
    started: int,
    input_tokens: int,
    state: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Convert one OpenAI-format chunk into Anthropic SSE event/data pairs.

    `state` dict tracks streaming progress between calls.
    """
    events: list[tuple[str, dict[str, Any]]] = []

    if not state.get("message_started"):
        state["message_started"] = True
        events.append(("message_start", {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        }))

    for choice in openai_chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        content_text = delta.get("content") or ""
        reasoning_text = delta.get("reasoning_content") or ""
        tool_calls_delta = delta.get("tool_calls") or []

        if content_text:
            if not state.get("text_block_index"):
                state["text_block_index"] = len(state)
                if not state.get("text_started"):
                    state["text_started"] = True
                    events.append(("content_block_start", {
                        "type": "content_block_start",
                        "index": state["text_block_index"],
                        "content_block": {"type": "text", "text": ""},
                    }))
                events.append(("content_block_delta", {
                    "type": "content_block_delta",
                    "index": state["text_block_index"],
                    "delta": {"type": "text_delta", "text": content_text},
                }))

        if reasoning_text:
            if not state.get("thinking_started"):
                state["thinking_started"] = True
                state["thinking_block_index"] = len(state) + 1
                # Anthropic doesn't have a standard SSE event for thinking,
                # but many clients accept raw content in message_delta.
                # We append thinking to a buffer tracked in state.
            state.setdefault("thinking_buffer", "")
            state["thinking_buffer"] += reasoning_text

        for tc_delta in tool_calls_delta:
            tc_index = tc_delta.get("index", 0)
            tc_key = f"tool_block_{tc_index}"
            if tc_key not in state:
                state[tc_key] = True
                events.append(("content_block_start", {
                    "type": "content_block_start",
                    "index": tc_index + 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": tc_delta.get("id", ""),
                        "name": tc_delta.get("function", {}).get("name", ""),
                        "input": {},
                    },
                }))
            fn = tc_delta.get("function", {})
            arg_delta = fn.get("arguments", "")
            if arg_delta:
                events.append(("content_block_delta", {
                    "type": "content_block_delta",
                    "index": tc_index + 1,
                    "delta": {"type": "input_json_delta", "partial_json": arg_delta},
                }))

    finish_reason = None
    for choice in openai_chunk.get("choices") or []:
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
            break

    if finish_reason:
        # Close any open content blocks
        for key in list(state.keys()):
            if key.startswith("tool_block_") and not state.get(f"{key}_done"):
                state[f"{key}_done"] = True
                idx = int(key.split("_")[-1])
                events.append(("content_block_stop", {
                    "type": "content_block_stop",
                    "index": idx + 1,
                }))
        if state.get("text_started") and not state.get("text_stopped"):
            state["text_stopped"] = True
            events.append(("content_block_stop", {
                "type": "content_block_stop",
                "index": state.get("text_block_index", 0),
            }))

        anthropic_reason = _STOP_REASON_MAP.get(finish_reason, finish_reason)
        usage = openai_chunk.get("usage") or {}
        output_tokens = usage.get("completion_tokens", 0)
        events.append(("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": anthropic_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        }))
        events.append(("message_stop", {"type": "message_stop"}))

    return events

def build_non_streaming_response(
    accumulator: dict[str, Any],
    *,
    message_id: str,
    model: str,
    started: int,
    input_tokens: int,
) -> dict[str, Any]:
    """Build an Anthropic-format non-streaming response."""
    content = accumulator.get("content", "")
    reasoning = accumulator.get("reasoning_content", "")
    tool_calls: list[dict[str, Any]] = accumulator.get("tool_calls") or []
    finish_reason = accumulator.get("finish_reason", "stop") or "stop"
    usage = accumulator.get("usage") or {}
    output_tokens = usage.get("completion_tokens", 0)

    content_blocks: list[dict[str, Any]] = []
    if reasoning:
        content_blocks.append({"type": "thinking", "thinking": reasoning})
    for tc in tool_calls:
        fn = tc.get("function", {})
        input_val = fn.get("arguments", "{}")
        if isinstance(input_val, str):
            import json as _json
            try:
                input_val = _json.loads(input_val)
            except Exception:
                input_val = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": fn.get("name", ""),
            "input": input_val,
        })
    if content or not (reasoning or tool_calls):
        content_blocks.append({"type": "text", "text": content})

    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": _STOP_REASON_MAP.get(finish_reason, finish_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }
