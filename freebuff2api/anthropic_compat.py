from __future__ import annotations

import json
import logging
from typing import Any

from .models import resolve_model
from .openai_compat import normalize_chat_messages

logger = logging.getLogger("freebuff2api.anthropic_compat")

ANTHROPIC_VERSION = "2023-06-01"

_STOP_REASON_MAP: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "content_filter",
}



def translate_anthropic_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Convert Anthropic tool schema to OpenAI Chat Completions tool format.

    Anthropic: {\"name\": \"x\", \"description\": \"...\", \"input_schema\": {\"type\": \"object\", ...}}
    OpenAI:    {\"type\": \"function\", \"function\": {\"name\": \"x\", \"description\": \"...\", \"parameters\": {...}}}
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


def translate_anthropic_tool_choice(tool_choice: Any) -> Any:
    """Convert Anthropic tool_choice to OpenAI Chat Completions format.

    Anthropic: \"auto\" | \"any\" | \"none\" | {\"type\": \"tool\", \"name\": \"x\"}
    OpenAI:    \"auto\" | \"required\" | \"none\" | {\"type\": \"function\", \"function\": {\"name\": \"x\"}}
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice == "any":
            return "required"
        return tool_choice
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        name = tool_choice.get("name", "")
        if choice_type == "tool" and name:
            return {"type": "function", "function": {"name": name}}
    return tool_choice


def _anthropic_msg_role(role: str) -> str:
    if role == "assistant":
        return "assistant"
    if role in ("user", "human"):
        return "user"
    # Anthropic messages array only contains user/assistant roles; anything else
    # (e.g. system) is passed via the separate `system` parameter.
    logger.debug("coercing unsupported anthropic message role=%s to user", role)
    return "user"


def _extract_text_content(blocks: list[dict[str, Any]]) -> str:
    """Extract text from a list of content blocks."""
    texts = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            texts.append(block.get("text", ""))
    return "".join(texts)


def _convert_image_block(block: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an Anthropic image block to OpenAI image_url format."""
    source = block.get("source", {})
    if source.get("type") == "base64":
        media_type = source.get("media_type", "image/jpeg")
        data = source.get("data", "")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{data}"},
        }
    return None


def _tool_result_to_message(block: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic tool_result block to an OpenAI tool message."""
    content = block.get("content", "")
    if isinstance(content, list):
        content = _extract_text_content(content)
    elif not isinstance(content, str):
        content = str(content)
    if block.get("is_error"):
        content = f"Error: {content}"
    return {
        "role": "tool",
        "tool_call_id": block.get("tool_use_id", ""),
        "content": content,
    }


def normalize_messages(
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert Anthropic messages + system to OpenAI-style messages.

    Handles:
    - text blocks -> string or text content parts
    - tool_use blocks -> assistant message with tool_calls
    - tool_result blocks -> separate role=\"tool\" messages with tool_call_id
    - image blocks -> OpenAI image_url content parts
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
        role = msg.get("role", "user")
        raw_content = msg.get("content", "")

        # Plain string content: map role and keep as-is
        if isinstance(raw_content, str):
            result.append({"role": _anthropic_msg_role(role), "content": raw_content})
            continue

        # Non-list content: stringify and map role
        if not isinstance(raw_content, list):
            result.append({"role": _anthropic_msg_role(role), "content": str(raw_content)})
            continue

        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in raw_content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                        },
                    })
            msg_out: dict[str, Any] = {"role": "assistant"}
            if text_parts or not tool_calls:
                msg_out["content"] = "".join(text_parts)
            if tool_calls:
                msg_out["tool_calls"] = tool_calls
            result.append(msg_out)

        elif role == "user":
            current_content: list[dict[str, Any]] = []
            for block in raw_content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")

                if btype == "tool_result":
                    # Flush any pending user content before the tool message
                    if current_content:
                        if all(c.get("type") == "text" for c in current_content):
                            result.append({"role": "user", "content": "".join(c.get("text", "") for c in current_content)})
                        else:
                            result.append({"role": "user", "content": list(current_content)})
                        current_content = []
                    result.append(_tool_result_to_message(block))
                elif btype == "text":
                    current_content.append({"type": "text", "text": block.get("text", "")})
                elif btype == "image":
                    image_block = _convert_image_block(block)
                    if image_block:
                        current_content.append(image_block)

            # Flush remaining user content
            if current_content:
                if all(c.get("type") == "text" for c in current_content):
                    result.append({"role": "user", "content": "".join(c.get("text", "") for c in current_content)})
                else:
                    result.append({"role": "user", "content": list(current_content)})

        else:
            # Fallback for unsupported roles: flatten to string
            result.append({"role": _anthropic_msg_role(role), "content": str(raw_content)})

    # Reuse OpenAI compat's normalize — adds Buffy prefix + ephemeral cache_control
    return normalize_chat_messages(result)


def model_id(requested: str | None = None) -> str:
    return resolve_model(requested).upstream_id


def encode_anthropic_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


class AnthropicStreamState:
    """Object-oriented state machine for Anthropic streaming translation."""

    def __init__(self) -> None:
        self.message_started: bool = False
        self.text_started: bool = False
        self.text_stopped: bool = False
        self.thinking_started: bool = False

        self.text_block_index: int = -1
        self.thinking_block_index: int = -1

        self.tool_blocks: set[int] = set()
        self.tool_blocks_done: set[int] = set()
        self.tool_block_indices: dict[int, int] = {}

        self.next_block_index: int = 0

    def get_index(self) -> int:
        idx = self.next_block_index
        self.next_block_index += 1
        return idx


def anthropic_stream_events(
    openai_chunk: dict[str, Any],
    *,
    message_id: str,
    model: str,
    started: int,
    input_tokens: int,
    state: AnthropicStreamState,
) -> list[tuple[str, dict[str, Any]]]:
    """Convert one OpenAI-format chunk into Anthropic SSE event/data pairs.

    Uses AnthropicStreamState for robust block tracking across chunks.
    """
    events: list[tuple[str, dict[str, Any]]] = []

    if not state.message_started:
        state.message_started = True
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

        # Reasoning content is not emitted in SSE to maintain compatibility
        # with standard Anthropic clients. It is still available in the
        # non-streaming response as a thinking block.

        if content_text:
            if not state.text_started:
                state.text_started = True
                state.text_block_index = state.get_index()
                events.append(("content_block_start", {
                    "type": "content_block_start",
                    "index": state.text_block_index,
                    "content_block": {"type": "text", "text": ""},
                }))
            events.append(("content_block_delta", {
                "type": "content_block_delta",
                "index": state.text_block_index,
                "delta": {"type": "text_delta", "text": content_text},
            }))

        for tc_delta in tool_calls_delta:
            tc_index = tc_delta.get("index", 0)

            if tc_index not in state.tool_blocks:
                state.tool_blocks.add(tc_index)
                state.tool_block_indices[tc_index] = state.get_index()

                events.append(("content_block_start", {
                    "type": "content_block_start",
                    "index": state.tool_block_indices[tc_index],
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
                    "index": state.tool_block_indices[tc_index],
                    "delta": {"type": "input_json_delta", "partial_json": arg_delta},
                }))

    finish_reason = None
    for choice in openai_chunk.get("choices") or []:
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
            break

    if finish_reason:
        # Close pending blocks
        for tc_index in state.tool_blocks:
            if tc_index not in state.tool_blocks_done:
                state.tool_blocks_done.add(tc_index)
                events.append(("content_block_stop", {
                    "type": "content_block_stop",
                    "index": state.tool_block_indices[tc_index],
                }))

        if state.text_started and not state.text_stopped:
            state.text_stopped = True
            events.append(("content_block_stop", {
                "type": "content_block_stop",
                "index": state.text_block_index,
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

    if content or not (reasoning or tool_calls):
        content_blocks.append({"type": "text", "text": content})

    for tc in tool_calls:
        fn = tc.get("function", {})
        input_val = fn.get("arguments", "{}")
        if isinstance(input_val, str):
            try:
                input_val = json.loads(input_val)
            except Exception:
                input_val = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": fn.get("name", ""),
            "input": input_val,
        })

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
