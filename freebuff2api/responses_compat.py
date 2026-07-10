from __future__ import annotations

import json
import time
import uuid
from typing import Any

from .openai_compat import normalize_chat_messages


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
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                content = "\n".join(text_parts) if text_parts else str(content)
            result.append({"role": role, "content": str(content) if content else ""})
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
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(",", ":"))}\n\n"


def responses_stream_events(
    openai_chunk: dict[str, Any],
    *,
    response_id: str,
    model: str,
    state: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Convert one OpenAI chunk into Responses API SSE events."""
    events: list[tuple[str, dict[str, Any]]] = []

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

    content_text = ""
    for choice in openai_chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        content_text += delta.get("content") or ""

    if content_text:
        item_id = state.get("item_id") or "msg_" + uuid.uuid4().hex
        state.setdefault("item_id", item_id)
        events.append(("response.output_text.delta", {
            "type": "response.output_text.delta",
            "delta": content_text,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
        }))

    finish_reason = None
    for choice in openai_chunk.get("choices") or []:
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
            break

    if finish_reason:
        item_id = state.get("item_id", "")
        usage = openai_chunk.get("usage") or {}
        all_text = state.get("full_text", "") + content_text
        state["full_text"] = all_text
        events.append(("response.output_text.done", {
            "type": "response.output_text.done",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "text": all_text,
        }))
        events.append(("response.completed", {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": int(time.time()),
                "model": model,
                "status": "completed",
                "output": [{
                    "type": "message",
                    "id": item_id,
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": all_text, "annotations": []}],
                }],
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
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
    """Build Responses API format from accumulated OpenAI response."""
    content = accumulator.get("content", "")
    usage = accumulator.get("usage") or {}
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "status": "completed",
        "output": [{
            "type": "message",
            "id": "msg_" + uuid.uuid4().hex,
            "role": "assistant",
            "content": [{"type": "output_text", "text": content, "annotations": []}],
        }],
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }
