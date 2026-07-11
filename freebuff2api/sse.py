from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx


def encode_sse(data: dict[str, Any] | str) -> bytes:
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"data: {payload}\n\n".encode("utf-8")


def decode_sse_data(line: str) -> dict[str, Any] | str | None:
    """Parse a single SSE data line (kept for backward compatibility)."""
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data:
        return None
    return _parse_sse_payload(data)


def _parse_sse_payload(data: str) -> dict[str, Any] | str | None:
    if data == "[DONE]":
        return data
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


async def iter_sse_frames(response: httpx.Response) -> AsyncIterator[bytes]:
    """Yield complete SSE frames bytes from an httpx streaming response.

    Buffers incoming bytes and splits on ``\\n\\n`` or ``\\r\\n\\r\\n`` frame
    delimiters. This keeps multi-byte UTF-8 characters from being split
    across frame boundaries before decoding.
    """
    buffer = b""
    async for chunk in response.aiter_bytes():
        buffer += chunk
        while True:
            pos_lf = buffer.find(b"\n\n")
            pos_crlf = buffer.find(b"\r\n\r\n")
            if pos_lf == -1 and pos_crlf == -1:
                break
            if pos_lf == -1:
                pos, delim_len = pos_crlf, 4
            elif pos_crlf == -1:
                pos, delim_len = pos_lf, 2
            else:
                if pos_lf < pos_crlf:
                    pos, delim_len = pos_lf, 2
                else:
                    pos, delim_len = pos_crlf, 4
            frame = buffer[:pos]
            buffer = buffer[pos + delim_len :]
            if frame:
                yield frame
    if buffer and buffer.strip():
        yield buffer


def parse_sse_frame(frame: bytes) -> dict[str, Any] | str | None:
    """Parse a complete SSE frame into its data payload.

    Handles frames with multiple ``data:`` lines by joining them with newlines.
    Ignores ``event:`` and other fields (this implementation is tailored to
    OpenAI-style SSE where only ``data:`` payloads are meaningful).
    """
    if not frame:
        return None
    text = frame.decode("utf-8", errors="replace")
    data_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return None
    data = "\n".join(data_lines)
    if not data:
        return None
    return _parse_sse_payload(data)
