"""Unit tests for freebuff2api.sse byte-buffered parsing."""

import unittest
from unittest.mock import AsyncMock

import httpx

from freebuff2api.sse import encode_sse, iter_sse_frames, parse_sse_frame


class SSETests(unittest.IsolatedAsyncioTestCase):
    async def _make_response(self, chunks: list[bytes]) -> httpx.Response:
        response = AsyncMock(spec=httpx.Response)
        response.aiter_bytes = lambda: self._async_iter(chunks)
        return response

    async def _async_iter(self, chunks: list[bytes]):
        for chunk in chunks:
            yield chunk

    async def test_iter_sse_frames_splits_on_double_newline(self) -> None:
        response = await self._make_response([
            b"data: {\"id\":\"1\"}\n\n",
            b"data: {\"id\":\"2\"}\n\n",
        ])
        frames = [frame async for frame in iter_sse_frames(response)]
        self.assertEqual(frames, [b"data: {\"id\":\"1\"}", b"data: {\"id\":\"2\"}"])

    async def test_iter_sse_frames_handles_split_multibyte_character(self) -> None:
        # UTF-8 encoding of 中 is b"\xe4\xb8\xad"; split across chunks
        chunks = [b"data: \xe4", b"\xb8\xad\n\n"]
        response = await self._make_response(chunks)
        frames = [frame async for frame in iter_sse_frames(response)]
        self.assertEqual(frames, [b"data: \xe4\xb8\xad"])

    async def test_iter_sse_frames_handles_crlf_delimiter(self) -> None:
        response = await self._make_response([b"data: hello\r\n\r\n"])
        frames = [frame async for frame in iter_sse_frames(response)]
        self.assertEqual(frames, [b"data: hello"])

    async def test_iter_sse_frames_yields_done(self) -> None:
        response = await self._make_response([b"data: [DONE]\n\n"])
        frames = [frame async for frame in iter_sse_frames(response)]
        self.assertEqual(frames, [b"data: [DONE]"])

    async def test_iter_sse_frames_ignores_trailing_whitespace(self) -> None:
        response = await self._make_response([b"data: hello\n\n   "])
        frames = [frame async for frame in iter_sse_frames(response)]
        self.assertEqual(frames, [b"data: hello"])

    async def test_parse_sse_frame_joins_multiple_data_lines(self) -> None:
        frame = b"data: {\"a\":\n\ndata: 1}\n\n"
        data = parse_sse_frame(frame)
        self.assertEqual(data, {"a": 1})

    async def test_parse_sse_frame_ignores_event_field(self) -> None:
        frame = b"event: message\ndata: {\"id\":\"1\"}"
        data = parse_sse_frame(frame)
        self.assertEqual(data, {"id": "1"})

    async def test_parse_sse_frame_extracts_json(self) -> None:
        frame = b"data: {\"id\":\"1\"}"
        data = parse_sse_frame(frame)
        self.assertEqual(data, {"id": "1"})

    def test_encode_sse(self) -> None:
        encoded = encode_sse({"id": "1"})
        self.assertIn(b"data: {\"id\":\"1\"}", encoded)

    def test_parse_sse_frame_returns_none_for_empty(self) -> None:
        self.assertIsNone(parse_sse_frame(b""))
        self.assertIsNone(parse_sse_frame(b"event: ping"))


if __name__ == "__main__":
    unittest.main()
