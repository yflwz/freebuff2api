"""Unit tests for freebuff2api.anthropic_compat message and streaming conversion."""

import unittest

from freebuff2api.anthropic_compat import (
    AnthropicStreamState,
    anthropic_stream_events,
    normalize_messages,
    translate_anthropic_tool_choice,
    translate_anthropic_tools,
)


class AnthropicNormalizeMessagesTests(unittest.TestCase):
    def test_plain_string_message(self) -> None:
        msgs = normalize_messages([{"role": "user", "content": "hello"}])
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertEqual(msgs[-1]["content"], "hello")

    def test_tool_use_becomes_assistant_tool_calls(self) -> None:
        msgs = normalize_messages([{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check"},
                {
                    "type": "tool_use",
                    "id": "tu_123",
                    "name": "get_weather",
                    "input": {"city": "Tokyo"},
                },
            ],
        }])
        assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
        self.assertEqual(len(assistant_msgs), 1)
        self.assertEqual(assistant_msgs[0]["content"], "Let me check")
        self.assertEqual(assistant_msgs[0]["tool_calls"][0]["id"], "tu_123")
        self.assertEqual(assistant_msgs[0]["tool_calls"][0]["function"]["name"], "get_weather")

    def test_tool_result_becomes_tool_message(self) -> None:
        msgs = normalize_messages([{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_123", "content": "Sunny"},
            ],
        }])
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["tool_call_id"], "tu_123")
        self.assertEqual(tool_msgs[0]["content"], "Sunny")

    def test_image_block_becomes_image_url(self) -> None:
        msgs = normalize_messages([{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "abc123",
                    },
                },
            ],
        }])
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        self.assertEqual(len(user_msgs), 1)
        content = user_msgs[0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertIn("data:image/png;base64,abc123", content[1]["image_url"]["url"])

    def test_mixed_text_and_tool_result(self) -> None:
        msgs = normalize_messages([{
            "role": "user",
            "content": [
                {"type": "text", "text": "Result:"},
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"},
                {"type": "text", "text": "Thanks"},
            ],
        }])
        roles = [m["role"] for m in msgs]
        self.assertIn("user", roles)
        self.assertIn("tool", roles)


class AnthropicToolTranslationTests(unittest.TestCase):
    def test_translate_tool_schema(self) -> None:
        tools = translate_anthropic_tools([{
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }])
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "get_weather")

    def test_translate_tool_choice(self) -> None:
        self.assertEqual(translate_anthropic_tool_choice("any"), "required")
        self.assertEqual(translate_anthropic_tool_choice("auto"), "auto")
        self.assertEqual(
            translate_anthropic_tool_choice({"type": "tool", "name": "x"}),
            {"type": "function", "function": {"name": "x"}},
        )


class AnthropicMultiTurnTests(unittest.TestCase):
    def test_full_tool_loop_conversation(self) -> None:
        """User asks, assistant calls tool, user gives result, assistant answers."""
        msgs = normalize_messages([
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "tu_weather",
                        "name": "get_weather",
                        "input": {"city": "Tokyo"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_weather",
                        "content": "Sunny, 25C",
                    },
                ],
            },
            {"role": "assistant", "content": "It's sunny in Tokyo."},
        ])
        roles = [m["role"] for m in msgs]
        # normalize_chat_messages prepends a system message; tool_result becomes role="tool"
        self.assertEqual(roles.count("user"), 1)
        self.assertEqual(roles.count("assistant"), 2)
        self.assertEqual(roles.count("tool"), 1)
        tool_msg = next(m for m in msgs if m["role"] == "tool")
        self.assertEqual(tool_msg["tool_call_id"], "tu_weather")
        self.assertEqual(tool_msg["content"], "Sunny, 25C")

    def test_system_as_list(self) -> None:
        msgs = normalize_messages(
            [{"role": "user", "content": "hi"}],
            system=[{"type": "text", "text": "Be helpful."}],
        )
        system_msgs = [m for m in msgs if m["role"] == "system"]
        self.assertEqual(len(system_msgs), 1)
        # normalize_chat_messages prepends the Buffy identity override to system content
        self.assertIn("Be helpful.", system_msgs[0]["content"])

    def test_unsupported_role_coerced_to_user(self) -> None:
        msgs = normalize_messages([{"role": "system", "content": "sys"}])
        # normalize_chat_messages prepends a system message, so the coerced message is last
        self.assertEqual(msgs[-1]["role"], "user")

    def test_empty_messages(self) -> None:
        msgs = normalize_messages([])
        # normalize_chat_messages always prepends the default system message
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "system")


class AnthropicStreamTests(unittest.TestCase):
    def test_stream_emits_message_start_once(self) -> None:
        state = AnthropicStreamState()
        events = anthropic_stream_events(
            {"choices": [{"index": 0, "delta": {"content": "hi"}}]},
            message_id="msg_1",
            model="test-model",
            started=0,
            input_tokens=0,
            state=state,
        )
        types = [t for t, _ in events]
        self.assertIn("message_start", types)
        self.assertEqual(types.count("message_start"), 1)

    def test_stream_text_delta(self) -> None:
        state = AnthropicStreamState()
        events = anthropic_stream_events(
            {"choices": [{"index": 0, "delta": {"content": "hello"}}]},
            message_id="msg_1",
            model="test-model",
            started=0,
            input_tokens=0,
            state=state,
        )
        deltas = [e for t, e in events if t == "content_block_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["delta"]["text"], "hello")

    def test_stream_finish_emits_message_stop(self) -> None:
        state = AnthropicStreamState()
        events = anthropic_stream_events(
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            message_id="msg_1",
            model="test-model",
            started=0,
            input_tokens=0,
            state=state,
        )
        types = [t for t, _ in events]
        self.assertIn("message_delta", types)
        self.assertIn("message_stop", types)

    def test_stream_tool_use_sequence(self) -> None:
        """Tool use should emit content_block_start with tool_use then input_json_delta."""
        state = AnthropicStreamState()
        events = anthropic_stream_events(
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": ""}},
            ]}}]},
            message_id="msg_1",
            model="test-model",
            started=0,
            input_tokens=0,
            state=state,
        )
        events += anthropic_stream_events(
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '{"city": "Tokyo"}'}},
            ]}}]},
            message_id="msg_1",
            model="test-model",
            started=0,
            input_tokens=0,
            state=state,
        )
        events += anthropic_stream_events(
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            message_id="msg_1",
            model="test-model",
            started=0,
            input_tokens=0,
            state=state,
        )
        types = [t for t, _ in events]
        self.assertIn("content_block_start", types)
        self.assertIn("content_block_delta", types)
        self.assertIn("content_block_stop", types)
        self.assertIn("message_delta", types)
        self.assertIn("message_stop", types)

    def test_non_streaming_with_tool_calls(self) -> None:
        from freebuff2api.anthropic_compat import build_non_streaming_response
        resp = build_non_streaming_response(
            {
                "content": "Calling weather",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
                    }
                ],
                "usage": {"completion_tokens": 10},
                "finish_reason": "tool_calls",
            },
            message_id="msg_1",
            model="test-model",
            input_tokens=5,
        )
        self.assertEqual(resp["stop_reason"], "tool_use")
        types = [b["type"] for b in resp["content"]]
        self.assertIn("text", types)
        self.assertIn("tool_use", types)
        tool_use = next(b for b in resp["content"] if b["type"] == "tool_use")
        self.assertEqual(tool_use["input"], {"city": "Tokyo"})


class AnthropicNonStreamingTests(unittest.TestCase):
    def test_reasoning_becomes_thinking_block(self) -> None:
        from freebuff2api.anthropic_compat import build_non_streaming_response
        resp = build_non_streaming_response(
            {
                "content": "The answer is 42.",
                "reasoning_content": "I will compute the answer.",
                "tool_calls": [],
                "usage": {"completion_tokens": 7},
                "finish_reason": "stop",
            },
            message_id="msg_1",
            model="test-model",
            input_tokens=5,
        )
        self.assertEqual(resp["stop_reason"], "end_turn")
        types = [b["type"] for b in resp["content"]]
        self.assertEqual(types, ["thinking", "text"])
        self.assertEqual(resp["content"][0]["thinking"], "I will compute the answer.")
        self.assertEqual(resp["content"][1]["text"], "The answer is 42.")

    def test_reasoning_and_tool_calls_together(self) -> None:
        from freebuff2api.anthropic_compat import build_non_streaming_response
        resp = build_non_streaming_response(
            {
                "content": "",
                "reasoning_content": "I need a tool.",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q":"weather"}'},
                    }
                ],
                "usage": {"completion_tokens": 12},
                "finish_reason": "tool_calls",
            },
            message_id="msg_2",
            model="test-model",
            input_tokens=5,
        )
        self.assertEqual(resp["stop_reason"], "tool_use")
        types = [b["type"] for b in resp["content"]]
        self.assertEqual(types, ["thinking", "tool_use"])
        self.assertEqual(resp["content"][0]["thinking"], "I need a tool.")
        self.assertEqual(resp["content"][1]["name"], "search")
        self.assertEqual(resp["content"][1]["input"], {"q": "weather"})

    def test_empty_content_emits_text_block_when_no_other_blocks(self) -> None:
        from freebuff2api.anthropic_compat import build_non_streaming_response
        resp = build_non_streaming_response(
            {
                "content": "",
                "reasoning_content": "",
                "tool_calls": [],
                "usage": {},
                "finish_reason": "stop",
            },
            message_id="msg_3",
            model="test-model",
            input_tokens=5,
        )
        self.assertEqual(resp["stop_reason"], "end_turn")
        self.assertEqual(len(resp["content"]), 1)
        self.assertEqual(resp["content"][0]["type"], "text")
        self.assertEqual(resp["content"][0]["text"], "")

    def test_usage_and_model_preserved(self) -> None:
        from freebuff2api.anthropic_compat import build_non_streaming_response
        resp = build_non_streaming_response(
            {
                "content": "hi",
                "reasoning_content": "",
                "tool_calls": [],
                "usage": {"completion_tokens": 123},
                "finish_reason": "stop",
            },
            message_id="msg_4",
            model="claude-test",
            input_tokens=42,
        )
        self.assertEqual(resp["model"], "claude-test")
        self.assertEqual(resp["usage"]["input_tokens"], 42)
        self.assertEqual(resp["usage"]["output_tokens"], 123)


if __name__ == "__main__":
    unittest.main()
