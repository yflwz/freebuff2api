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


if __name__ == "__main__":
    unittest.main()
