"""Unit tests for freebuff2api.responses_compat streaming and input handling.

Run with: python -m unittest tests.test_responses_compat
"""

import json
import unittest

from freebuff2api.responses_compat import (
    _input_item_to_messages,
    _input_to_messages,
    build_non_streaming_response,
    build_upstream_payload,
    responses_stream_events,
)


def _collapse(item):
    return json.dumps(item, ensure_ascii=False, separators=(",", ":"))


def _feed(chunks):
    state: dict = {}
    events = []
    response_id = "resp_ut"
    for chunk in chunks:
        for ev_type, ev_data in responses_stream_events(
            chunk, response_id=response_id, model="test-model", state=state,
        ):
            events.append((ev_type, ev_data))
    return state, events


class ResponsesCompatInputTests(unittest.TestCase):
    def test_string_input_becomes_user_message(self) -> None:
        messages = _input_to_messages("hello")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("hello", messages[-1]["content"])

    def test_message_item_with_string_content(self) -> None:
        msgs = _input_item_to_messages({"role": "user", "content": "hi"})
        self.assertEqual(msgs, [{"role": "user", "content": "hi"}])

    def test_message_item_with_text_blocks(self) -> None:
        msgs = _input_item_to_messages({
            "role": "user",
            "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
        })
        self.assertEqual(msgs, [{"role": "user", "content": "A\nB"}])

    def test_function_call_input_becomes_assistant_tool_call(self) -> None:
        msgs = _input_item_to_messages({
            "type": "function_call",
            "call_id": "call_42",
            "name": "get_weather",
            "arguments": '{"city": "Tokyo"}',
        })
        self.assertEqual(msgs[0]["role"], "assistant")
        self.assertIsNone(msgs[0]["content"])
        self.assertEqual(msgs[0]["tool_calls"][0]["id"], "call_42")
        self.assertEqual(msgs[0]["tool_calls"][0]["function"]["name"], "get_weather")

    def test_function_call_output_becomes_tool_message(self) -> None:
        msgs = _input_item_to_messages({
            "type": "function_call_output",
            "call_id": "call_42",
            "output": "Sunny, 20C",
        })
        self.assertEqual(msgs[0]["role"], "tool")
        self.assertEqual(msgs[0]["tool_call_id"], "call_42")
        self.assertEqual(msgs[0]["content"], "Sunny, 20C")

    def test_input_array_with_history_and_tool_use(self) -> None:
        history = [
            {"role": "user", "content": "weather in Tokyo?"},
            {"type": "function_call", "call_id": "c1", "name": "get_weather",
             "arguments": '{"city":"Tokyo"}'},
            {"type": "function_call_output", "call_id": "c1", "output": "Sunny 20C"},
            {"role": "user", "content": "and in Osaka?"},
        ]
        msgs = _input_to_messages(history)
        # user, assistant (tool_calls), tool, user
        self.assertGreaterEqual(len(msgs), 4)
        roles = [m["role"] for m in msgs if m.get("role") != "system"]
        self.assertIn("assistant", roles)
        self.assertIn("tool", roles)


class ResponsesCompatStreamTests(unittest.TestCase):
    def test_accumulates_full_text_with_multiple_deltas(self) -> None:
        state, events = _feed([
            {"choices": [{"index": 0, "delta": {"content": "Hi"}}]},
            {"choices": [{"index": 0, "delta": {"content": " there."}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ])
        # find output_text.done
        done = [e for t, e in events if t == "response.output_text.done"]
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["text"], "Hi there.")
        # find response.completed output text
        completed = [e for t, e in events if t == "response.completed"]
        self.assertEqual(len(completed), 1)
        msg_output = completed[0]["response"]["output"][0]
        self.assertEqual(msg_output["content"][0]["text"], "Hi there.")

    def test_emits_function_call_items_for_tool_calls(self) -> None:
        state, events = _feed([
            {"choices": [{"index": 0, "delta": {"content": "Let me check"}}]},
            {"choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "id": "call_abc", "type": "function",
                                "function": {"name": "get_weather", "arguments": ""}}],
            }}]},
            {"choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}],
            }}]},
            {"choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": '"Tokyo"}'}}],
            }}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ])
        completed = next(e for t, e in events if t == "response.completed")
        output = completed["response"]["output"]
        types = [item["type"] for item in output]
        self.assertIn("message", types)
        self.assertIn("function_call", types)
        fc_item = next(item for item in output if item["type"] == "function_call")
        self.assertEqual(fc_item["name"], "get_weather")
        self.assertEqual(fc_item["arguments"], '{"city":"Tokyo"}')

    def test_emits_reasoning_events_when_reasoning_content_present(self) -> None:
        state, events = _feed([
            {"choices": [{"index": 0, "delta": {"reasoning_content": "I will say hi"}}]},
            {"choices": [{"index": 0, "delta": {"content": "Hi"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ])
        types = {t for t, _ in events}
        self.assertIn("response.reasoning_text.delta", types)
        self.assertIn("response.reasoning_text.done", types)
        completed = next(e for t, e in events if t == "response.completed")
        output = completed["response"]["output"]
        self.assertEqual(output[0]["type"], "reasoning")
        self.assertEqual(output[0]["summary"][0]["text"], "I will say hi")
        self.assertEqual(output[1]["content"][0]["text"], "Hi")

    def test_no_output_completed_event_yields_at_least_one_message(self) -> None:
        state, events = _feed([
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ])
        completed = next(e for t, e in events if t == "response.completed")
        # Must emit at least one output item even when upstream had nothing
        self.assertGreaterEqual(len(completed["response"]["output"]), 1)


class ResponsesNonStreamingTests(unittest.TestCase):
    def test_tool_calls_become_function_call_items(self) -> None:
        accumulator = {
            "content": "Calling weather",
            "reasoning_content": "",
            "tool_calls": [{
                "id": "call_xyz",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Beijing"}'},
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            "finish_reason": "tool_calls",
        }
        response = build_non_streaming_response(
            accumulator, response_id="resp_xyz", model="m",
        )
        types = [item["type"] for item in response["output"]]
        self.assertIn("function_call", types)
        self.assertIn("message", types)
        fc = next(item for item in response["output"] if item["type"] == "function_call")
        self.assertEqual(fc["name"], "get_weather")
        self.assertEqual(fc["arguments"], '{"city":"Beijing"}')

    def test_reasoning_becomes_reasoning_item(self) -> None:
        accumulator = {
            "content": "Final answer",
            "reasoning_content": "thinking",
            "tool_calls": [],
            "usage": {},
        }
        response = build_non_streaming_response(
            accumulator, response_id="resp_x", model="m",
        )
        types = [item["type"] for item in response["output"]]
        self.assertIn("reasoning", types)
        rs = next(item for item in response["output"] if item["type"] == "reasoning")
        self.assertEqual(rs["summary"][0]["text"], "thinking")


if __name__ == "__main__":
    unittest.main()
