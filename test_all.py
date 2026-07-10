"""Comprehensive endpoint tests for freebuff2api."""
import http.client
import json
import sys
import time

KEY = "sk-yflwz3210"
MODEL = "deepseek/deepseek-v4-flash"
HOST = "127.0.0.1"
PORT = 8000

passed = 0
failed = 0


def do_get(path, timeout=60):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    conn.request("GET", path, headers={"Authorization": f"Bearer {KEY}"})
    resp = conn.getresponse()
    raw = resp.read().decode()
    conn.close()
    return resp.status, raw


def do_post(path, body, extra_headers=None, timeout=120):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    h = {"Content-Type": "application/json"}
    if extra_headers:
        h.update(extra_headers)
    if path.startswith("/v1/chat") or path.startswith("/v1/responses"):
        h["Authorization"] = f"Bearer {KEY}"
    conn.request("POST", path, json.dumps(body), headers=h)
    resp = conn.getresponse()
    raw = resp.read().decode()
    conn.close()
    return resp.status, raw


def check(name, status, raw, fn=None):
    global passed, failed
    if not (200 <= status < 300):
        print(f"  FAIL: {name} -> {status}: {raw[:300]}")
        failed += 1
        return None
    if fn is None:
        print(f"  PASS: {name}")
        passed += 1
        return None
    try:
        fn(raw)  # pass raw string; fn decides parsing
        print(f"  PASS: {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {name} -> {status}: {e!s} | {raw[:300]}")
        failed += 1


# ===== 1. healthz (GET) =====
s, r = do_get("/healthz")
check("healthz", s, r, lambda r: json.loads(r))

# ===== 2. list models (GET) =====
s, r = do_get("/v1/models")
check("list_models", s, r, lambda r: f"{len(json.loads(r)['data'])} models")

# ===== 3. chat/completions non-streaming =====
s, r = do_post("/v1/chat/completions", {
    "model": MODEL, "stream": False,
    "messages": [{"role": "user", "content": "Say hi in one word."}],
})
check("chat_nonstream", s, r, lambda r: json.loads(r)["choices"][0]["message"]["content"])

# ===== 4. chat/completions streaming =====
s, r = do_post("/v1/chat/completions", {
    "model": MODEL, "stream": True,
    "messages": [{"role": "user", "content": "Say hi."}],
})
def chat_stream_check(raw):
    lines = [l for l in raw.split("\n") if l.startswith("data:") and "[DONE]" not in l]
    assert len(lines) >= 2, f"only {len(lines)} SSE lines"
    # last non-DONE SSE should have finish_reason
    last = json.loads(lines[-1][5:].strip())
    assert last["choices"][0].get("finish_reason"), "no finish_reason"
check("chat_stream", s, r, chat_stream_check)

# ===== 5. chat/completions with tools =====
s, r = do_post("/v1/chat/completions", {
    "model": MODEL, "stream": False, "tool_choice": "required",
    "messages": [{"role": "user", "content": "Weather in Tokyo?"}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
})
def chat_tools_check(raw):
    obj = json.loads(raw)
    tc = obj["choices"][0]["message"].get("tool_calls")
    assert tc, "no tool_calls"
    assert tc[0]["function"]["name"] == "get_weather"
check("chat_tools", s, r, chat_tools_check)

# ===== 6. chat/completions multi-turn with tool history =====
s, r = do_post("/v1/chat/completions", {
    "model": MODEL, "stream": False,
    "messages": [
        {"role": "user", "content": "Weather in Osaka?"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Osaka"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 25C"},
        {"role": "user", "content": "Summarize."},
    ],
})
check("chat_multiturn", s, r, lambda r: json.loads(r)["choices"][0]["message"]["content"])

# ===== 7. Anthropic /v1/messages non-streaming =====
s, r = do_post("/v1/messages", {
    "model": MODEL, "max_tokens": 64, "stream": False,
    "messages": [{"role": "user", "content": "Say hi in 3 words."}],
}, extra_headers={"x-api-key": KEY, "anthropic-version": "2023-06-01"})
def messages_check(raw):
    obj = json.loads(raw)
    content = obj.get("content", [])
    assert content, "no content"
    assert content[0]["type"] in ("text", "thinking")
check("messages_nonstream", s, r, messages_check)

# ===== 8. Anthropic /v1/messages streaming =====
s, r = do_post("/v1/messages", {
    "model": MODEL, "max_tokens": 64, "stream": True,
    "messages": [{"role": "user", "content": "Say hi."}],
}, extra_headers={"x-api-key": KEY, "anthropic-version": "2023-06-01"})
def messages_stream_check(raw):
    events = [l for l in raw.split("\n") if l.startswith("event:")]
    assert len(events) >= 3, f"only {len(events)} events"
    assert "event: message_start" in raw
    assert "event: message_stop" in raw or "event: content_block_stop" in raw
check("messages_stream", s, r, messages_stream_check)

# ===== 9. Anthropic /v1/messages with tools (no tool_choice, let model decide) =====
s, r = do_post("/v1/messages", {
    "model": MODEL, "max_tokens": 256, "stream": False,
    "messages": [{"role": "user", "content": "Use get_weather to check Tokyo weather."}],
    "tools": [{"name": "get_weather", "description": "Get current weather for a city", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}],
}, extra_headers={"x-api-key": KEY, "anthropic-version": "2023-06-01"})
def messages_tools_check(raw):
    obj = json.loads(raw)
    content = obj.get("content", [])
    assert content, "no content"
    types = [b["type"] for b in content]
    assert "tool_use" in types or "text" in types, f"unexpected content types: {types}"
    print(f" [content types: {types}]", end="")
check("messages_tools", s, r, messages_tools_check)

# ===== 10. Responses /v1/responses non-streaming =====
s, r = do_post("/v1/responses", {
    "model": MODEL, "stream": False,
    "input": "Say hi in one word.",
})
def resp_check(raw):
    obj = json.loads(raw)
    output = obj.get("output", [])
    assert output, "no output"
    msg = [i for i in output if i["type"] == "message"]
    assert msg, f"no message in output: {[i['type'] for i in output]}"
check("responses_nonstream", s, r, resp_check)

# ===== 11. Responses /v1/responses streaming =====
s, r = do_post("/v1/responses", {
    "model": MODEL, "stream": True,
    "input": "Count 1 to 3.",
})
def resp_stream_check(raw):
    lines = [l for l in raw.split("\n") if l.startswith("data:") and "[DONE]" not in l]
    assert len(lines) >= 5, f"only {len(lines)} SSE lines"
    # should have response.completed at the end
    has_completed = any("response.completed" in l for l in lines)
    assert has_completed, "no response.completed event"
check("responses_stream", s, r, resp_stream_check)

# ===== 12. Responses /v1/responses with tools =====
s, r = do_post("/v1/responses", {
    "model": MODEL, "stream": False, "tool_choice": "required",
    "input": "Weather in Tokyo?",
    "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}],
})
def resp_tools_check(raw):
    obj = json.loads(raw)
    output = obj.get("output", [])
    fc = [i for i in output if i["type"] == "function_call"]
    assert fc, f"no function_call: {[i['type'] for i in output]}"
check("responses_tools", s, r, resp_tools_check)

# ===== 13. Responses multi-turn with function_call history =====
s, r = do_post("/v1/responses", {
    "model": MODEL, "stream": False,
    "input": [
        {"type": "message", "role": "user", "content": "Weather in Osaka?"},
        {"type": "function_call", "call_id": "fc1", "name": "get_weather", "arguments": '{"city": "Osaka"}'},
        {"type": "function_call_output", "call_id": "fc1", "output": "Sunny, 25C"},
        {"type": "message", "role": "user", "content": "Summarize."},
    ],
})
check("responses_history", s, r, lambda r: json.loads(r)["output"])

# ===== 14. Responses with duplicate function_call_output (dedup should work) =====
s, r = do_post("/v1/responses", {
    "model": MODEL, "stream": False,
    "input": [
        {"type": "function_call", "call_id": "d1", "name": "f", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "d1", "output": "ok"},
        {"type": "message", "role": "user", "content": "Next."},
    ],
})
check("responses_dedup_history", s, r, lambda r: json.loads(r)["output"])

# ===== Summary =====
print(f"\n=== RESULTS: {passed} passed, {failed} failed ===")
sys.exit(0 if failed == 0 else 1)
