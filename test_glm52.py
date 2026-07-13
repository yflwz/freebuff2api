"""End-to-end test for z-ai/glm-5.2 via freebuff2api."""
import http.client
import json
import sys
import time

KEY = "sk-yflwz3210"
MODEL = "z-ai/glm-5.2"
HOST = "127.0.0.1"
PORT = 8000

passed = 0
failed = 0


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
        fn(raw)
        print(f"  PASS: {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {name} -> {status}: {e!s} | {raw[:300]}")
        failed += 1


print(f"Testing model: {MODEL}")

# ===== 1. chat/completions non-streaming =====
s, r = do_post("/v1/chat/completions", {
    "model": MODEL, "stream": False,
    "messages": [{"role": "user", "content": "Say hi in one word."}],
})
check("glm52_chat_nonstream", s, r, lambda r: json.loads(r)["choices"][0]["message"]["content"])

# ===== 2. chat/completions streaming =====
s, r = do_post("/v1/chat/completions", {
    "model": MODEL, "stream": True,
    "messages": [{"role": "user", "content": "Say hi."}],
})
def chat_stream_check(raw):
    lines = [l for l in raw.split("\n") if l.startswith("data:") and "[DONE]" not in l]
    assert len(lines) >= 2, f"only {len(lines)} SSE lines"
    last = json.loads(lines[-1][5:].strip())
    assert last["choices"][0].get("finish_reason"), "no finish_reason"
check("glm52_chat_stream", s, r, chat_stream_check)

# ===== 3. chat/completions with tools =====
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
check("glm52_chat_tools", s, r, chat_tools_check)

# ===== 4. Responses non-streaming =====
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
check("glm52_responses_nonstream", s, r, resp_check)

# ===== 5. Responses streaming =====
s, r = do_post("/v1/responses", {
    "model": MODEL, "stream": True,
    "input": "Count 1 to 3.",
})
def resp_stream_check(raw):
    lines = [l for l in raw.split("\n") if l.startswith("data:") and "[DONE]" not in l]
    assert len(lines) >= 5, f"only {len(lines)} SSE lines"
    has_completed = any("response.completed" in l for l in lines)
    assert has_completed, "no response.completed event"
check("glm52_responses_stream", s, r, resp_stream_check)

# ===== Summary =====
print(f"\n=== GLM 5.2 RESULTS: {passed} passed, {failed} failed ===")
sys.exit(0 if failed == 0 else 1)
