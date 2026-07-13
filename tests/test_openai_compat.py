import unittest

from freebuff2api.codebuff import FreebuffSession
from freebuff2api.models import (
    ALL_MODELS,
    CONTEXT_PRUNER_AGENT_ID,
    GEMINI_AGENT_ID,
    FreebuffModel,
    agent_validation_payload,
    get_active_models,
    models_response,
    resolve_model,
    set_dynamic_models,
)
from freebuff2api.openai_compat import (
    CompletionAccumulator,
    build_upstream_payload,
    sanitize_stream_chunk,
)


class OpenAICompatTests(unittest.TestCase):
    def test_models_response_lists_all_freebuff_models(self) -> None:
        response = models_response()

        self.assertEqual(
            [item["id"] for item in response["data"]],
            [model.id for model in ALL_MODELS],
        )

    def test_models_response_includes_display_name(self) -> None:
        response = models_response()
        display_names = {item.get("id"): item.get("display_name") for item in response["data"]}

        expected = {
            "deepseek/deepseek-v4-flash": "DeepSeek V4 Flash",
            "deepseek/deepseek-v4-pro": "DeepSeek V4 Pro",
            "moonshotai/kimi-k2.7-code": "Kimi K2.7 Code",
            "minimax/minimax-m3": "MiniMax M3",
            "mimo/mimo-v2.5": "MiMo 2.5",
            "mimo/mimo-v2.5-pro": "MiMo 2.5 Pro",
            "kwaipilot/kat-coder-pro-v2": "KAT Coder Pro V2",
            "z-ai/glm-5.2": "GLM 5.2",
            "tencent/hy3:free": "Hunyuan 3",
            "google/gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
            "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite Preview",
            "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
        }
        for model_id, expected_name in expected.items():
            self.assertEqual(display_names.get(model_id), expected_name)

    def test_derive_display_name_handles_dotted_tencent_variant(self) -> None:
        from freebuff2api.models import derive_display_name

        self.assertEqual(derive_display_name("z-ai/glm-5.2"), "GLM 5.2")
        self.assertEqual(derive_display_name("tencent/hy3"), "Hunyuan 3")
        self.assertEqual(derive_display_name("tencent/hy3:free"), "Hunyuan 3 Free")
        self.assertEqual(derive_display_name("tencent/hy3.free"), "Hunyuan 3 Free")

    def test_resolve_model_maps_agent_id(self) -> None:
        model = resolve_model("moonshotai/kimi-k2.7-code")

        self.assertEqual(model.agent_id, "base2-free-kimi")

    def test_resolve_minimax_m3_maps_har_agent_id(self) -> None:
        model = resolve_model("minimax/minimax-m3")

        self.assertEqual(model.agent_id, "base2-free-minimax-m3")

    def test_resolve_glm_5_2_maps_agent_id(self) -> None:
        model = resolve_model("z-ai/glm-5.2")

        self.assertEqual(model.agent_id, "base2-free")
        self.assertEqual(model.display_name, "GLM 5.2")

    def test_resolve_gemini_pro_routes_via_mimo(self) -> None:
        # Option B: gemini-pro is a thin alias for mimo/mimo-v2.5. Same
        # session quota pool, same chat dispatch agent, no parent lineage.
        # This avoids the upstream's session_model_mismatch 409 (chat
        # payload's model field must equal the bound session model's id).
        model = resolve_model("google/gemini-3.1-pro-preview")

        self.assertEqual(model.agent_id, GEMINI_AGENT_ID)
        self.assertIsNone(model.parent_agent_id)
        self.assertEqual(model.session_id, "mimo/mimo-v2.5")
        self.assertEqual(model.upstream_id, "mimo/mimo-v2.5")
        # Contract we depend on upstream: chat payload's model field must
        # equal the bound session model's id; desyncing them brings back
        # the upstream 409 session_model_mismatch.
        self.assertEqual(model.upstream_id, model.session_id)

    def test_resolve_gemini_flash_lite_routes_via_mimo(self) -> None:
        model = resolve_model("google/gemini-2.5-flash-lite")

        self.assertEqual(model.agent_id, GEMINI_AGENT_ID)
        self.assertIsNone(model.parent_agent_id)
        self.assertEqual(model.session_id, "mimo/mimo-v2.5")
        self.assertEqual(model.upstream_id, model.session_id)

    def test_resolve_gemini_flash_preview_routes_via_mimo(self) -> None:
        model = resolve_model("google/gemini-3.1-flash-lite-preview")

        self.assertEqual(model.agent_id, GEMINI_AGENT_ID)
        self.assertIsNone(model.parent_agent_id)
        self.assertEqual(model.upstream_id, "mimo/mimo-v2.5")
        self.assertEqual(model.upstream_id, model.session_id)

    def test_agent_validation_payload_defines_spawnable_agents(self) -> None:
        payload = agent_validation_payload()
        definitions = payload["agentDefinitions"]
        ids = {definition["id"] for definition in definitions}
        spawnable_ids = {
            agent_id
            for definition in definitions
            for agent_id in definition.get("spawnableAgents", [])
        }

        self.assertIn(CONTEXT_PRUNER_AGENT_ID, ids)
        self.assertLessEqual(spawnable_ids, ids)

    def test_agent_validation_payload_has_spawn_agent_tool_when_spawnable(self) -> None:
        payload = agent_validation_payload()

        for definition in payload["agentDefinitions"]:
            if definition.get("spawnableAgents"):
                self.assertIn("spawn_agents", definition["toolNames"])

    def test_build_upstream_payload_uses_explicit_client_id(self) -> None:
        payload = build_upstream_payload(
            {"model": "deepseek/deepseek-v4-pro", "messages": []},
            session=FreebuffSession(
                instance_id="instance-1",
                model="deepseek/deepseek-v4-pro",
            ),
            run_id="run-1",
            client_id="client-1",
            trace_session_id="trace-1",
        )

        self.assertTrue(payload["stream"])
        self.assertEqual(payload["model"], "deepseek/deepseek-v4-pro")
        self.assertEqual(payload["provider"], {"data_collection": "deny"})
        self.assertEqual(
            payload["codebuff_metadata"],
            {
                "freebuff_instance_id": "instance-1",
                "trace_session_id": "trace-1",
                "run_id": "run-1",
                "client_id": "client-1",
                "cost_mode": "free",
            },
        )

    def test_build_upstream_payload_can_override_upstream_model(self) -> None:
        payload = build_upstream_payload(
            {
                "model": "google/gemini-3.1-flash-lite-preview",
                "messages": [],
            },
            session=FreebuffSession(
                instance_id="instance-1",
                model="deepseek/deepseek-v4-flash",
            ),
            run_id="run-1",
            client_id="client-1",
            trace_session_id="trace-1",
            upstream_model_id="google/gemini-3.1-flash-lite-preview",
        )

        self.assertEqual(payload["model"], "google/gemini-3.1-flash-lite-preview")

    def test_build_upstream_payload_maps_developer_role_to_system(self) -> None:
        body = {
            "model": "deepseek/deepseek-v4-pro",
            "messages": [
                {"role": "developer", "content": "be helpful"},
                {"role": "user", "content": "hello"},
            ],
            "temperature": 0.2,
        }

        payload = build_upstream_payload(
            body,
            session=FreebuffSession(
                instance_id="instance-1",
                model="deepseek/deepseek-v4-pro",
            ),
            run_id="run-1",
            client_id="client-1",
            trace_session_id="trace-1",
        )

        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(body["messages"][0]["role"], "developer")

    def test_build_upstream_payload_filters_unknown_request_fields(self) -> None:
        payload = build_upstream_payload(
            {
                "model": "deepseek/deepseek-v4-pro",
                "messages": [],
                "temperature": 0.2,
                "provider": {"data_collection": "allow"},
                "codebuff_metadata": {"cost_mode": "paid"},
                "unexpected": "client-owned",
            },
            session=FreebuffSession(
                instance_id="instance-1",
                model="deepseek/deepseek-v4-pro",
            ),
            run_id="run-1",
            client_id="client-1",
            trace_session_id="trace-1",
        )

        self.assertEqual(payload["temperature"], 0.2)
        self.assertNotIn("unexpected", payload)
        self.assertEqual(payload["provider"], {"data_collection": "deny"})
        self.assertEqual(payload["codebuff_metadata"]["cost_mode"], "free")

    def test_accumulator_keeps_reasoning_content_separate(self) -> None:
        accumulator = CompletionAccumulator("deepseek/deepseek-v4-flash")

        accumulator.add(
            {
                "id": "chunk-1",
                "created": 1,
                "model": "deepseek/deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": None, "reasoning_content": "hello"},
                        "finish_reason": None,
                    }
                ],
            }
        )

        response = accumulator.final_response()

        message = response["choices"][0]["message"]
        self.assertEqual(message["content"], "")
        self.assertEqual(message["reasoning_content"], "hello")

    def test_accumulator_keeps_final_answer_as_content(self) -> None:
        accumulator = CompletionAccumulator("deepseek/deepseek-v4-flash")

        accumulator.add(
            {
                "id": "chunk-1",
                "created": 1,
                "model": "deepseek/deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": None, "reasoning_content": "thinking"},
                        "finish_reason": None,
                    },
                    {
                        "index": 0,
                        "delta": {"content": "answer", "reasoning_content": None},
                        "finish_reason": "stop",
                    },
                ],
            }
        )

        message = accumulator.final_response()["choices"][0]["message"]

        self.assertEqual(message["content"], "answer")
        self.assertEqual(message["reasoning_content"], "thinking")

    def test_stream_chunk_keeps_reasoning_content_separate(self) -> None:
        chunk = sanitize_stream_chunk(
            {
                "id": "chunk-1",
                "created": 1,
                "model": "deepseek/deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": None, "reasoning_content": "hello"},
                        "finish_reason": None,
                    }
                ],
            }
        )

        delta = chunk["choices"][0]["delta"]
        self.assertNotIn("content", delta)
        self.assertEqual(delta["reasoning_content"], "hello")


class DynamicModelsTests(unittest.TestCase):
    def tearDown(self) -> None:
        # Reset dynamic models so other tests are not affected
        set_dynamic_models(list(ALL_MODELS))

    def test_set_dynamic_models_merges_with_hardcoded_models(self) -> None:
        set_dynamic_models([FreebuffModel("custom/model", "custom-agent")])
        active = get_active_models()
        ids = [model.id for model in active]
        self.assertIn("custom/model", ids)
        self.assertIn("deepseek/deepseek-v4-flash", ids)
        self.assertIn("google/gemini-3.1-pro-preview", ids)

    def test_dynamic_models_take_precedence_over_hardcoded_duplicates(self) -> None:
        set_dynamic_models([FreebuffModel("deepseek/deepseek-v4-flash", "custom-agent", display_name="Custom Flash")])
        active = get_active_models()
        flash = [model for model in active if model.id == "deepseek/deepseek-v4-flash"][0]
        self.assertEqual(flash.agent_id, "custom-agent")
        self.assertEqual(flash.display_name, "Custom Flash")

    def test_resolve_model_uses_dynamic_models(self) -> None:
        set_dynamic_models([FreebuffModel("custom/model", "custom-agent")])
        model = resolve_model("custom/model")
        self.assertEqual(model.agent_id, "custom-agent")

    def test_models_response_uses_dynamic_models(self) -> None:
        set_dynamic_models([FreebuffModel("a/b", "agent-a"), FreebuffModel("c/d", "agent-c")])
        response = models_response()
        ids = [item["id"] for item in response["data"]]
        self.assertIn("a/b", ids)
        self.assertIn("c/d", ids)
        self.assertIn("deepseek/deepseek-v4-flash", ids)

    def test_agent_validation_payload_uses_dynamic_models(self) -> None:
        set_dynamic_models([FreebuffModel("a/b", "agent-a"), FreebuffModel("c/d", "agent-c")])
        payload = agent_validation_payload()
        ids = {definition["id"] for definition in payload["agentDefinitions"]}
        self.assertIn("agent-a", ids)
        self.assertIn("agent-c", ids)

    def test_resolve_model_falls_back_to_hardcoded_when_dynamic_empty(self) -> None:
        set_dynamic_models([])
        # empty dynamic list should be ignored and fallback used
        model = resolve_model("moonshotai/kimi-k2.7-code")
        self.assertEqual(model.agent_id, "base2-free-kimi")

    def test_empty_dynamic_models_falls_back_to_all_models(self) -> None:
        set_dynamic_models([])
        active = get_active_models()
        self.assertEqual(active, ALL_MODELS)


if __name__ == "__main__":
    unittest.main()
