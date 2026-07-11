from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FreebuffModel:
    id: str
    agent_id: str
    owned_by: str = "freebuff"
    upstream_model_id: str | None = None
    session_model_id: str | None = None
    parent_agent_id: str | None = None
    display_name: str | None = None

    @property
    def upstream_id(self) -> str:
        return self.upstream_model_id or self.id

    @property
    def session_id(self) -> str:
        return self.session_model_id or self.upstream_id


FREEBUFF_MODELS: tuple[FreebuffModel, ...] = (
    FreebuffModel("deepseek/deepseek-v4-flash", "base2-free-deepseek-flash", display_name="DeepSeek V4 Flash"),
    FreebuffModel("deepseek/deepseek-v4-pro", "base2-free-deepseek", display_name="DeepSeek V4 Pro"),
    FreebuffModel("moonshotai/kimi-k2.7-code", "base2-free-kimi", display_name="Kimi K2.7 Code"),
    FreebuffModel("minimax/minimax-m3", "base2-free-minimax-m3", display_name="MiniMax M3"),
    FreebuffModel("mimo/mimo-v2.5", "base2-free-mimo", display_name="MiMo 2.5"),
    FreebuffModel("mimo/mimo-v2.5-pro", "base2-free-mimo-pro", display_name="MiMo 2.5 Pro"),
    FreebuffModel("kwaipilot/kat-coder-pro-v2", "base2-free", display_name="KAT Coder Pro V2"),
    FreebuffModel("tencent/hy3:free", "base2-free", display_name="GLM 5.2"),
)

DEFAULT_MODEL = FREEBUFF_MODELS[0]
CONTEXT_PRUNER_AGENT_ID = "context-pruner"
GEMINI_THINKER_AGENT_ID = "thinker-with-files-gemini"
# All "free" Gemini variants route their quota pool + parent lineage onto
# mimo/mimo-v2.5 (unlimited upstream) instead of kimi's 5/day pool. Chat
# dispatch (agent_id) is unchanged so users still get the Gemini-flavored
# responses; only the rate-limit counter and parent agent move to mimo.
GEMINI_SESSION_MODEL_ID = "mimo/mimo-v2.5"
GEMINI_PARENT_AGENT_ID = "base2-free-mimo"
# Backwards-compatible aliases for downstream test imports / external
# callers that still reference the older constants by name.
GEMINI_THINKER_PARENT_AGENT_ID = GEMINI_PARENT_AGENT_ID
GEMINI_THINKER_PARENT_MODEL_ID = GEMINI_SESSION_MODEL_ID
GEMINI_FLASH_LITE_SESSION_MODEL_ID = GEMINI_SESSION_MODEL_ID

GEMINI_FREE_MODELS: tuple[FreebuffModel, ...] = (
    FreebuffModel(
        "google/gemini-2.5-flash-lite",
        "file-picker",
        owned_by="google",
        session_model_id=GEMINI_SESSION_MODEL_ID,
        parent_agent_id=GEMINI_PARENT_AGENT_ID,
        display_name="Gemini 2.5 Flash Lite",
    ),
    FreebuffModel(
        "google/gemini-3.1-flash-lite-preview",
        "file-picker-max",
        owned_by="google",
        session_model_id=GEMINI_SESSION_MODEL_ID,
        parent_agent_id=GEMINI_PARENT_AGENT_ID,
        display_name="Gemini 3.1 Flash Lite Preview",
    ),
    FreebuffModel(
        "google/gemini-3.1-pro-preview",
        GEMINI_THINKER_AGENT_ID,
        owned_by="google",
        session_model_id=GEMINI_THINKER_PARENT_MODEL_ID,
        parent_agent_id=GEMINI_THINKER_PARENT_AGENT_ID,
        display_name="Gemini 3.1 Pro Preview",
    ),
)

ALL_MODELS = FREEBUFF_MODELS + GEMINI_FREE_MODELS

# Mapping from upstream provider prefix to the agent id used by Codebuff.
# Used when a dynamically discovered model is not in the hardcoded ALL_MODELS.
_PROVIDER_AGENT_MAP: dict[str, str] = {
    "deepseek/": "base2-free-deepseek",
    "moonshotai/": "base2-free-kimi",
    "minimax/": "base2-free",
    "mimo/": "base2-free-mimo",
    "tencent/": "base2-free",
    "kwaipilot/": "base2-free",
}

# Explicit display-name aliases for upstream model ids that don't map cleanly
# to a human-readable name via the generic derivation logic.
_DISPLAY_NAME_ALIASES: dict[str, str] = {
    "tencent/hy3": "GLM 5.2",
    "tencent/hy3:free": "GLM 5.2 Free",
    "tencent/hy3.free": "GLM 5.2 Free",
}


# Runtime-discovered models from upstream Codebuff. Populated at app startup.
_DYNAMIC_MODELS: tuple[FreebuffModel, ...] | None = None


def map_model_to_agent_id(model_id: str) -> str:
    """Map an upstream model id to the Codebuff agent id.

    First tries exact matches against the hardcoded model list, then falls back
    to provider-prefix heuristics, and finally returns the model id itself.
    """
    # Exact match from hardcoded models
    for model in ALL_MODELS:
        if model.id == model_id:
            return model.agent_id

    # Provider-prefix heuristic for dynamically discovered variants
    for prefix, agent_id in _PROVIDER_AGENT_MAP.items():
        if model_id.startswith(prefix):
            return agent_id

    # Fallback: the upstream model id is often a valid agent id in Codebuff
    return model_id


def set_dynamic_models(models: list[FreebuffModel]) -> None:
    """Replace the active model list with dynamically discovered models."""
    global _DYNAMIC_MODELS
    _DYNAMIC_MODELS = tuple(models)


def get_active_models() -> tuple[FreebuffModel, ...]:
    """Return the merged active model list.

    Hardcoded models are always included so that special agents (e.g. Gemini
    free models) remain available even when upstream discovery omits them.
    Dynamically discovered models take precedence when ids overlap.
    """
    if not _DYNAMIC_MODELS:
        return ALL_MODELS
    dynamic_by_id = {model.id: model for model in _DYNAMIC_MODELS}
    merged: list[FreebuffModel] = list(_DYNAMIC_MODELS)
    for model in ALL_MODELS:
        if model.id not in dynamic_by_id:
            merged.append(model)
    return tuple(merged)


def resolve_model(requested: str | None) -> FreebuffModel:
    active = get_active_models()
    if not requested:
        return active[0]
    for model in active:
        if model.id == requested:
            return model
    # fallback: match by suffix for clients that omit provider prefix
    suffix = f"/{requested}"
    for model in active:
        if model.id.endswith(suffix):
            return model
    raise ValueError(f"Unsupported Freebuff model: {requested}")


def derive_display_name(model_id: str) -> str:
    """Derive a human-readable display name from an upstream model id."""
    if model_id in _DISPLAY_NAME_ALIASES:
        return _DISPLAY_NAME_ALIASES[model_id]
    display = model_id.split("/")[-1]
    display = display.replace("-", " ").replace(":", " ")
    return display.title()


def models_response() -> dict[str, object]:
    active = get_active_models()
    return {
        "object": "list",
        "data": [
            {
                "id": model.id,
                "object": "model",
                "created": 0,
                "owned_by": model.owned_by,
                "display_name": model.display_name or derive_display_name(model.id),
            }
            for model in active
        ],
    }


def agent_validation_payload() -> dict[str, object]:
    active = get_active_models()
    models_by_agent: dict[str, FreebuffModel] = {}
    spawnable_by_agent: dict[str, set[str]] = {}
    for model in active:
        models_by_agent.setdefault(model.agent_id, model)
        spawnable_by_agent.setdefault(model.agent_id, set()).add(CONTEXT_PRUNER_AGENT_ID)
        if model.parent_agent_id:
            spawnable_by_agent.setdefault(model.parent_agent_id, set()).add(model.agent_id)

    definitions = [
        _agent_definition(
            agent_id=model.agent_id,
            model_id=model.upstream_id,
            display_name=f"Freebuff {model.upstream_id}",
            spawnable_agents=sorted(spawnable_by_agent.get(model.agent_id, set())),
        )
        for model in models_by_agent.values()
    ]
    definitions.append(
        _agent_definition(
            agent_id=CONTEXT_PRUNER_AGENT_ID,
            model_id=active[0].id,
            display_name="Context Pruner",
            spawnable_agents=[],
        )
    )

    return {"agentDefinitions": definitions}


def _agent_definition(
    *,
    agent_id: str,
    model_id: str,
    display_name: str,
    spawnable_agents: list[str],
) -> dict[str, object]:
    return {
        "id": agent_id,
        "publisher": "codebuff",
        "model": model_id,
        "displayName": display_name,
        "spawnerPrompt": "Freebuff OpenAI-compatible orchestrator",
        "inputSchema": {
            "prompt": {
                "type": "string",
                "description": "A coding task to complete",
            },
            "params": {"type": "object", "properties": {}, "required": []},
        },
        "outputMode": "last_message",
        "includeMessageHistory": True,
        "toolNames": ["spawn_agents"] if spawnable_agents else [],
        "spawnableAgents": spawnable_agents,
        "systemPrompt": "Act as a helpful coding assistant.",
    }
