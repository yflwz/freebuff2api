from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import time
from typing import Any, AsyncIterator
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .codebuff import (
    CodebuffAccountLease,
    CodebuffAccountPool,
    CodebuffClient,
    CodebuffError,
    FreebuffRun,
    SessionManager,
    utc_now_iso,
)
from .config import Settings, load_settings
from .logging_config import configure_logging, redact_headers, render_debug
from .openai_compat import (
    CompletionAccumulator,
    build_upstream_payload,
    normalize_chat_messages,
    sanitize_stream_chunk,
)
from .models import CONTEXT_PRUNER_AGENT_ID, FreebuffModel, models_response, resolve_model
from .sse import encode_sse
from . import anthropic_compat
from . import responses_compat


logger = logging.getLogger("freebuff2api.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    configure_logging(settings)
    accounts = CodebuffAccountPool(settings)
    app.state.accounts = accounts
    app.state.settings = settings
    app.state.codebuff = accounts.default_client
    app.state.sessions = accounts.default_sessions
    logger.info("configured freebuff accounts count=%s", accounts.account_count)
    try:
        yield
    finally:
        await accounts.aclose()


app = FastAPI(title="freebuff2api", version="0.1.0", lifespan=lifespan)


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _client(request: Request) -> CodebuffClient:
    return request.app.state.codebuff


def _sessions(request: Request) -> SessionManager:
    return request.app.state.sessions


def _accounts(request: Request) -> CodebuffAccountPool:
    return request.app.state.accounts


def _check_local_auth(request: Request) -> None:
    api_key = _settings(request).local_api_key
    if not api_key:
        return
    expected = f"Bearer {api_key}"
    if request.headers.get("authorization") != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _check_anthropic_auth(request: Request) -> None:
    api_key = _settings(request).local_api_key
    if not api_key:
        return
    header = (
        request.headers.get("x-api-key")
        or _bearer_anthropic_key(request)
        or ""
    )
    if header != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _bearer_anthropic_key(request: Request) -> str | None:
    """Some clients (CC Switch, etc.) send Authorization: Bearer <key>."""
    auth = request.headers.get("authorization") or ""
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


def _error_response(error: Exception) -> JSONResponse:
    if isinstance(error, CodebuffError):
        return JSONResponse(
            status_code=error.status_code,
            content={
                "error": {
                    "message": str(error),
                    "type": "upstream_error",
                    "code": "codebuff_error",
                }
            },
        )
    if isinstance(error, HTTPException):
        return JSONResponse(
            status_code=error.status_code,
            content={
                "error": {
                    "message": str(error.detail),
                    "type": "invalid_request_error",
                    "code": "http_error",
                }
            },
        )
    logger.exception("unhandled error in chat/responses endpoint")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": f"Internal server error: {error}",
                "type": "internal_server_error",
                "code": "internal_error",
            }
        },
    )


def _anthropic_error_response(error: Exception) -> JSONResponse:
    if isinstance(error, CodebuffError):
        return JSONResponse(
            status_code=error.status_code,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": str(error),
                },
            },
        )
    if isinstance(error, HTTPException):
        return JSONResponse(
            status_code=error.status_code,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": str(error.detail),
                },
            },
        )
    logger.exception("unhandled error in anthropic endpoint")
    return JSONResponse(
        status_code=500,
        content={
            "type": "error",
            "error": {
                "type": "api_error",
                "message": f"Internal server error: {error}",
            },
        },
    )




@app.api_route("/v1", methods=["GET", "HEAD"])
@app.api_route("/", methods=["GET", "HEAD"])
async def root() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/healthz")
async def healthz(request: Request) -> dict[str, Any]:
    _check_local_auth(request)
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    _check_local_auth(request)
    return models_response()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    _check_local_auth(request)
    body = await request.json()
    settings = _settings(request)
    try:
        model_config = resolve_model(body.get("model"))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    model = model_config.id
    logger.info(
        "chat completion request model=%s stream=%s messages=%s",
        model,
        body.get("stream") is True,
        len(body.get("messages") or []),
    )
    if settings.debug:
        logger.debug(
            "incoming request headers=%s",
            redact_headers(dict(request.headers)),
        )
        logger.debug(
            "chat completion request body=%s",
            render_debug(body, settings.log_body_chars),
        )

    messages = normalize_chat_messages(body.get("messages"))
    lease: CodebuffAccountLease | None = None
    try:
        lease = await _accounts(request).acquire_session(
            model_config.session_id,
            messages=messages,
        )
        client = lease.client
        await client.request_ad_chain(messages=messages)
        await client.validate_agents()
        run = await _start_freebuff_run_chain(client, model_config)
        trace_session_id = str(uuid.uuid4())
        payload = build_upstream_payload(
            {**body, "messages": messages},
            session=lease.session,
            run_id=run.payload_run_id,
            client_id=settings.client_id,
            trace_session_id=trace_session_id,
            upstream_model_id=model_config.upstream_id,
        )
        if settings.debug:
            logger.debug(
                "prepared upstream chat trace=%s run=%s payload=%s",
                trace_session_id,
                run,
                render_debug(payload, settings.log_body_chars),
            )
    except CodebuffError as error:
        if lease is not None:
            await lease.aclose()
        logger.warning(
            "failed to prepare chat completion: %s",
            error,
            exc_info=settings.debug,
        )
        return _error_response(error)
    except Exception as error:
        if lease is not None:
            await lease.aclose()
        logger.exception("failed to prepare chat completion")
        return _error_response(error)

    if body.get("stream") is True:
        return StreamingResponse(
            _stream_openai_chunks(request, payload, run, account_lease=lease),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        response = await _collect_completion(
            request,
            payload,
            run,
            model,
            client=lease.client,
        )
        return JSONResponse(response)
    except Exception as error:
        return _error_response(error)
    finally:
        await lease.aclose()


@app.post("/v1/messages")
async def anthropic_messages(request: Request) -> Any:
    _check_anthropic_auth(request)
    body = await request.json()
    settings = _settings(request)
    try:
        model_config = resolve_model(body.get("model"))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    model = model_config.id
    logger.info(
        "anthropic messages request model=%s stream=%s messages=%s",
        model,
        body.get("stream") is True,
        len(body.get("messages") or []),
    )

    messages = anthropic_compat.normalize_messages(
        body.get("messages"),
        body.get("system"),
    )
    lease: CodebuffAccountLease | None = None
    try:
        lease = await _accounts(request).acquire_session(
            model_config.session_id,
            messages=messages,
        )
        client = lease.client
        await client.request_ad_chain(messages=messages)
        await client.validate_agents()
        run = await _start_freebuff_run_chain(client, model_config)
        trace_session_id = str(uuid.uuid4())
        body_with_messages = {**body, "messages": messages}
        tools = body_with_messages.get("tools")
        if tools:
            body_with_messages["tools"] = anthropic_compat.translate_anthropic_tools(tools)
        tool_choice = body_with_messages.get("tool_choice")
        if tool_choice is not None:
            body_with_messages["tool_choice"] = anthropic_compat.translate_anthropic_tool_choice(tool_choice)
        payload = build_upstream_payload(
            body_with_messages,
            session=lease.session,
            run_id=run.payload_run_id,
            client_id=settings.client_id,
            trace_session_id=trace_session_id,
            upstream_model_id=model_config.upstream_id,
        )
        if settings.debug:
            logger.debug(
                "prepared anthropic upstream chat trace=%s run=%s payload=%s",
                trace_session_id,
                run,
                render_debug(payload, settings.log_body_chars),
            )
    except CodebuffError as error:
        if lease is not None:
            await lease.aclose()
        logger.warning(
            "failed to prepare anthropic chat completion: %s",
            error,
            exc_info=settings.debug,
        )
        return _anthropic_error_response(error)
    except Exception as error:
        if lease is not None:
            await lease.aclose()
        logger.exception("failed to prepare anthropic chat completion")
        return _anthropic_error_response(error)

    if body.get("stream") is True:
        return StreamingResponse(
            _stream_anthropic_chunks(request, payload, run, model, account_lease=lease),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        openai_resp = await _collect_completion(
            request, payload, run, model, client=lease.client,
        )
        choice = openai_resp["choices"][0]
        msg = choice["message"]
        anthropic_resp = anthropic_compat.build_non_streaming_response(
            {"content": msg.get("content", ""),
             "reasoning_content": msg.get("reasoning_content", ""),
             "tool_calls": msg.get("tool_calls") or [],
             "finish_reason": choice.get("finish_reason"),
             "usage": openai_resp.get("usage")},
            message_id="msg_" + uuid.uuid4().hex, model=model,
            input_tokens=0,
        )
        return JSONResponse(anthropic_resp)
    except Exception as error:
        return _anthropic_error_response(error)
    finally:
        await lease.aclose()



@app.post("/v1/responses")
async def responses_api(request: Request) -> Any:
    _check_local_auth(request)
    body = await request.json()
    settings = _settings(request)
    try:
        model_config = resolve_model(body.get("model"))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    model = model_config.id
    logger.info("responses api request model=%s input=%s", model, type(body.get("input")).__name__)

    messages = responses_compat._input_to_messages(body.get("input"), body.get("instructions"))
    lease: CodebuffAccountLease | None = None
    try:
        lease = await _accounts(request).acquire_session(model_config.session_id, messages=messages)
        client = lease.client
        await client.request_ad_chain(messages=messages)
        await client.validate_agents()
        run = await _start_freebuff_run_chain(client, model_config)
        trace_session_id = str(uuid.uuid4())
        payload = responses_compat.build_upstream_payload(
            {**body, "messages": messages},
            session=lease.session, run_id=run.payload_run_id,
            client_id=settings.client_id, trace_session_id=trace_session_id,
            upstream_model_id=model_config.upstream_id,
        )
    except CodebuffError as error:
        if lease: await lease.aclose()
        logger.warning("failed to prepare responses api: %s", error, exc_info=settings.debug)
        return _error_response(error)
    except Exception as error:
        if lease: await lease.aclose()
        logger.exception("failed to prepare responses api")
        return _error_response(error)

    if body.get("stream") is True:
        return StreamingResponse(
            _stream_responses_chunks(request, payload, run, model, account_lease=lease),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    try:
        accumulator = CompletionAccumulator(model)
        async for data in client.chat_events(payload):
            if data is None:
                continue
            if data == "[DONE]":
                break
            accumulator.add(data)
        await _finalize_run_with_client(client, run, None)
        await lease.aclose()
        response = accumulator.final_response()
        msg = response["choices"][0]["message"]
        choice = response["choices"][0]
        resp = responses_compat.build_non_streaming_response(
            {
                "content": msg.get("content", "") or "",
                "reasoning_content": msg.get("reasoning_content", "") or "",
                "tool_calls": msg.get("tool_calls") or [],
                "usage": response.get("usage"),
                "finish_reason": choice.get("finish_reason"),
            },
            response_id="resp_" + uuid.uuid4().hex, model=model,
        )
        return JSONResponse(resp)
    except Exception as error:
        return _error_response(error)


async def _stream_responses_chunks(
    request: Request, payload: dict[str, Any], run: FreebuffRun, model: str, *,
    account_lease: CodebuffAccountLease | None = None, client: CodebuffClient | None = None,
) -> AsyncIterator[bytes]:
    message_id: str | None = None
    client = client or (account_lease.client if account_lease else _client(request))
    settings = _settings(request)
    resp_id = "resp_" + uuid.uuid4().hex
    state = responses_compat.ResponsesStreamState()
    try:
        async for data in client.chat_events(payload):
            if data is None:
                continue
            if data == "[DONE]":
                break
            if isinstance(data, dict):
                message_id = data.get("id") or message_id
                chunk = sanitize_stream_chunk(data)
            else:
                chunk = None
            if chunk is None: continue
            for event_type, event_data in responses_compat.responses_stream_events(
                chunk, response_id=resp_id, model=model, state=state,
            ):
                yield responses_compat.encode_responses_event(event_type, event_data).encode("utf-8")

    except CodebuffError as error:
        logger.warning("responses stream failed run_id=%s: %s", run.run_id, error, exc_info=settings.debug)
        yield responses_compat.encode_responses_event("error", {
            "type": "error", "error": {"type": "api_error", "message": str(error)},
        }).encode("utf-8")
        if not state.final_emitted:
            for event_type, event_data in responses_compat.responses_stream_events(
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                response_id=resp_id, model=model, state=state,
            ):
                yield responses_compat.encode_responses_event(event_type, event_data).encode("utf-8")
    finally:
        _schedule_finalize_run(client, run, message_id)
        if account_lease is not None:
            await account_lease.aclose()

async def _stream_openai_chunks(
    request: Request,
    payload: dict[str, Any],
    run: FreebuffRun,
    *,
    account_lease: CodebuffAccountLease | None = None,
    client: CodebuffClient | None = None,
) -> AsyncIterator[bytes]:
    message_id: str | None = None
    client = client or (account_lease.client if account_lease else _client(request))
    settings = _settings(request)
    try:
        async for data in client.chat_events(payload):
            if data is None:
                continue
            if data == "[DONE]":
                if settings.debug:
                    logger.debug(
                        "chat stream done run_id=%s message_id=%s",
                        run.run_id,
                        message_id,
                    )
                yield encode_sse("[DONE]")
                break

            if isinstance(data, dict):
                message_id = data.get("id") or message_id
                chunk = sanitize_stream_chunk(data)
            else:
                chunk = None
            if chunk is not None:
                if settings.debug:
                    logger.debug(
                        "chat stream chunk=%s",
                        render_debug(chunk, settings.log_body_chars),
                    )
                yield encode_sse(chunk)
            elif settings.debug:
                logger.debug(
                    "chat stream ignored data=%s",
                    render_debug(data, settings.log_body_chars),
                )
    except CodebuffError as error:
        logger.warning(
            "chat stream failed run_id=%s: %s",
            run.run_id,
            error,
            exc_info=settings.debug,
        )
        yield encode_sse(
            {
                "error": {
                    "message": str(error),
                    "type": "upstream_error",
                    "code": "codebuff_error",
                }
            }
        )
        yield encode_sse("[DONE]")
    finally:
        _schedule_finalize_run(client, run, message_id)
        if account_lease is not None:
            await account_lease.aclose()


async def _stream_anthropic_chunks(
    request: Request,
    payload: dict[str, Any],
    run: FreebuffRun,
    model: str,
    *,
    account_lease: CodebuffAccountLease | None = None,
    client: CodebuffClient | None = None,
) -> AsyncIterator[bytes]:
    message_id: str | None = None
    client = client or (account_lease.client if account_lease else _client(request))
    settings = _settings(request)
    started = int(time.time())
    anthropic_msg_id = f"msg_{uuid.uuid4().hex}"
    state = anthropic_compat.AnthropicStreamState()
    try:
        async for data in client.chat_events(payload):
            if data is None:
                continue
            if data == "[DONE]":
                if settings.debug:
                    logger.debug(
                        "anthropic chat stream done run_id=%s message_id=%s",
                        run.run_id,
                        message_id,
                    )
                break

            if isinstance(data, dict):
                message_id = data.get("id") or message_id
                chunk = sanitize_stream_chunk(data)
            else:
                chunk = None
            if chunk is None:
                continue
            events = anthropic_compat.anthropic_stream_events(
                chunk,
                message_id=anthropic_msg_id,
                model=model,
                started=started,
                input_tokens=0,
                state=state,
            )
            for event_type, event_data in events:
                yield anthropic_compat.encode_anthropic_event(event_type, event_data).encode("utf-8")
    except CodebuffError as error:
        logger.warning(
            "anthropic chat stream failed run_id=%s: %s",
            run.run_id,
            error,
            exc_info=settings.debug,
        )
        yield anthropic_compat.encode_anthropic_event("error", {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": str(error),
            },
        }).encode("utf-8")
    finally:
        _schedule_finalize_run(client, run, message_id)
        if account_lease is not None:
            await account_lease.aclose()


async def _collect_completion(
    request: Request,
    payload: dict[str, Any],
    run: FreebuffRun,
    model: str,
    *,
    client: CodebuffClient | None = None,
) -> dict[str, Any]:
    message_id: str | None = None
    accumulator = CompletionAccumulator(model)
    client = client or _client(request)
    try:
        async for data in client.chat_events(payload):
            if data is None:
                continue
            if data == "[DONE]":
                break
            message_id = data.get("id") or message_id
            accumulator.add(data)
        response = accumulator.final_response()
        logger.info(
            "chat completion response run_id=%s message_id=%s content_chars=%s finish_reason=%s",
            run.run_id,
            message_id,
            len(response["choices"][0]["message"].get("content") or ""),
            response["choices"][0].get("finish_reason"),
        )
        if _settings(request).debug:
            logger.debug(
                "chat completion response body=%s",
                render_debug(response, _settings(request).log_body_chars),
            )
        return response
    finally:
        await _finalize_run(request, run, message_id, client=client)


async def _start_freebuff_run_chain(
    client: CodebuffClient,
    model: FreebuffModel | str,
) -> FreebuffRun:
    if isinstance(model, str):
        model = FreebuffModel(model, model)
    if model.parent_agent_id:
        return await _start_child_chat_run_chain(client, model)

    agent_id = model.agent_id
    started_at = utc_now_iso()
    run_id = await client.start_run(agent_id)
    child_started_at = utc_now_iso()
    child_run_id = await client.start_run(
        CONTEXT_PRUNER_AGENT_ID,
        ancestor_run_ids=[run_id],
    )
    await client.record_run_step(
        child_run_id,
        step_number=1,
        child_run_ids=[],
        message_id=None,
        start_time=child_started_at,
    )
    await client.finish_run(child_run_id, total_steps=2)
    await client.record_run_step(
        run_id,
        step_number=1,
        child_run_ids=[child_run_id],
        message_id=None,
        start_time=started_at,
    )
    return FreebuffRun(
        run_id=run_id,
        agent_id=agent_id,
        started_at=started_at,
        child_run_id=child_run_id,
    )


async def _start_child_chat_run_chain(
    client: CodebuffClient,
    model: FreebuffModel,
) -> FreebuffRun:
    assert model.parent_agent_id is not None

    started_at = utc_now_iso()
    parent_run_id = await client.start_run(model.parent_agent_id)
    chat_started_at = utc_now_iso()
    chat_run_id = await client.start_run(
        model.agent_id,
        ancestor_run_ids=[parent_run_id],
    )
    return FreebuffRun(
        run_id=parent_run_id,
        agent_id=model.parent_agent_id,
        started_at=started_at,
        child_run_id=chat_run_id,
        chat_run_id=chat_run_id,
        chat_started_at=chat_started_at,
    )


async def _finalize_run(
    request: Request,
    run: FreebuffRun,
    message_id: str | None,
    *,
    client: CodebuffClient | None = None,
) -> None:
    await _finalize_run_with_client(client or _client(request), run, message_id)


def _schedule_finalize_run(
    client: CodebuffClient,
    run: FreebuffRun,
    message_id: str | None,
) -> None:
    task = asyncio.create_task(_finalize_run_with_client(client, run, message_id))

    def _log_background_error(done: asyncio.Task[None]) -> None:
        try:
            done.result()
        except asyncio.CancelledError:
            logger.debug("background finalize task cancelled run_id=%s", run.run_id)
        except Exception:
            logger.exception("background finalize task failed run_id=%s", run.run_id)

    task.add_done_callback(_log_background_error)


async def _finalize_run_with_client(
    client: CodebuffClient,
    run: FreebuffRun,
    message_id: str | None,
) -> None:
    try:
        logger.debug(
            "finalize run start run_id=%s message_id=%s started_at=%s",
            run.run_id,
            message_id,
            run.started_at,
        )
        if run.chat_run_id and run.chat_run_id != run.run_id:
            await client.record_run_step(
                run.chat_run_id,
                step_number=1,
                child_run_ids=[],
                message_id=message_id,
                start_time=run.chat_started_at or run.started_at,
            )
            await client.finish_run(run.chat_run_id, total_steps=2)
            await client.record_run_step(
                run.run_id,
                step_number=1,
                child_run_ids=[run.chat_run_id],
                message_id=None,
                start_time=run.started_at,
            )
            await client.finish_run(run.run_id, total_steps=2)
            logger.debug("finalize parent/child run done run_id=%s", run.run_id)
            return

        await client.record_run_step(
            run.run_id,
            step_number=2,
            child_run_ids=[],
            message_id=message_id,
            start_time=run.started_at,
        )
        await client.finish_run(run.run_id, total_steps=3)
        logger.debug("finalize run done run_id=%s", run.run_id)
    except CodebuffError as error:
        logger.warning(
            "finalize run failed run_id=%s: %s",
            run.run_id,
            error,
            exc_info=client.settings.debug,
        )
    except Exception:
        logger.exception("finalize run failed run_id=%s", run.run_id)
