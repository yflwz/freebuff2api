import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import httpx

from freebuff2api.codebuff import CodebuffError
from freebuff2api.codebuff import CodebuffClient
from freebuff2api.codebuff import CHAT_COMPLETIONS_USER_AGENT
from freebuff2api.codebuff import CODEBUFF_ACCEPT_ENCODING
from freebuff2api.codebuff import FreebuffSession
from freebuff2api.codebuff import RateLimit
from freebuff2api.codebuff import SessionManager
from freebuff2api.config import HAR_BROWSER_USER_AGENT
from freebuff2api.config import Settings


class QueuedSessionClient(CodebuffClient):
    def __init__(self) -> None:
        super().__init__(
            Settings(
                codebuff_token="token",
                local_api_key=None,
                request_timeout=1,
            )
        )
        self.calls = []
        self.responses = [
            {
                "status": "queued",
                "instanceId": "queued-instance",
                "model": "moonshotai/kimi-k2.6",
                "position": 0,
                "queueDepth": 0,
                "estimatedWaitMs": 0,
            },
            {
                "status": "active",
                "instanceId": "queued-instance",
                "model": "moonshotai/kimi-k2.6",
                "expiresAt": "2026-05-23T16:04:31.177Z",
                "remainingMs": 3_000_000,
            },
        ]

    async def _json(self, method, path, *, body=None, headers=None):
        self.calls.append((method, path))
        return self.responses.pop(0)


class CapturingAdsClient(CodebuffClient):
    def __init__(self) -> None:
        super().__init__(
            Settings(
                codebuff_token="token",
                local_api_key=None,
                request_timeout=1,
            )
        )
        self.body = None

    async def _json(self, method, path, *, body=None, headers=None):
        self.body = body
        return {"ads": []}


class FailingAdsClient(CodebuffClient):
    def __init__(self) -> None:
        super().__init__(
            Settings(
                codebuff_token="token",
                local_api_key=None,
                ad_providers=("gravity", "carbon"),
                request_timeout=1,
            )
        )
        self.providers = []

    async def request_ads(self, provider, messages=None, surface=None):
        self.providers.append(provider)
        raise CodebuffError(f"{provider} unavailable", 502)


class CodebuffClientTests(unittest.IsolatedAsyncioTestCase):
    def test_client_uses_explicit_proxy_only_when_enabled(self) -> None:
        captured = {}

        class FakeAsyncClient:
            def __init__(self, **kwargs) -> None:
                captured.update(kwargs)

        settings = Settings(
            codebuff_token="token",
            local_api_key=None,
            proxy_enabled=True,
            proxy_url="socks5://127.0.0.1:1080",
        )

        with patch("freebuff2api.codebuff.httpx.AsyncClient", FakeAsyncClient):
            CodebuffClient(settings)

        self.assertEqual(captured["proxy"], "socks5://127.0.0.1:1080")
        self.assertFalse(captured["trust_env"])

    async def test_create_session_polls_queued_session_until_active(self) -> None:
        client = QueuedSessionClient()
        try:
            session = await client.create_session("moonshotai/kimi-k2.6")
        finally:
            await client.aclose()

        self.assertEqual(session.instance_id, "queued-instance")
        self.assertEqual(session.model, "moonshotai/kimi-k2.6")
        self.assertEqual(
            client.calls,
            [
                ("POST", "/api/v1/freebuff/session"),
                ("GET", "/api/v1/freebuff/session"),
            ],
        )

    async def test_request_ads_converts_openai_content_parts_to_string(self) -> None:
        client = CapturingAdsClient()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
                    {"type": "text", "text": "world"},
                ],
            },
            {"role": "assistant", "content": None},
        ]

        try:
            await client.request_ads("gravity", messages=messages)
        finally:
            await client.aclose()

        self.assertEqual(
            client.body["messages"],
            [
                {"role": "user", "content": "hello\nworld"},
                {"role": "assistant", "content": ""},
            ],
        )
        self.assertEqual(client.body["userAgent"], HAR_BROWSER_USER_AGENT)
        self.assertIsInstance(messages[0]["content"], list)

    async def test_request_ads_maps_developer_role_to_system(self) -> None:
        client = CapturingAdsClient()

        try:
            await client.request_ads(
                "gravity",
                messages=[{"role": "developer", "content": "be helpful"}],
            )
        finally:
            await client.aclose()

        self.assertEqual(
            client.body["messages"],
            [{"role": "system", "content": "be helpful"}],
        )

    async def test_request_ad_chain_does_not_block_when_all_providers_fail(self) -> None:
        client = FailingAdsClient()

        try:
            with self.assertLogs("freebuff2api.codebuff", level="WARNING") as logs:
                await client.request_ad_chain(
                    messages=[{"role": "user", "content": "hi"}]
                )
        finally:
            await client.aclose()

        self.assertEqual(client.providers, ["gravity", "carbon"])
        self.assertIn("ads provider=gravity failed", logs.output[0])
        self.assertIn("ads provider=carbon failed", logs.output[1])

    async def test_json_wraps_network_error_as_codebuff_error(self) -> None:
        def raise_connect_error(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("proxy connect failed", request=request)

        client = CodebuffClient(
            Settings(
                codebuff_token="token",
                local_api_key=None,
                request_timeout=1,
            )
        )
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(raise_connect_error),
            timeout=1,
        )

        try:
            with self.assertRaises(CodebuffError) as ctx:
                await client.get_session()
        finally:
            await client.aclose()

        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("network error", str(ctx.exception))
        self.assertIn("ConnectError", str(ctx.exception))

    async def test_json_explains_session_model_mismatch_as_region_limit(self) -> None:
        def session_model_mismatch(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                409,
                json={
                    "error": "session_model_mismatch",
                    "message": (
                        "Limited free access is only available with DeepSeek V4 Flash."
                    ),
                },
            )

        client = CodebuffClient(
            Settings(
                codebuff_token="token",
                local_api_key=None,
                request_timeout=1,
            )
        )
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(session_model_mismatch),
            timeout=1,
        )

        try:
            with self.assertRaises(CodebuffError) as ctx:
                await client.create_session("deepseek/deepseek-v4-pro")
        finally:
            await client.aclose()

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("当前 IP/区域", str(ctx.exception))
        self.assertIn("US", str(ctx.exception))

    async def test_chat_stream_explains_session_model_mismatch_as_region_limit(self) -> None:
        def session_model_mismatch(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                409,
                json={
                    "error": "session_model_mismatch",
                    "message": (
                        "Limited free access is only available with DeepSeek V4 Flash."
                    ),
                },
            )

        client = CodebuffClient(
            Settings(
                codebuff_token="token",
                local_api_key=None,
                request_timeout=1,
            )
        )
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(session_model_mismatch),
            timeout=1,
        )

        try:
            with self.assertRaises(CodebuffError) as ctx:
                async for _ in client.chat_events({"messages": []}):
                    pass
        finally:
            await client.aclose()

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("当前 IP/区域", str(ctx.exception))
        self.assertIn("US", str(ctx.exception))

    async def test_chat_events_uses_har_fingerprint_headers(self) -> None:
        captured_headers = {}

        def capture_headers(request: httpx.Request) -> httpx.Response:
            captured_headers.update(dict(request.headers))
            return httpx.Response(200, content=b"data: [DONE]\n\n")

        client = CodebuffClient(
            Settings(
                codebuff_token="token",
                local_api_key=None,
                request_timeout=1,
            )
        )
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(capture_headers),
            timeout=1,
        )

        try:
            async for _ in client.chat_events({"messages": []}):
                pass
        finally:
            await client.aclose()

        self.assertEqual(captured_headers["authorization"], "Bearer token")
        self.assertEqual(captured_headers["content-type"], "application/json")
        self.assertEqual(captured_headers["user-agent"], CHAT_COMPLETIONS_USER_AGENT)
        self.assertEqual(captured_headers["connection"], "keep-alive")
        self.assertEqual(captured_headers["accept"], "*/*")
        self.assertEqual(captured_headers["host"], "www.codebuff.com")
        self.assertEqual(captured_headers["accept-encoding"], CODEBUFF_ACCEPT_ENCODING)

    async def test_fetch_available_models_parses_session_rate_limits(self) -> None:
        def session_response(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "status": "active",
                    "instanceId": "test-instance",
                    "model": "moonshotai/kimi-k2.7-code",
                    "rateLimitsByModel": {
                        "moonshotai/kimi-k2.7-code": {"limit": 5, "period": "pacific_day"},
                        "deepseek/deepseek-v4-pro": {"limit": 5, "period": "pacific_day"},
                        "tencent/hy3:free": {"limit": 999, "period": "pacific_day"},
                    },
                },
            )

        client = CodebuffClient(
            Settings(
                codebuff_token="token",
                local_api_key=None,
                request_timeout=1,
                models_api_path="/api/v1/freebuff/session",
            )
        )
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(session_response),
            timeout=1,
        )

        try:
            models = await client.fetch_available_models()
        finally:
            await client.aclose()

        ids = {model.id for model in models}
        self.assertEqual(ids, {"moonshotai/kimi-k2.7-code", "deepseek/deepseek-v4-pro", "tencent/hy3:free"})
        expected_agents = {
            "moonshotai/kimi-k2.7-code": "base2-free-kimi",
            "deepseek/deepseek-v4-pro": "base2-free-deepseek",
            "tencent/hy3:free": "base2-free",
        }
        for model in models:
            self.assertEqual(model.agent_id, expected_agents[model.id])


class RateLimitTests(unittest.IsolatedAsyncioTestCase):
    def test_rate_limit_is_active_returns_true_until_reset(self) -> None:
        rate_limit = RateLimit(
            model="moonshotai/kimi-k2.7-code",
            reset_at=datetime.now(tz=timezone.utc) + timedelta(hours=1),
            retry_after_ms=3_600_000,
            raw_body="{}",
        )
        self.assertTrue(rate_limit.is_active)

    def test_rate_limit_is_active_returns_false_after_reset(self) -> None:
        rate_limit = RateLimit(
            model="moonshotai/kimi-k2.7-code",
            reset_at=datetime.now(tz=timezone.utc) - timedelta(seconds=1),
            retry_after_ms=0,
            raw_body="{}",
        )
        self.assertFalse(rate_limit.is_active)

    def test_rate_limit_format_error_includes_raw_body_and_status(self) -> None:
        body = '{"model":"moonshotai/kimi-k2.7-code","resetAt":"2099-01-01T00:00:00Z"}'
        rate_limit = RateLimit(
            model="moonshotai/kimi-k2.7-code",
            reset_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
            retry_after_ms=0,
            raw_body=body,
        )
        self.assertEqual(
            rate_limit.format_error(),
            f"Codebuff request failed: 429 {body}",
        )

    async def test_json_caches_rate_limit_on_429(self) -> None:
        def return_429(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={
                    "model": "moonshotai/kimi-k2.7-code",
                    "limit": 5,
                    "resetTimeZone": "America/Los_Angeles",
                    "resetAt": "2099-01-01T00:00:00.000Z",
                    "retryAfterMs": 42_500_000,
                    "status": "rate_limited",
                },
            )

        client = CodebuffClient(
            Settings(codebuff_token="token", local_api_key=None, request_timeout=1)
        )
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(return_429), timeout=1
        )

        try:
            with self.assertRaises(CodebuffError) as ctx:
                await client._json("GET", "/api/v1/freebuff/session")
            self.assertEqual(ctx.exception.status_code, 429)
            self.assertIn(
                "moonshotai/kimi-k2.7-code", client._rate_limit_cache
            )
            cached = client._rate_limit_cache["moonshotai/kimi-k2.7-code"]
            self.assertTrue(cached.is_active)
            self.assertEqual(cached.retry_after_ms, 42_500_000)
            self.assertIn("\"model\":\"moonshotai/kimi-k2.7-code\"", cached.raw_body)
        finally:
            await client.aclose()

    async def test_json_does_not_cache_non_429_with_rate_like_body(self) -> None:
        def return_400(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "error": "Invalid request body",
                    "details": {"provider": {"_errors": ["expected gravity|carbon"]}},
                    "model": "moonshotai/kimi-k2.7-code",
                },
            )

        client = CodebuffClient(
            Settings(codebuff_token="token", local_api_key=None, request_timeout=1)
        )
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(return_400), timeout=1
        )

        try:
            with self.assertRaises(CodebuffError):
                await client._json("GET", "/api/v1/freebuff/session")
            self.assertEqual(client._rate_limit_cache, {})
        finally:
            await client.aclose()

    async def test_json_falls_back_to_retry_after_when_reset_at_missing(self) -> None:
        def return_429(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={
                    "model": "moonshotai/kimi-k2.7-code",
                    "retryAfterMs": 60_000,
                },
            )

        client = CodebuffClient(
            Settings(codebuff_token="token", local_api_key=None, request_timeout=1)
        )
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(return_429), timeout=1
        )

        try:
            with self.assertRaises(CodebuffError):
                await client._json("GET", "/api/v1/freebuff/session")
            cached = client._rate_limit_cache.get("moonshotai/kimi-k2.7-code")
            self.assertIsNotNone(cached)
            # Within ~60s window from "now".
            self.assertTrue(cached.is_active)
            self.assertEqual(cached.retry_after_ms, 60_000)
        finally:
            await client.aclose()

    async def test_session_manager_fails_fast_on_cached_429(self) -> None:
        # Pre-populate cache with an active rate limit and confirm we never
        # reach the get_session/delete_session/create_session pipeline.
        class _TrackingClient(CodebuffClient):
            def __init__(self) -> None:
                super().__init__(
                    Settings(
                        codebuff_token="token",
                        local_api_key=None,
                        request_timeout=1,
                    )
                )

            async def get_session(self, instance_id=None):
                raise AssertionError("get_session should not be called")

            async def delete_session(self) -> None:
                raise AssertionError("delete_session should not be called")

            async def create_session(self, model):
                raise AssertionError("create_session should not be called")

        client = _TrackingClient()
        client._rate_limit_cache["moonshotai/kimi-k2.7-code"] = RateLimit(
            model="moonshotai/kimi-k2.7-code",
            reset_at=datetime.now(tz=timezone.utc) + timedelta(hours=12),
            retry_after_ms=43_200_000,
            raw_body=(
                '{"model":"moonshotai/kimi-k2.7-code",'
                '"resetAt":"2099-01-01T00:00:00.000Z",'
                '"retryAfterMs":43200000}'
            ),
        )
        manager = SessionManager(client, client.settings)
        try:
            with self.assertRaises(CodebuffError) as ctx:
                await manager.ensure_session("moonshotai/kimi-k2.7-code")
            self.assertEqual(ctx.exception.status_code, 429)
            self.assertIn(
                "moonshotai/kimi-k2.7-code", str(ctx.exception)
            )
        finally:
            await client.aclose()

    async def test_session_manager_proceeds_when_cached_429_expired(self) -> None:
        # When the cached reset_at is in the past, ensure_session should
        # proceed past the cache check into the normal flow.
        class _CreatingClient(CodebuffClient):
            def __init__(self) -> None:
                super().__init__(
                    Settings(
                        codebuff_token="token",
                        local_api_key=None,
                        request_timeout=1,
                    )
                )
                self.created = []

            async def get_session(self, instance_id=None):
                return {"status": "none"}

            async def delete_session(self) -> None:
                return None

            async def create_session(self, model):
                self.created.append(model)
                return FreebuffSession(
                    instance_id=f"{model}-instance",
                    model=model,
                    remaining_ms=3_000_000,
                )

            async def request_ad_chain(self, messages=None, *, surface=None):
                return None

        client = _CreatingClient()
        client._rate_limit_cache["moonshotai/kimi-k2.7-code"] = RateLimit(
            model="moonshotai/kimi-k2.7-code",
            reset_at=datetime.now(tz=timezone.utc) - timedelta(seconds=1),
            retry_after_ms=0,
            raw_body="{}",
        )
        manager = SessionManager(client, client.settings)
        try:
            session = await manager.ensure_session("moonshotai/kimi-k2.7-code")
            self.assertEqual(session.instance_id, "moonshotai/kimi-k2.7-code-instance")
            self.assertEqual(client.created, ["moonshotai/kimi-k2.7-code"])
        finally:
            await client.aclose()


if __name__ == "__main__":
    unittest.main()
