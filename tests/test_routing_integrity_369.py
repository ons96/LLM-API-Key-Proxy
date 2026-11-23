"""Regression tests for the routing-integrity review tracked in #369.

Covers the findings verified against live code on 2026-11-22:
#370 post-output stream failover, #372 max_tokens rewrite, #373 truncated
response replacement, #379 preflight denial, #381 oversized reinstatement,
#382 telemetry units, #383 Retry-After parsing, #386 cache_control
cross-candidate mutation, #387 prompt-cache lookup determinism.
"""

import json
import sqlite3
import time

import pytest
import yaml

import src.proxy_app.router_core as router_core
from src.proxy_app.router_core import (
    ErrorCategory,
    ProviderCandidate,
    RouterCore,
    _parse_retry_after,
)
from src.rotator_library.client import preflight_allows


def _router(tmp_path, **extra):
    config = {
        "free_only_mode": True,
        "routing": {
            "default_cooldown_seconds": 1,
            "rate_limit_cooldown_seconds": 1,
        },
        "providers": {
            "provider_a": {
                "enabled": True,
                "free_tier": True,
                "free_tier_models": ["model-a"],
            },
            "provider_b": {
                "enabled": True,
                "free_tier": True,
                "free_tier_models": ["model-b"],
            },
        },
    }
    config.update(extra)
    path = tmp_path / "router_config.yaml"
    path.write_text(yaml.dump(config))
    return RouterCore(str(path))


def _candidate(provider="provider_a", model="model-a"):
    return ProviderCandidate(provider=provider, model=model)


class TestExplicitOutputLimits:
    """#372: an explicit caller max_tokens is forwarded unchanged."""

    def test_non_stream_payload_keeps_max_tokens(self, tmp_path):
        router = _router(tmp_path)
        request = {
            "model": "x",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "hi"}],
        }
        payload = router._prepare_candidate_payload(request, _candidate(), "req")
        assert payload["max_tokens"] == 4096
        assert request["max_tokens"] == 4096

    def test_stream_flag_not_leaked_and_limit_preserved(self, tmp_path):
        router = _router(tmp_path)
        request = {
            "model": "x",
            "max_tokens": 8192,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        }
        payload = router._prepare_candidate_payload(request, _candidate(), "req")
        assert payload["max_tokens"] == 8192
        assert "stream" not in payload
        assert payload["model"] == "provider_a/model-a"


class TestCacheDirectiveIsolation:
    """#386: sanitizing one candidate must not mutate the original request."""

    def test_original_request_untouched_across_candidates(self, tmp_path):
        router = _router(tmp_path)
        request = {
            "model": "x",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hi",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "parameters": {"type": "object", "cache_control": "x"},
                    },
                }
            ],
        }
        snapshot = json.dumps(request, sort_keys=True)

        first = router._prepare_candidate_payload(request, _candidate(), "req-a")
        router._prepare_candidate_payload(
            request, _candidate("provider_b", "model-b"), "req-b"
        )

        assert json.dumps(request, sort_keys=True) == snapshot, (
            "per-candidate sanitizing must not mutate the caller's request"
        )
        assert "cache_control" not in first["messages"][0]["content"][0]
        assert "cache_control" not in first["tools"][0]["function"]["parameters"]


class TestTelemetryUnits:
    """#382: callers pass milliseconds; stored latency and TPS must match."""

    @pytest.mark.asyncio
    async def test_latency_and_tps_are_not_scaled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TELEMETRY_DB_PATH", str(tmp_path / "telemetry.db"))
        router = _router(tmp_path)
        captured = {}

        class FakeTelemetry:
            def record_call(self, **kwargs):
                captured.update(kwargs)

            def record_tps(self, *args, **kwargs):
                pass

        monkeypatch.setattr(router_core, "get_telemetry_manager", lambda: FakeTelemetry())

        await router._update_metrics(
            "provider_a",
            "model-a",
            2000,
            True,
            response={"usage": {"prompt_tokens": 10, "completion_tokens": 100}},
        )

        assert captured["response_time_ms"] == 2000
        assert captured["tokens_per_second"] == pytest.approx(50.0)


class TestRetryAfterParsing:
    """#383: delta-seconds and HTTP-date forms, including zero."""

    def test_delta_forms(self):
        assert _parse_retry_after("30") == 30
        assert _parse_retry_after("0") == 0
        assert _parse_retry_after("1.5") == 1

    def test_http_date_forms(self):
        from email.utils import formatdate

        future = formatdate(time.time() + 45, usegmt=True)
        assert 40 <= _parse_retry_after(future) <= 45
        past = formatdate(time.time() - 60, usegmt=True)
        assert _parse_retry_after(past) == 0

    def test_missing_malformed_and_clamped(self):
        assert _parse_retry_after(None) is None
        assert _parse_retry_after("") is None
        assert _parse_retry_after("soon") is None
        assert _parse_retry_after("999999") == 300

    @pytest.mark.asyncio
    async def test_zero_retry_after_is_honored(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TELEMETRY_DB_PATH", str(tmp_path / "telemetry.db"))
        router = _router(tmp_path)

        class Response:
            headers = {"retry-after": "0"}

        class Error(Exception):
            status_code = 429
            response = Response()

        category, retry_after = await router._classify_error(Error("429"))
        assert category is ErrorCategory.RATE_LIMIT
        assert retry_after == 0

    @pytest.mark.asyncio
    async def test_error_headers_container_is_read(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TELEMETRY_DB_PATH", str(tmp_path / "telemetry.db"))
        router = _router(tmp_path)

        class Error(Exception):
            status_code = 429
            headers = {"Retry-After": "25"}

        category, retry_after = await router._classify_error(Error("429"))
        assert category is ErrorCategory.RATE_LIMIT
        assert retry_after == 25


class TestPromptCacheLookup:
    """#387: the newest warm row wins deterministically."""

    def test_newest_row_wins(self, tmp_path, monkeypatch):
        db = tmp_path / "telemetry.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE prefix_cache_state ("
            "provider TEXT NOT NULL, model TEXT NOT NULL, prefix_hash TEXT NOT NULL,"
            " last_seen_ts INTEGER NOT NULL, last_cached_tokens INTEGER DEFAULT 0,"
            " PRIMARY KEY (provider, model, prefix_hash))"
        )
        now = int(time.time())
        conn.executemany(
            "INSERT INTO prefix_cache_state "
            "(provider, model, prefix_hash, last_seen_ts) VALUES (?,?,?,?)",
            [
                ("provider_a", "model-a", "hash", now - 10),
                ("provider_b", "model-b", "hash", now - 1),
            ],
        )
        conn.commit()
        conn.close()

        monkeypatch.setenv("TELEMETRY_DB_PATH", str(db))
        router = _router(tmp_path)
        assert router._prompt_cache_lookup("hash") == ("provider_b", "model-b")


class TestPreflightDenial:
    """#379: a denied preflight must never dispatch."""

    def test_denied_result_is_not_allowed(self):
        assert preflight_allows({"ok": False, "reason": "cooldown_shared"}) is False

    def test_missing_key_defaults_to_deny(self):
        assert preflight_allows({}) is False
        assert preflight_allows(None) is False

    def test_granted_result_is_allowed(self):
        assert preflight_allows({"ok": True, "reason": ""}) is True


class TestStreamOutputIntegrity:
    """#370/#385: no cross-upstream failover after client-visible output."""

    @pytest.mark.asyncio
    async def test_no_failover_after_first_chunk(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TELEMETRY_DB_PATH", str(tmp_path / "telemetry.db"))
        router = _router(tmp_path)
        calls = []

        async def fake_stream(candidate, request, request_id):
            calls.append(candidate.provider)
            if candidate.provider == "provider_a":
                yield {"choices": [{"delta": {"content": "partial"}}]}
                raise RuntimeError("boom after output")
            yield {"choices": [{"delta": {"content": "other"}}]}

        monkeypatch.setattr(router, "_execute_single_candidate_stream", fake_stream)

        chunks = []
        gen = router._stream_with_fallback(
            [_candidate(), _candidate("provider_b", "model-b")],
            {"model": "x", "stream": True, "messages": []},
            "req",
        )
        with pytest.raises(RuntimeError, match="boom after output"):
            async for chunk in gen:
                chunks.append(chunk)

        assert chunks == [{"choices": [{"delta": {"content": "partial"}}]}]
        assert calls == ["provider_a"]

    @pytest.mark.asyncio
    async def test_pre_output_failure_still_fails_over(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TELEMETRY_DB_PATH", str(tmp_path / "telemetry.db"))
        router = _router(tmp_path)
        calls = []

        async def fake_stream(candidate, request, request_id):
            calls.append(candidate.provider)
            if candidate.provider == "provider_a":
                raise RuntimeError("died before output")
            yield {"choices": [{"delta": {"content": "from-b"}}]}

        monkeypatch.setattr(router, "_execute_single_candidate_stream", fake_stream)

        gen = router._stream_with_fallback(
            [_candidate(), _candidate("provider_b", "model-b")],
            {"model": "x", "stream": True, "messages": []},
            "req",
        )
        chunks = [chunk async for chunk in gen]

        assert calls == ["provider_a", "provider_b"]
        assert chunks == [{"choices": [{"delta": {"content": "from-b"}}]}]


class TestTruncatedResponseIsReturned:
    """#373: a finish_reason="length" response is returned unchanged."""

    @pytest.mark.asyncio
    async def test_original_response_preserved(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TELEMETRY_DB_PATH", str(tmp_path / "telemetry.db"))
        router = _router(tmp_path)
        response = {
            "id": "original-id",
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "truncated", "tool_calls": [{"id": "c1"}]},
                }
            ],
        }
        route_calls = []

        async def fake_candidates(model_id, requirements):
            return [_candidate()]

        async def fake_execute(candidate, request, request_id):
            return response

        async def fake_route(request, request_id):
            route_calls.append(request_id)
            return {"id": "continuation"}

        monkeypatch.setattr(router, "_get_candidates", fake_candidates)
        monkeypatch.setattr(router, "_execute_single_candidate", fake_execute)
        monkeypatch.setattr(router, "route_request", fake_route)

        result = await router._route_request_inner(
            {
                "model": "provider_a/model-a",
                "messages": [{"role": "user", "content": "hi"}],
            },
            "req",
        )

        assert result is response
        assert route_calls == [], "continuation must not issue a second request"
        assert result["choices"][0]["finish_reason"] == "length"
        assert result["usage"]["completion_tokens"] == 10
        assert not hasattr(router, "_handle_truncated_response")


class TestContextAdmission:
    """#380/#381: tools count toward admission; oversized never dispatched."""

    def test_tool_schemas_count_toward_context_estimate(self, tmp_path):
        router = _router(tmp_path)
        messages = [{"role": "user", "content": "hi"}]
        without_tools = router._extract_requirements({"messages": messages})
        with_tools = router._extract_requirements(
            {
                "messages": messages,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "description": "x" * 4000,
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        )
        assert with_tools.min_context_tokens > without_tools.min_context_tokens

    def test_tool_call_arguments_count(self, tmp_path):
        router = _router(tmp_path)
        base = router._extract_requirements(
            {"messages": [{"role": "user", "content": "hi"}]}
        )
        with_args = router._extract_requirements(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": json.dumps({"q": "y" * 4000}),
                                },
                            }
                        ],
                    },
                ]
            }
        )
        assert with_args.min_context_tokens > base.min_context_tokens

    @pytest.mark.asyncio
    async def test_no_fit_chain_raises_instead_of_dispatching(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("TELEMETRY_DB_PATH", str(tmp_path / "telemetry.db"))
        router = _router(
            tmp_path,
            router_models={
                "huge": {
                    "fallback_chain": [
                        {"provider": "provider_a", "model": "model-a"}
                    ]
                }
            },
        )
        dispatched = []

        async def fake_execute(candidate, request, request_id):
            dispatched.append(candidate.provider)
            return {"choices": []}

        monkeypatch.setattr(router, "_execute_single_candidate", fake_execute)

        request = {
            "model": "huge",
            "messages": [{"role": "user", "content": "z" * 200000}],
        }
        with pytest.raises(router_core.HTTPException) as excinfo:
            await router._route_request_inner(request, "req")

        assert excinfo.value.status_code == 400
        assert "context window" in excinfo.value.detail
        assert dispatched == []


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
