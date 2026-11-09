"""Tests for DB-derived static rate limits and preemptive quota enforcement."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from proxy_app.rate_limiter import (
    REASON_RATE_LIMIT,
    REASON_USAGE_CAP,
    RateLimitTracker,
)
from proxy_app.router_core import ProviderCandidate, RouterCore

from tests.fixtures.provider_mocks import MockProviderResponse
from tests.fixtures.scenarios import ensure_providers_enabled


@pytest.fixture
def tracker(tmp_path):
    return RateLimitTracker(active_days_path=tmp_path / "active_days.json")


def test_load_provider_limits_filters_invalid_values(monkeypatch, tmp_path):
    import proxy_app.router_core as router_core

    path = tmp_path / "provider_limits.yaml"
    path.write_text(
        "providers:\n"
        "  groq:\n"
        "    rpm: 30\n"
        "    daily_tokens: 14400\n"
        "  disabled:\n"
        "    rpm: 0\n"
        "    tpm: -1\n"
        "    daily: '10'\n"
    )
    monkeypatch.setattr(router_core, "_provider_limits_path", path)
    monkeypatch.setattr(router_core, "_provider_limits", None)

    assert router_core._load_provider_limits() == {
        "groq": {"rpm": 30, "daily_tokens": 14400}
    }


def test_compute_rate_limits_uses_static_defaults_and_overrides(monkeypatch):
    import proxy_app.router_core as router_core

    monkeypatch.setattr(
        router_core,
        "_load_provider_limits",
        lambda: {"provider_a": {"rpm": 5, "daily_tokens": 100}},
    )
    router = RouterCore.__new__(RouterCore)
    router.provider_configs = {
        "provider_a": {"rate_limits": {"rpm": 10, "tpm": 50}}
    }

    limits = router._compute_rate_limits("provider_a", "model_a", {})

    assert limits == {"rpm": 10, "tpm": 50, "daily_tokens": 100}


def test_crax_daily_token_and_rpm_caps_are_enforced():
    """Crax's free-daily allowance must have a local preemptive cap."""
    config_path = Path(__file__).parent.parent / "config" / "router_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    router = RouterCore.__new__(RouterCore)
    router.provider_configs = config["providers"]

    limits = router._compute_rate_limits("crax", "glm-5.3", {})

    assert limits["rpm"] == 200
    assert limits["daily_tokens"] == 20_000_000


@pytest.mark.asyncio
async def test_rpm_exhaustion_blocks_before_next_request(tracker):
    await tracker.record_request("p", "m")
    await tracker.record_request("p", "m")

    allowed, reason = await tracker.can_use_provider("p", "m", {"rpm": 2})

    assert allowed is False
    assert reason == REASON_RATE_LIMIT


@pytest.mark.asyncio
async def test_rpm_window_resets(tracker):
    await tracker.record_request("p", "m")
    status = tracker._usage["p/m"]
    status.requests_this_minute = 2
    status.last_reset_minute = time.time() - 61

    allowed, reason = await tracker.can_use_provider("p", "m", {"rpm": 2})

    assert allowed is True
    assert reason == ""


@pytest.mark.asyncio
async def test_rpm_limit_is_provider_wide_across_models(tracker):
    await tracker.record_request("p", "model_one")

    allowed, reason = await tracker.can_use_provider(
        "p", "model_two", {"rpm": 1}
    )

    assert allowed is False
    assert reason == REASON_RATE_LIMIT


@pytest.mark.asyncio
async def test_daily_tokens_are_provider_wide_across_models(tracker):
    await tracker.record_tokens("p", "model_one", 100)

    allowed, reason = await tracker.can_use_provider(
        "p", "model_two", {"daily_tokens": 100}
    )

    assert allowed is False
    assert reason == REASON_USAGE_CAP


@pytest.mark.asyncio
async def test_daily_tokens_exhaustion_blocks(tracker):
    await tracker.record_tokens("p", "m", 100)

    allowed, reason = await tracker.can_use_provider(
        "p", "m", {"daily_tokens": 100}
    )

    assert allowed is False
    assert reason == REASON_USAGE_CAP


@pytest.mark.asyncio
async def test_daily_tokens_reset_after_window(tracker):
    await tracker.record_tokens("p", "m", 100)
    tracker._usage["p/m"].last_reset_day = time.time() - 86401

    allowed, reason = await tracker.can_use_provider(
        "p", "m", {"daily_tokens": 100}
    )

    assert allowed is True
    assert reason == ""
    assert tracker._usage["p/m"].tokens_today == 0


@pytest.mark.asyncio
async def test_unknown_quota_is_allowed(tracker):
    allowed, reason = await tracker.can_use_provider("p", "m", {})

    assert allowed is True
    assert reason == ""


@pytest.mark.asyncio
async def test_rpm_limit_falls_back_to_next_provider(tmp_path, monkeypatch):
    import proxy_app.router_core as router_core

    config = {
        "free_only_mode": True,
        "router_models": {
            "coding-smart": {
                "candidates": [
                    {
                        "provider": "provider_a",
                        "model": "model_a",
                        "priority": 1,
                        "free_tier_only": True,
                    },
                    {
                        "provider": "provider_b",
                        "model": "model_b",
                        "priority": 2,
                        "free_tier_only": True,
                    },
                ]
            }
        },
        "routing": {"default_cooldown_seconds": 60},
    }
    config = ensure_providers_enabled(config)
    for provider in ("provider_a", "provider_b"):
        config["providers"][provider].update({"free_tier": True})

    config_file = tmp_path / "router_config.yaml"
    config_file.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        router_core,
        "_load_provider_limits",
        lambda: {"provider_a": {"rpm": 1}, "provider_b": {"rpm": 100}},
    )
    router = RouterCore(str(config_file))
    calls = []

    async def fake_execute(candidate, request, request_id):
        calls.append(candidate.provider)
        await router.rate_limiter.record_request(candidate.provider, candidate.model)
        return MockProviderResponse(model=f"{candidate.provider}/{candidate.model}")

    request = {
        "model": "coding-smart",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
    }
    with patch.object(router, "_execute_single_candidate", side_effect=fake_execute):
        await router.route_request(request, "rpm-fallback-1")
        await router.route_request(request, "rpm-fallback-2")

    assert calls == ["provider_a", "provider_b"]


@pytest.mark.asyncio
async def test_stream_usage_is_recorded(tmp_path):
    config = {
        "free_only_mode": False,
        "providers": {"provider_a": {"enabled": True}},
        "router_models": {},
    }
    config_file = tmp_path / "router_config.yaml"
    config_file.write_text(yaml.safe_dump(config))
    router = RouterCore(str(config_file))
    candidate = ProviderCandidate(
        provider="provider_a", model="model_a", priority=1, capabilities=set()
    )

    async def chunk_stream():
        yield {"choices": [{"delta": {"content": "hi"}}]}
        yield {
            "choices": [],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            },
        }

    class FakeAdapter:
        async def chat_completions(self, request, stream=False):
            return chunk_stream()

    with patch.object(router, "max_retries_for", return_value=0), patch(
        "proxy_app.router_core.ProviderAdapterFactory.create_adapter",
        return_value=FakeAdapter(),
    ):
        chunks = [
            chunk
            async for chunk in router._execute_single_candidate_stream(
                candidate,
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
                "stream-usage",
            )
        ]

    assert len(chunks) == 2
    assert router.rate_limiter._usage["provider_a/model_a"].tokens_today == 10
