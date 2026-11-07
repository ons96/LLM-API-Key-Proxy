"""Per-key rate-limit counters and selector key rotation."""

import pytest

from src.proxy_app.rate_limiter import REASON_RATE_LIMIT, RateLimitTracker
from src.proxy_app.router_core import RouterCore


@pytest.fixture
def tracker(tmp_path):
    return RateLimitTracker(active_days_path=tmp_path / "active_days.json")


@pytest.mark.asyncio
async def test_rpm_and_daily_requests_are_isolated_per_key(tracker):
    limits = {"rpm": 1, "daily": 2}

    assert (await tracker.reserve_request("prov", "model", limits, api_key="key-a"))[0]
    assert not (
        await tracker.reserve_request("prov", "model", limits, api_key="key-a")
    )[0]
    assert (await tracker.reserve_request("prov", "model", limits, api_key="key-b"))[0]

    stats = await tracker.get_usage_stats()
    assert stats["usage"]["prov/model"]["rpm"] == 2
    assert len(stats["key_usage"]) == 2
    assert all("key-a" not in name for name in stats["key_usage"])


@pytest.mark.asyncio
async def test_tpm_and_daily_tokens_are_isolated_per_key(tracker):
    limits = {"tpm": 10, "daily_tokens": 10}

    await tracker.record_tokens("prov", "model", 6, api_key="key-a")
    assert (await tracker.can_use_provider("prov", "model", limits, api_key="key-a"))[0]

    await tracker.record_tokens("prov", "model", 5, api_key="key-a")
    allowed, reason = await tracker.can_use_provider(
        "prov", "model", limits, api_key="key-a"
    )
    assert not allowed
    assert reason == REASON_RATE_LIMIT
    assert (await tracker.can_use_provider("prov", "model", limits, api_key="key-b"))[0]

    daily_limits = {"tpm": 100, "daily_tokens": 10}
    await tracker.record_tokens("prov", "daily", 11, api_key="key-a")
    allowed, reason = await tracker.can_use_provider(
        "prov", "daily", daily_limits, api_key="key-a"
    )
    assert not allowed
    assert reason != REASON_RATE_LIMIT
    assert (
        await tracker.can_use_provider(
            "prov", "daily", daily_limits, api_key="key-b"
        )
    )[0]


@pytest.mark.asyncio
async def test_key_rate_limit_hit_does_not_block_sibling_key(tracker):
    await tracker.record_rate_limit_hit(
        "prov", "model", retry_after=60, api_key="key-a"
    )

    assert not (
        await tracker.can_use_provider("prov", "model", {}, api_key="key-a")
    )[0]
    assert (await tracker.can_use_provider("prov", "model", {}, api_key="key-b"))[0]

    stats = await tracker.get_usage_stats()
    assert any(item["limited"] for item in stats["key_usage"].values())
    assert all("key-a" not in name for name in stats["key_usage"])


def test_router_resolves_numbered_keys_in_order(monkeypatch):
    monkeypatch.setenv("TEST_PROV_API_KEY", "secret-one")
    monkeypatch.setenv("TEST_PROV_API_KEY_2", "secret-two")
    monkeypatch.setenv("TEST_PROV_API_KEY_1", "secret-one")

    router = RouterCore()
    keys = router._resolve_llm_api_keys(
        {"env_var": "TEST_PROV_API_KEY"}, "test_prov"
    )
    assert keys == ["secret-one", "secret-two"]


@pytest.mark.asyncio
async def test_router_reserves_next_key_when_first_is_at_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_PROV_API_KEY", "secret-one")
    monkeypatch.setenv("TEST_PROV_API_KEY_1", "secret-two")

    router = RouterCore()
    router.rate_limiter = RateLimitTracker(
        active_days_path=tmp_path / "active_days.json"
    )
    candidate_cfg = {"env_var": "TEST_PROV_API_KEY"}
    limits = {"rpm": 1}

    allowed, key, reason = await router._reserve_api_key(
        "test_prov", "model", candidate_cfg, limits
    )
    assert allowed and key == "secret-one" and reason == ""

    allowed, key, reason = await router._reserve_api_key(
        "test_prov", "model", candidate_cfg, limits
    )
    assert allowed and key == "secret-two" and reason == ""

    allowed, key, reason = await router._reserve_api_key(
        "test_prov", "model", candidate_cfg, limits
    )
    assert not allowed
    assert key is None
    assert reason == REASON_RATE_LIMIT
