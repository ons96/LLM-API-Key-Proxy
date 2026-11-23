"""Request/token counters and usage-cap blocks must survive a restart (#377).

Without this, every restart starts each daily quota from zero, so a gateway
that bounces (config refresh, deploy, crash) silently overruns free tiers.
"""

import json

import pytest

import src.rotator_library.telemetry as telemetry
from src.proxy_app.rate_limiter import REASON_USAGE_CAP, RateLimitTracker


@pytest.fixture
def telemetry_db(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEMETRY_DB_PATH", str(tmp_path / "telemetry.db"))
    telemetry._telemetry_manager = None
    yield
    telemetry._telemetry_manager = None


def _tracker(tmp_path):
    return RateLimitTracker(active_days_path=tmp_path / "active_days.json")


@pytest.mark.asyncio
async def test_request_counters_survive_restart(tmp_path):
    limits = {"rpm": 5, "daily": 2}
    first = _tracker(tmp_path)
    assert (await first.reserve_request("prov", "model", limits, api_key="k1"))[0]
    assert (await first.reserve_request("prov", "model", limits, api_key="k1"))[0]
    assert (tmp_path / "rate_counters.json").exists()

    second = _tracker(tmp_path)
    allowed, _ = await second.reserve_request("prov", "model", limits, api_key="k1")
    assert not allowed
    # A sibling key keeps its own budget after the restore.
    assert (await second.reserve_request("prov", "model", limits, api_key="k2"))[0]


@pytest.mark.asyncio
async def test_token_counters_survive_restart(tmp_path):
    first = _tracker(tmp_path)
    await first.record_tokens("prov", "model", 60, api_key="k1")

    second = _tracker(tmp_path)
    allowed, _ = await second.can_use_provider(
        "prov", "model", {"tpm": 50, "daily_tokens": 100}, api_key="k1"
    )
    assert not allowed


@pytest.mark.asyncio
async def test_stale_minute_counters_are_not_restored(tmp_path):
    """A counter from a minute that already elapsed must not block anything."""
    stale = {
        "version": 1,
        "usage": {
            "prov/model": {
                "requests_this_minute": 99,
                "requests_today": 0,
                "tokens_this_minute": 999,
                "tokens_today": 0,
                "last_reset_minute": 1.0,
                "last_reset_day": 1.0,
            }
        },
        "key_usage": {},
    }
    (tmp_path / "rate_counters.json").write_text(json.dumps(stale))

    tracker = _tracker(tmp_path)
    allowed, _ = await tracker.can_use_provider("prov", "model", {"rpm": 1})
    assert allowed


@pytest.mark.asyncio
async def test_usage_cap_block_is_restored_from_telemetry(tmp_path, telemetry_db):
    first = _tracker(tmp_path)
    await first.record_rate_limit_hit(
        "prov", "model", retry_after=60, reason=REASON_USAGE_CAP
    )

    second = _tracker(tmp_path)
    allowed, reason = await second.can_use_provider("prov", "model", {})
    assert not allowed
    assert reason == REASON_USAGE_CAP
