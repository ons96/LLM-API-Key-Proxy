"""Tests for the active-day window tracking in RateLimitTracker.

Covers:
  * unconfigured provider short-circuit
  * under-cap / at-cap / over-cap behavior
  * sliding-window pruning
  * JSON persistence round-trip
  * can_use_provider integration blocks when cap exceeded
  * record_request stamps the active day
  * get_usage_stats includes active_days section
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, TypeVar

import pytest

from proxy_app.rate_limiter import (
    REASON_USAGE_CAP,
    RateLimitTracker,
)

T = TypeVar("T")


def _today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _offset_day(days_ago: int) -> str:
    return (datetime.now(timezone.utc).date() - timedelta(days=days_ago)).isoformat()


@pytest.fixture
def tracker(tmp_path: Path) -> RateLimitTracker:
    """A fresh RateLimitTracker with persistence rooted at tmp_path."""
    return RateLimitTracker(active_days_path=tmp_path / "active_days.json")


def test_unconfigured_provider_allowed(tracker: RateLimitTracker) -> None:
    allowed, reason, info = tracker.check_active_days("noconfig")
    assert allowed is True
    assert reason == ""
    assert info == {}


def test_under_cap_allowed(tracker: RateLimitTracker) -> None:
    tracker.configure_active_days_windows(
        {"p": {"window_days": 12, "max_active_days": 3, "tz": "UTC"}}
    )
    tracker._active_days["p"] = {_offset_day(5), _offset_day(2)}
    allowed, reason, info = tracker.check_active_days("p")
    assert allowed is True
    assert reason == ""
    assert info["used"] == 2
    assert info["limit"] == 3
    assert info["remaining"] == 1
    assert info["window_days"] == 12
    assert info["tz"] == "UTC"


def test_at_cap_blocks_with_unblock_date(tracker: RateLimitTracker) -> None:
    tracker.configure_active_days_windows(
        {"p": {"window_days": 12, "max_active_days": 3, "tz": "UTC"}}
    )
    today = _today_utc()
    tracker._active_days["p"] = {today, _offset_day(1), _offset_day(3)}
    allowed, reason, info = tracker.check_active_days("p")
    assert allowed is False
    assert reason == REASON_USAGE_CAP
    expected_unlock = (
        datetime.now(timezone.utc).date()
        - timedelta(days=3)
        + timedelta(days=12)
    )
    expected_iso = datetime.combine(
        expected_unlock, datetime.min.time(), tzinfo=timezone.utc
    ).isoformat()
    assert info["next_slot_unlocks_on"] == expected_iso
    assert info["oldest_active_date"] == _offset_day(3)
    assert sorted(info["active_dates"]) == sorted(
        tracker._active_days["p"]
    )


def test_window_roll_prunes_oldest(tracker: RateLimitTracker) -> None:
    tracker.configure_active_days_windows(
        {"p": {"window_days": 12, "max_active_days": 3, "tz": "UTC"}}
    )
    tracker._active_days["p"] = {
        _offset_day(12),  # boundary
        _offset_day(5),
        _offset_day(1),
    }
    allowed, reason, info = tracker.check_active_days("p")
    assert allowed is True
    assert info["used"] == 2
    assert _offset_day(12) not in info["active_dates"]


def test_record_active_day_dedup(tracker: RateLimitTracker) -> None:
    tracker.configure_active_days_windows(
        {"p": {"window_days": 12, "max_active_days": 3, "tz": "UTC"}}
    )
    # First call returns True (new), second call returns False (already there).
    assert asyncio_run(tracker.record_active_day("p", _today_utc())) is True
    assert asyncio_run(tracker.record_active_day("p", _today_utc())) is False
    assert tracker._active_days["p"] == {_today_utc()}


def test_record_active_day_persists(tracker: RateLimitTracker, tmp_path: Path) -> None:
    tracker.configure_active_days_windows(
        {"p": {"window_days": 12, "max_active_days": 3, "tz": "UTC"}}
    )
    asyncio_run(tracker.record_active_day("p", _offset_day(2)))
    path = tmp_path / "active_days.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["version"] == 1
    assert _offset_day(2) in data["providers"]["p"]["dates"]


def test_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "active_days.json"
    t1 = RateLimitTracker(active_days_path=path)
    t1.configure_active_days_windows(
        {"p": {"window_days": 12, "max_active_days": 3, "tz": "UTC"}}
    )
    asyncio_run(t1.record_active_day("p", _offset_day(4)))
    asyncio_run(t1.record_active_day("p", _offset_day(1)))
    # New instance should pick up the persisted state.
    t2 = RateLimitTracker(active_days_path=path)
    assert t2._active_days.get("p") == {_offset_day(4), _offset_day(1)}


def test_can_use_provider_blocks_at_cap(tracker: RateLimitTracker) -> None:
    tracker.configure_active_days_windows(
        {"g4f": {"window_days": 12, "max_active_days": 3, "tz": "UTC"}}
    )
    tracker._active_days["g4f"] = {
        _today_utc(),
        _offset_day(2),
        _offset_day(5),
    }
    allowed, reason = asyncio_run(
        tracker.can_use_provider("g4f", "gpt-4o", limits={})
    )
    assert allowed is False
    assert reason == REASON_USAGE_CAP


def test_can_use_provider_allows_under_cap(tracker: RateLimitTracker) -> None:
    tracker.configure_active_days_windows(
        {"g4f": {"window_days": 12, "max_active_days": 3, "tz": "UTC"}}
    )
    tracker._active_days["g4f"] = {_offset_day(1), _offset_day(4)}
    allowed, reason = asyncio_run(
        tracker.can_use_provider("g4f", "gpt-4o", limits={})
    )
    assert allowed is True
    assert reason == ""


def test_record_request_stamps_active_day(tracker: RateLimitTracker) -> None:
    tracker.configure_active_days_windows(
        {"g4f": {"window_days": 12, "max_active_days": 3, "tz": "UTC"}}
    )
    asyncio_run(tracker.record_request("g4f", "gpt-4o"))
    assert _today_utc() in tracker._active_days["g4f"]


def test_get_usage_stats_includes_active_days(tracker: RateLimitTracker) -> None:
    tracker.configure_active_days_windows(
        {"g4f": {"window_days": 12, "max_active_days": 3, "tz": "UTC"}}
    )
    asyncio_run(tracker.record_active_day("g4f", _offset_day(2)))
    stats = asyncio_run(tracker.get_usage_stats())
    assert "active_days" in stats
    assert "g4f" in stats["active_days"]
    g4f_info = stats["active_days"]["g4f"]
    assert g4f_info["configured"] is True
    assert g4f_info["used"] == 1
    assert g4f_info["limit"] == 3


def test_unconfigured_provider_does_not_persist(tracker: RateLimitTracker, tmp_path: Path) -> None:
    # Recording active day for a provider without config is a no-op and
    # must not create or modify the on-disk state file.
    result = asyncio_run(tracker.record_active_day("noconfig", _today_utc()))
    assert result is False
    path = tmp_path / "active_days.json"
    assert not path.exists()


def test_record_active_day_prunes_on_record(tracker: RateLimitTracker) -> None:
    tracker.configure_active_days_windows(
        {"p": {"window_days": 12, "max_active_days": 3, "tz": "UTC"}}
    )
    # Pre-seed with a date that is already outside the window.
    tracker._active_days["p"] = {_offset_day(30)}
    asyncio_run(tracker.record_active_day("p", _today_utc()))
    # The 30-day-old entry should be pruned on record; only today remains.
    assert tracker._active_days["p"] == {_today_utc()}


# ---------------------------------------------------------------------------- #
# Async runner shim for sync tests                                            #
# ---------------------------------------------------------------------------- #

def asyncio_run(coro: Awaitable[T]) -> T:
    """Run a coroutine to completion in tests (avoids pytest-asyncio dep)."""
    import asyncio

    return asyncio.run(coro)
