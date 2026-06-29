"""CLI verification script for active-day window tracking in RateLimitTracker.

Mirrors scripts/verify_rate_limiter.py: each test prints PASSED/FAILED and
returns bool; main() exits 1 on any failure.

Usage:
    python scripts/verify_active_days.py
"""

import asyncio
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add src to path (mirrors verify_rate_limiter.py)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from proxy_app.rate_limiter import (  # noqa: E402
    REASON_USAGE_CAP,
    RateLimitTracker,
)


def _today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _offset_day(days_ago: int) -> str:
    return (datetime.now(timezone.utc).date() - timedelta(days=days_ago)).isoformat()


async def test_unconfigured_provider(tracker: RateLimitTracker) -> bool:
    print("Testing unconfigured provider...")
    allowed, reason, info = tracker.check_active_days("noconfig")
    if not allowed or reason != "" or info != {}:
        print(f"FAILED: expected (True, '', {{}}), got ({allowed}, {reason!r}, {info!r})")
        return False
    print("PASSED: unconfigured provider always allowed")
    return True


async def test_under_cap_allowed(tracker: RateLimitTracker) -> bool:
    print("Testing under cap allowed...")
    tracker.configure_active_days_windows(
        {"prov_under": {"window_days": 12, "max_active_days": 3, "tz": "UTC"}}
    )
    # Seed two prior days, leave today unused.
    tracker._active_days["prov_under"] = {_offset_day(5), _offset_day(2)}
    allowed, reason, info = tracker.check_active_days("prov_under")
    if not allowed:
        print(f"FAILED: expected allowed, got reason={reason!r} info={info!r}")
        return False
    if info.get("used") != 2 or info.get("limit") != 3 or info.get("remaining") != 1:
        print(f"FAILED: expected used=2/3 remaining=1, got {info!r}")
        return False
    print(f"PASSED: under cap allowed (used={info['used']}/{info['limit']})")
    return True


async def test_at_cap_blocked(tracker: RateLimitTracker) -> bool:
    print("Testing at cap blocked with USAGE_CAP...")
    tracker.configure_active_days_windows(
        {"prov_cap": {"window_days": 12, "max_active_days": 3, "tz": "UTC"}}
    )
    today = _today_utc()
    # Seed 3 active days: today + two recent prior days. We expect blocked.
    tracker._active_days["prov_cap"] = {today, _offset_day(1), _offset_day(3)}
    allowed, reason, info = tracker.check_active_days("prov_cap")
    if allowed:
        print(f"FAILED: expected blocked, got allowed info={info!r}")
        return False
    if reason != REASON_USAGE_CAP:
        print(f"FAILED: expected reason={REASON_USAGE_CAP!r}, got {reason!r}")
        return False
    unlock_iso = info.get("next_slot_unlocks_on")
    if not unlock_iso:
        print(f"FAILED: expected next_slot_unlocks_on to be set, got {info!r}")
        return False
    # unlock should be exactly 12 days after oldest (which is _offset_day(3))
    expected_unlock_date = (
        datetime.now(timezone.utc).date() - timedelta(days=3) + timedelta(days=12)
    )
    expected_iso = datetime.combine(
        expected_unlock_date, datetime.min.time(), tzinfo=timezone.utc
    ).isoformat()
    if unlock_iso != expected_iso:
        print(f"FAILED: expected unlock {expected_iso}, got {unlock_iso}")
        return False
    print(f"PASSED: at cap blocked, unlocks {unlock_iso}")
    return True


async def test_window_rolls_unblock(tracker: RateLimitTracker) -> bool:
    print("Testing window roll unblocks...")
    tracker.configure_active_days_windows(
        {"prov_roll": {"window_days": 12, "max_active_days": 3, "tz": "UTC"}}
    )
    # Seed 3 days where the OLDEST is exactly 12 days old -> should be pruned
    # at the start of today. The other two (yesterday and 5 days ago) remain.
    tracker._active_days["prov_roll"] = {
        _offset_day(12),  # boundary, will be pruned
        _offset_day(5),
        _offset_day(1),
    }
    allowed, reason, info = tracker.check_active_days("prov_roll")
    if not allowed:
        print(f"FAILED: expected allowed after prune, got reason={reason!r} info={info!r}")
        return False
    if info.get("used") != 2:
        print(f"FAILED: expected used=2 after prune, got {info!r}")
        return False
    print(f"PASSED: window roll pruned oldest, count={info['used']}/{info['limit']}")
    return True


async def test_persistence_round_trip() -> bool:
    print("Testing persistence round-trip...")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "active_days.json"
        t1 = RateLimitTracker(active_days_path=path)
        t1.configure_active_days_windows(
            {"prov_persist": {"window_days": 12, "max_active_days": 3, "tz": "UTC"}}
        )
        # Stamp two days
        await t1.record_active_day("prov_persist", _offset_day(4))
        await t1.record_active_day("prov_persist", _offset_day(1))
        if not path.exists():
            print(f"FAILED: expected {path} to exist after record")
            return False
        # Reload
        t2 = RateLimitTracker(active_days_path=path)
        # Config is reloaded from yaml in production; in this CLI we
        # re-configure explicitly because the persistence file only
        # contains the dates, not the window rules.
        t2.configure_active_days_windows(
            {"prov_persist": {"window_days": 12, "max_active_days": 3, "tz": "UTC"}}
        )
        loaded = t2._active_days.get("prov_persist", set())
        if loaded != {_offset_day(4), _offset_day(1)}:
            print(f"FAILED: expected reload to restore dates, got {loaded!r}")
            return False
        # Sanity: check returns the right info
        allowed, reason, info = t2.check_active_days("prov_persist")
        if not allowed or info.get("used") != 2:
            print(f"FAILED: expected allowed used=2 after reload, got {info!r}")
            return False
    print("PASSED: persistence round-trip")
    return True


async def test_can_use_provider_integration() -> bool:
    print("Testing can_use_provider integration with active-day cap...")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "active_days.json"
        tracker = RateLimitTracker(active_days_path=path)
        tracker.configure_active_days_windows(
            {"g4f": {"window_days": 12, "max_active_days": 3, "tz": "UTC"}}
        )
        # Seed 3 days including today -> next request should be blocked
        tracker._active_days["g4f"] = {
            _today_utc(),
            _offset_day(2),
            _offset_day(5),
        }
        allowed, reason = await tracker.can_use_provider(
            "g4f", "gpt-4o", limits={}
        )
        if allowed:
            print("FAILED: expected can_use_provider to block at cap")
            return False
        if reason != REASON_USAGE_CAP:
            print(f"FAILED: expected reason={REASON_USAGE_CAP!r}, got {reason!r}")
            return False
        # Under cap should pass
        tracker._active_days["g4f"] = {_offset_day(1), _offset_day(4)}
        allowed, reason = await tracker.can_use_provider(
            "g4f", "gpt-4o", limits={}
        )
        if not allowed or reason != "":
            print(f"FAILED: expected allowed under cap, got ({allowed}, {reason!r})")
            return False
    print("PASSED: can_use_provider honors active-day cap")
    return True


async def main() -> int:
    tracker = RateLimitTracker()  # in-memory only (no path side-effects)
    tests = [
        await test_unconfigured_provider(tracker),
        await test_under_cap_allowed(tracker),
        await test_at_cap_blocked(tracker),
        await test_window_rolls_unblock(tracker),
        await test_persistence_round_trip(),
        await test_can_use_provider_integration(),
    ]
    if all(tests):
        print("\nAll Active-Day Tests PASSED")
        return 0
    print("\nSome Active-Day Tests FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
