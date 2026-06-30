"""Standalone smoke test for llm_provider_credits additions to telemetry.py.

Bypasses pytest module-path issues — directly imports telemetry, runs each new
method against a tmp DB, asserts behavior. Used to gate commit before PR-A.

Run from project root: PYTHONPATH=/home/osees/CodingProjects/LLM-API-Key-Proxy python3 scripts/smoke_llm_provider_credits.py
"""

from __future__ import annotations

import sys
import os
import tempfile
import traceback

# Ensure src is importable; supports both running from project root and elsewhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # project root -> src/
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))

import importlib.util
_SRC = os.path.join(os.path.dirname(_HERE), "src", "rotator_library", "telemetry.py")
_spec = importlib.util.spec_from_file_location("telemetry_mod", _SRC)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TelemetryManager = _mod.TelemetryManager


def assert_eq(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def main() -> int:
    tmpdir = tempfile.mkdtemp(prefix="llm_credits_smoke_")
    db_path = os.path.join(tmpdir, "telemetry.db")
    tm = TelemetryManager(db_path=db_path)

    key_a = "sk-test-alpha-" + "x" * 30
    key_b = "sk-test-beta-" + "y" * 30

    # 1. register_llm_provider_credits for groq (daily_renewable, 14400)
    tm.register_llm_provider_credits(
        provider="groq",
        api_key=key_a,
        free_type="daily_renewable",
        initial_allowance=14400,
        reset_period="daily",
    )
    status = tm.get_llm_provider_credit_status("groq", key_a)
    assert_eq(status["free_type"], "daily_renewable", "register→status free_type")
    assert_eq(status["credits_remaining"], 14400.0, "register→remaining=initial")
    assert_eq(status["is_exhausted"], False, "register→not exhausted")
    assert_eq(status["reset_period"], "daily", "register→reset_period")

    # 2. check available
    assert tm.check_llm_provider_available("groq", key_a), "should be available"

    # 3. decrement after a successful call
    tm.decrement_llm_credentials("groq", key_a, credits_consumed=100, success=True)
    s = tm.get_llm_provider_credit_status("groq", key_a)
    assert_eq(s["credits_remaining"], 14300.0, "after decrement 100")
    assert_eq(s["credits_used_total"], 100, "credits_used_total tracks delta")

    # 4. mark exhausted
    tm.mark_llm_provider_exhausted("groq", key_a)
    s = tm.get_llm_provider_credit_status("groq", key_a)
    assert_eq(s["is_exhausted"], True, "marked exhausted")
    assert_eq(s["credits_remaining"], 0.0, "exhausted→remaining=0")
    assert not tm.check_llm_provider_available("groq", key_a), "must NOT be available"

    # 5. reset_daily resets a daily-registered exhausted key
    import datetime as _dt
    past_iso = (_dt.datetime.utcnow() - _dt.timedelta(days=2)).isoformat(timespec="seconds")
    tm.register_llm_provider_credits(
        provider="poe",
        api_key=key_b,
        free_type="daily_renewable",
        initial_allowance=1_000_000,
        reset_period="daily",
        reset_date=past_iso,
    )
    tm.decrement_llm_credentials("poe", key_b, credits_consumed=500, success=True)
    tm.mark_llm_provider_exhausted("poe", key_b)
    tm.reset_daily_llm_provider_credits("poe", key_b)
    s = tm.get_llm_provider_credit_status("poe", key_b)
    assert_eq(s["is_exhausted"], False, "reset→not exhausted")
    assert_eq(s["credits_remaining"], 1_000_000.0, "reset→full allowance back")

    # 6. unlimited_rate_limited stays -1 forever
    tm.register_llm_provider_credits(
        provider="cerebras",
        api_key=key_a,
        free_type="unlimited_rate_limited",
        initial_allowance=-1,
        reset_period="none",
    )
    s = tm.get_llm_provider_credit_status("cerebras", key_a)
    assert_eq(s["credits_remaining"], -1.0, "unlimited stays -1 on register")
    tm.decrement_llm_credentials("cerebras", key_a, credits_consumed=10_000_000, success=True)
    s = tm.get_llm_provider_credit_status("cerebras", key_a)
    assert_eq(s["credits_remaining"], -1.0, "unlimited never depleted")
    assert tm.check_llm_provider_available("cerebras", key_a), "unlimited always available"

    # 7. unknown provider defaults to unlimited (safe default)
    s = tm.get_llm_provider_credit_status("never-seen-provider", key_a)
    assert_eq(s["free_type"], "unlimited_rate_limited", "unknown→unlimited")
    assert_eq(s["credits_remaining"], -1.0, "unknown→credits_remaining -1")
    assert tm.check_llm_provider_available("never-seen-provider", key_a), "unknown available"

    # 8. separate keys independent
    tm.register_llm_provider_credits(
        provider="groq",
        api_key=key_b,
        free_type="daily_renewable",
        initial_allowance=100,
        reset_period="daily",
    )
    s_a = tm.get_llm_provider_credit_status("groq", key_a)
    s_b = tm.get_llm_provider_credit_status("groq", key_b)
    assert_eq(s_a["credits_remaining"], 0.0, "key A still exhausted")
    assert_eq(s_b["credits_remaining"], 100.0, "key B unaffected")

    print("OK — all 8 smoke checks passed against", db_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        traceback.print_exc()
        print(f"\nFAIL: {e}", file=sys.stderr)
        sys.exit(1)
