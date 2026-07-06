"""
Tests for LLM provider free_type credit tracking (task-board #290).

Covers the 6 TelemetryManager methods:
  - register_llm_provider_credits
  - decrement_llm_credentials
  - get_llm_provider_credit_status
  - check_llm_provider_available
  - mark_llm_provider_exhausted
  - reset_daily_llm_provider_credits
"""

import sqlite3
import sys
import tempfile
import os
import datetime as _dt

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.join(ROOT, "src")
for _p in (ROOT, SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.rotator_library.telemetry import TelemetryManager


@pytest.fixture
def tm():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    m = TelemetryManager(db_path=path)
    yield m
    try:
        os.unlink(path)
    except OSError:
        pass


def _key(s: str) -> str:
    """Stable fake API key per provider for hashing."""
    return f"fake-key-{s}"


def test_register_and_get_status_for_daily_renewable(tm):
    tm.register_llm_provider_credits(
        provider="groq",
        api_key=_key("groq"),
        free_type="daily_renewable",
        initial_allowance=14400,
        reset_period="daily",
        reset_date="2099-01-01",
    )
    st = tm.get_llm_provider_credit_status("groq", _key("groq"))
    assert st["free_type"] == "daily_renewable"
    assert st["credits_remaining"] == 14400
    assert st["initial_allowance"] == 14400
    assert st["reset_period"] == "daily"
    assert st["is_exhausted"] is False


def test_register_unlimited_provider_remains_negative(tm):
    tm.register_llm_provider_credits(
        provider="cerebras",
        api_key=_key("cerebras"),
        free_type="unlimited_rate_limited",
        initial_allowance=-1,
        reset_period="never",
    )
    st = tm.get_llm_provider_credit_status("cerebras", _key("cerebras"))
    assert st["free_type"] == "unlimited_rate_limited"
    assert st["credits_remaining"] == -1
    assert tm.check_llm_provider_available("cerebras", _key("cerebras")) is True


def test_decrement_reduces_balance_and_flags_exhaustion(tm):
    tm.register_llm_provider_credits(
        provider="groq",
        api_key=_key("groq"),
        free_type="daily_renewable",
        initial_allowance=10.0,
        reset_period="daily",
        reset_date="2099-01-01",
    )
    for _ in range(9):
        tm.decrement_llm_credentials(
            provider="groq", api_key=_key("groq"), model="llama-3.1-8b-instant", credits_consumed=1.0
        )
    st = tm.get_llm_provider_credit_status("groq", _key("groq"))
    assert st["credits_remaining"] == 1.0
    assert st["credits_used_total"] == 9.0
    assert st["is_exhausted"] is False

    # One more → should empty and mark exhausted
    tm.decrement_llm_credentials(
        provider="groq", api_key=_key("groq"), model="llama-3.1-8b-instant"
    )
    st = tm.get_llm_provider_credit_status("groq", _key("groq"))
    assert st["credits_remaining"] == 0
    assert st["is_exhausted"] is True
    assert tm.check_llm_provider_available("groq", _key("groq")) is False


def test_decrement_unlimited_does_not_change_remaining(tm):
    """credits_remaining=-1 means 'unlimited'; decrement must NOT mutate it."""
    tm.register_llm_provider_credits(
        provider="cerebras",
        api_key=_key("cerebras"),
        free_type="unlimited_rate_limited",
        initial_allowance=-1,
        reset_period="never",
    )
    for _ in range(3):
        tm.decrement_llm_credentials(
            provider="cerebras",
            api_key=_key("cerebras"),
            model="llama-3.3-70b",
        )
    st = tm.get_llm_provider_credit_status("cerebras", _key("cerebras"))
    assert st["credits_remaining"] == -1
    assert st["is_exhausted"] is False


def test_check_available_unknown_provider_returns_true_by_default(tm):
    """Unregistered providers default to unlimited (safer than blocking all)."""
    assert tm.check_llm_provider_available("brand-new-provider", "fake-key") is True
    st = tm.get_llm_provider_credit_status("brand-new-provider", "fake-key")
    assert st["free_type"] == "unlimited_rate_limited"


def test_mark_exhausted_blocks_check(tm):
    tm.register_llm_provider_credits(
        provider="together",
        api_key=_key("together"),
        free_type="one_time_credit",
        initial_allowance=5.0,
        reset_period="never",
    )
    assert tm.check_llm_provider_available("together", _key("together")) is True
    tm.mark_llm_provider_exhausted("together", _key("together"))
    assert tm.check_llm_provider_available("together", _key("together")) is False
    st = tm.get_llm_provider_credit_status("together", _key("together"))
    assert st["is_exhausted"] is True


def test_reset_daily_recovers_exhausted_provider(tm):
    tm.register_llm_provider_credits(
        provider="groq",
        api_key=_key("groq"),
        free_type="daily_renewable",
        initial_allowance=100.0,
        reset_period="daily",
        reset_date="2000-01-01T00:00:00",  # past date so reset triggers
    )
    tm.mark_llm_provider_exhausted("groq", _key("groq"))
    assert tm.check_llm_provider_available("groq", _key("groq")) is False
    affected = tm.reset_daily_llm_provider_credits()
    assert affected >= 1
    st = tm.get_llm_provider_credit_status("groq", _key("groq"))
    assert st["credits_remaining"] == 100.0
    assert st["is_exhausted"] is False


def test_separate_keys_track_independently(tm):
    """Two API keys on the same provider are independent rows."""
    k1, k2 = _key("a"), _key("b")
    tm.register_llm_provider_credits(
        provider="groq", api_key=k1,
        free_type="daily_renewable", initial_allowance=2.0,
        reset_period="daily", reset_date="2099-01-01",
    )
    tm.register_llm_provider_credits(
        provider="groq", api_key=k2,
        free_type="daily_renewable", initial_allowance=5.0,
        reset_period="daily", reset_date="2099-01-01",
    )
    tm.mark_llm_provider_exhausted("groq", k1)
    s1 = tm.get_llm_provider_credit_status("groq", k1)
    s2 = tm.get_llm_provider_credit_status("groq", k2)
    assert s1["is_exhausted"] is True
    assert s2["is_exhausted"] is False


# ---------------------------------------------------------------------------
# Phase 2 (#290) — glue-method + wiring acceptance tests
# ---------------------------------------------------------------------------

def test_record_llm_call_outcome_success_decrements_daily_renewable(tm):
    """Router-side success path: credit decrement on successful call."""
    tm.register_llm_provider_credits(
        provider="groq", api_key=_key("groq"),
        free_type="daily_renewable", initial_allowance=5.0,
        reset_period="daily", reset_date="2099-01-01",
    )
    tm.record_llm_call_outcome("groq", _key("groq"), success=True, model="llama-3.1-8b-instant")
    st = tm.get_llm_provider_credit_status("groq", _key("groq"))
    assert st["credits_remaining"] == 4.0
    assert st["is_exhausted"] is False


def test_record_llm_call_outcome_quota_marks_one_time_permanently_exhausted(tm):
    """Router skips permanently-exhausted one_time_credit provider after quota error."""
    tm.register_llm_provider_credits(
        provider="together", api_key=_key("together"),
        free_type="one_time_credit", initial_allowance=2.0,
        reset_period="never",
    )
    tm.record_llm_call_outcome(
        "together", _key("together"), success=False,
        error_type="quota_exceeded", model="qwen-2.5-72b",
    )
    st = tm.get_llm_provider_credit_status("together", _key("together"))
    assert st["is_exhausted"] is True
    # Permanence: reset_daily must NOT recover one_time_credit entries.
    affected = tm.reset_daily_llm_provider_credits()
    assert affected == 0
    st2 = tm.get_llm_provider_credit_status("together", _key("together"))
    assert st2["is_exhausted"] is True
    assert tm.check_llm_provider_available("together", _key("together")) is False


def test_record_llm_call_outcome_quota_on_daily_recovers_after_reset(tm):
    """daily_renewable: marked exhausted on quota, then reset_daily restores availability."""
    tm.register_llm_provider_credits(
        provider="groq", api_key=_key("groq"),
        free_type="daily_renewable", initial_allowance=10.0,
        reset_period="daily", reset_date="2000-01-01T00:00:00",  # past -> reset triggers
    )
    tm.record_llm_call_outcome(
        "groq", _key("groq"), success=False,
        error_type="quota_exceeded", model="llama-3.3-70b",
    )
    assert tm.check_llm_provider_available("groq", _key("groq")) is False
    affected = tm.reset_daily_llm_provider_credits()
    assert affected >= 1
    st = tm.get_llm_provider_credit_status("groq", _key("groq"))
    assert st["is_exhausted"] is False
    assert st["credits_remaining"] == 10.0
    assert tm.check_llm_provider_available("groq", _key("groq")) is True


def test_record_llm_call_outcome_noop_for_unlimited(tm):
    """unlimited_rate_limited providers: quota errors must NOT mark exhausted (backwards-compat)."""
    tm.register_llm_provider_credits(
        provider="cerebras", api_key=_key("cerebras"),
        free_type="unlimited_rate_limited", initial_allowance=-1,
        reset_period="never",
    )
    tm.record_llm_call_outcome(
        "cerebras", _key("cerebras"), success=False,
        error_type="quota_exceeded", model="llama-3.3-70b",
    )
    st = tm.get_llm_provider_credit_status("cerebras", _key("cerebras"))
    assert st["is_exhausted"] is False
    assert st["credits_remaining"] == -1
    assert tm.check_llm_provider_available("cerebras", _key("cerebras")) is True


def test_check_and_lazy_register_skips_for_unlimited(tm):
    """check_and_lazy_register: unlimited/freemium are always True, no row created for unlimited."""
    # No prior row -> default returns unlimited -> treated as unlimited -> no register, short-circuits True.
    assert tm.check_and_lazy_register(
        "cerebras", _key("cerebras"),
        free_type="unlimited_rate_limited", initial_allowance=-1.0,
    ) is True
    # Row never created because the helper short-circuits for unlimited
    st = tm.get_llm_provider_credit_status("cerebras", _key("cerebras"))
    assert st["free_type"] == "unlimited_rate_limited"


def test_check_and_lazy_register_registers_and_allows_daily_renewable(tm):
    """check_and_lazy_register: daily_renewable w/ no prior row -> register then available."""
    available = tm.check_and_lazy_register(
        "groq", _key("groq"),
        free_type="daily_renewable", initial_allowance=100.0,
        reset_period="daily", reset_date="2099-01-01",
    )
    assert available is True
    st = tm.get_llm_provider_credit_status("groq", _key("groq"))
    assert st["free_type"] == "daily_renewable"
    assert st["credits_remaining"] == 100.0


def test_backwards_compat_missing_free_type_defaults_unlimited(tm):
    """AC: free_type defaults to unlimited_rate_limited when missing -> never blocked."""
    # Pre-Phase-2 callers don't pass free_type anywhere; helper must default to unlimited.
    available = tm.check_and_lazy_register("legacyprovider", _key("legacy"))
    assert available is True
    st = tm.get_llm_provider_credit_status("legacyprovider", _key("legacy"))
    assert st["free_type"] == "unlimited_rate_limited"
    assert st["is_exhausted"] is False


def test_router_credit_check_uses_provider_database_fallback(tm, monkeypatch):
    """Router reads free_type from providers_database when router_config lacks it."""
    import rotator_library.telemetry as telemetry_module
    from proxy_app.router_core import RouterCore

    monkeypatch.setattr(telemetry_module, "_telemetry_manager", tm)
    router = RouterCore.__new__(RouterCore)
    router.config = {"providers": {"together": {}}}
    router._providers_db_cache = {
        "providers": {
            "together": {
                "free_type": "one_time_credit",
                "initial_allowance": 2.0,
                "reset_period": "never",
            }
        }
    }

    assert router._check_credit_exhaustion({}, "together", _key("together")) is True
    st = tm.get_llm_provider_credit_status("together", _key("together"))
    assert st["initial_allowance"] == 2.0
    assert st["reset_period"] == "never"
    tm.mark_llm_provider_exhausted("together", _key("together"))
    assert router._check_credit_exhaustion({}, "together", _key("together")) is False


def test_client_credit_check_uses_provider_database_fields(tm, monkeypatch):
    """Client path lazy-registers tracked providers with DB allowance/reset fields."""
    import src.rotator_library.telemetry as telemetry_module
    from src.rotator_library.client import RotatingClient

    monkeypatch.setattr(telemetry_module, "_telemetry_manager", tm)
    client = RotatingClient.__new__(RotatingClient)
    client._providers_db_cache = {
        "providers": {
            "groq": {
                "free_type": "daily_renewable",
                "initial_allowance": 7.0,
                "reset_period": "daily",
            }
        }
    }

    assert client._is_cred_credit_available("groq", _key("groq")) is True
    st = tm.get_llm_provider_credit_status("groq", _key("groq"))
    assert st["free_type"] == "daily_renewable"
    assert st["initial_allowance"] == 7.0
    assert st["reset_period"] == "daily"
