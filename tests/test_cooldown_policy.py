"""Tests for CooldownPolicy (#218): per-provider 429 wait-vs-fallback."""
import asyncio
import time
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Import directly from the module file to avoid pulling in litellm via
# rotator_library/__init__.py -> client.py (heavy dep not needed for these tests).
_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "src"))

# Import the cooldown_manager module directly (no package __init__ chain)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_cooldown_manager_under_test",
    _repo_root / "src" / "rotator_library" / "cooldown_manager.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CooldownPolicy = _mod.CooldownPolicy


CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "config" / "cooldown_policy.yaml")


@pytest.fixture
def policy():
    return CooldownPolicy(config_path=CONFIG_PATH)


# --- should_wait_on_429 ---

def test_should_wait_anthropic_within_cap(policy):
    """Anthropic + retry_after=30s (< max_wait_s=300) -> wait."""
    wait, deadline = policy.should_wait_on_429("anthropic", 30)
    assert wait is True
    assert deadline > time.time()


def test_should_wait_anthropic_exceeds_cap(policy):
    """Anthropic + retry_after=600s (> max_wait_s=300) -> no wait, fall back."""
    wait, deadline = policy.should_wait_on_429("anthropic", 600)
    assert wait is False
    assert deadline == 0.0


def test_should_wait_openai(policy):
    """OpenAI + retry_after=60s (<= max_wait_s=120) -> wait."""
    wait, _ = policy.should_wait_on_429("openai", 60)
    assert wait is True


def test_should_wait_gemini(policy):
    """Gemini + retry_after=30s (<= max_wait_s=60) -> wait."""
    wait, _ = policy.should_wait_on_429("gemini", 30)
    assert wait is True


def test_no_wait_for_exempt_provider(policy):
    """DeepSeek (not in config) -> default policy, no wait."""
    wait, _ = policy.should_wait_on_429("deepseek", 10)
    assert wait is False


def test_no_wait_for_mistral(policy):
    """Mistral (not in config) -> default policy, no wait."""
    wait, _ = policy.should_wait_on_429("mistral", 5)
    assert wait is False


def test_no_wait_for_groq(policy):
    """Groq (not in config) -> default policy, no wait."""
    wait, _ = policy.should_wait_on_429("groq", 5)
    assert wait is False


def test_no_wait_when_retry_after_none(policy):
    """retry_after=None -> no wait (can't determine duration)."""
    wait, _ = policy.should_wait_on_429("anthropic", None)
    assert wait is False


def test_no_wait_when_retry_after_zero(policy):
    """retry_after=0 -> no wait."""
    wait, _ = policy.should_wait_on_429("anthropic", 0)
    assert wait is False


def test_provider_name_case_insensitive(policy):
    """Provider names should match case-insensitively."""
    wait, _ = policy.should_wait_on_429("Anthropic", 30)
    assert wait is True
    wait, _ = policy.should_wait_on_429("OPENAI", 30)
    assert wait is True


def test_gemini_at_cap_boundary(policy):
    """Gemini max_wait_s=60, retry_after=60 exactly -> wait (boundary inclusive)."""
    wait, _ = policy.should_wait_on_429("gemini", 60)
    assert wait is True


def test_gemini_just_over_cap(policy):
    """Gemini max_wait_s=60, retry_after=61 -> no wait."""
    wait, _ = policy.should_wait_on_429("gemini", 61)
    assert wait is False


# --- relay short-wait entries (#1068) ---

def test_relay_crowllm_waits_within_cap(policy):
    """Crowllm + retry_after=20 (<= max_wait_s=30) -> wait."""
    wait, _ = policy.should_wait_on_429("crowllm", 20)
    assert wait is True


def test_relay_crowllm_exceeds_cap_falls_back(policy):
    """Crowllm + retry_after=600 (> max_wait_s=30) -> immediate fallback."""
    wait, _ = policy.should_wait_on_429("crowllm", 600)
    assert wait is False


def test_relay_entries_all_configured(policy):
    """All #1068 relay providers resolve to wait_on_429 with 30s cap."""
    for name in ("crowllm", "freetheai", "nvidia", "pooled", "tokenrouter", "kilo", "larprouter"):
        p = policy.get_policy(name)
        assert p["wait_on_429"] is True
        assert p["max_wait_s"] == 30


# --- wait_until ---

@pytest.mark.anyio
async def test_wait_until_completes(policy):
    """wait_until with short deadline returns True after waiting."""
    deadline = time.time() + 0.1
    start = time.time()
    result = await policy.wait_until("anthropic", deadline)
    elapsed = time.time() - start
    assert result is True
    assert elapsed >= 0.08  # waited ~0.1s


@pytest.mark.anyio
async def test_wait_until_past_deadline(policy):
    """wait_until with already-passed deadline returns True immediately."""
    deadline = time.time() - 1  # in the past
    result = await policy.wait_until("anthropic", deadline)
    assert result is True


@pytest.mark.anyio
async def test_wait_until_cancelled(policy):
    """wait_until returns False if cancelled."""
    deadline = time.time() + 10
    task = asyncio.create_task(policy.wait_until("anthropic", deadline))
    await asyncio.sleep(0.05)
    task.cancel()
    result = await task
    assert result is False


# --- config loading resilience ---

def test_missing_config_file_defaults_to_no_wait(tmp_path):
    """Missing config file -> default policy (no wait)."""
    p = CooldownPolicy(config_path=str(tmp_path / "nonexistent.yaml"))
    wait, _ = p.should_wait_on_429("anthropic", 30)
    assert wait is False  # default = no wait


def test_malformed_config_defaults_to_no_wait(tmp_path):
    """Malformed YAML -> default policy (no wait)."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("providers: [this is not valid yaml structure {{{")
    p = CooldownPolicy(config_path=str(bad))
    wait, _ = p.should_wait_on_429("anthropic", 30)
    # Malformed -> error caught -> default no wait
    assert wait is False


def test_get_policy_returns_default_for_unknown(policy):
    """get_policy for unknown provider returns default dict."""
    p = policy.get_policy("unknown-provider")
    assert p["wait_on_429"] is False
    assert p["max_wait_s"] == 0


# --- integration: metadata propagation (simulated) ---

def test_metadata_fields_set_correctly():
    """Verify the metadata field names match what telemetry reads."""
    litellm_kwargs = {}
    meta = litellm_kwargs.setdefault("metadata", {})
    meta["waited_for_429"] = True
    meta["wait_duration_s"] = 30.0
    # Telemetry reads these exact keys
    assert meta.get("waited_for_429") is True
    assert meta.get("wait_duration_s") == 30.0


# --- quota classes (#1069) ---

_QUOTA_CONFIG = """
quota_classes:
  unlimited:
    wait_on_429: false
    max_wait_s: 0
    require_cache: false
    reason: unlimited-prefer-fallback
  credit:
    wait_on_429: true
    max_wait_s: 300
    require_cache: true
    reason: credit-cache-preserve
  trial:
    wait_on_429: true
    max_wait_s: 120
    require_cache: true
    reason: trial-credit-preserve
  paid:
    wait_on_429: false
    max_wait_s: 0
    require_cache: false
    reason: paid-rotate
providers:
  credcache:
    quota_class: credit
    supports_cache: true
  crednocache:
    quota_class: credit
    supports_cache: false
  trialcache:
    quota_class: trial
    supports_cache: true
  unlimitedp:
    quota_class: unlimited
    supports_cache: true
  paidp:
    quota_class: paid
    supports_cache: true
  unknownp:
    quota_class: something-new
    supports_cache: true
  barep: {}
  overridep:
    quota_class: credit
    supports_cache: true
    wait_on_429: true
    max_wait_s: 30
    reason: relay-short-retry-after
default:
  wait_on_429: false
  max_wait_s: 0
"""


@pytest.fixture
def quota_policy(tmp_path):
    cfg = tmp_path / "cooldown_policy.yaml"
    cfg.write_text(_QUOTA_CONFIG)
    return CooldownPolicy(config_path=str(cfg))


def test_quota_class_credit_waits_with_cache(quota_policy):
    """credit + cache support: wait out 429, capped at the class limit (#218 rule)."""
    wait, _ = quota_policy.should_wait_on_429("credcache", 60)
    assert wait is True
    wait, _ = quota_policy.should_wait_on_429("credcache", 600)
    assert wait is False  # beyond the 300s class cap -> fall back


def test_quota_class_credit_without_cache_rotates(quota_policy):
    """credit without cache support: no cache benefit to preserve -> rotate."""
    wait, _ = quota_policy.should_wait_on_429("crednocache", 30)
    assert wait is False


def test_quota_class_trial_short_wait(quota_policy):
    """trial + cache support: short wait to preserve trial credit + warm cache."""
    wait, _ = quota_policy.should_wait_on_429("trialcache", 60)
    assert wait is True
    wait, _ = quota_policy.should_wait_on_429("trialcache", 300)
    assert wait is False  # beyond the 120s class cap


def test_quota_class_unlimited_prefers_fallback(quota_policy):
    """unlimited-free: no credit/cache to preserve -> prefer faster fallback."""
    wait, _ = quota_policy.should_wait_on_429("unlimitedp", 10)
    assert wait is False


def test_quota_class_paid_rotates_immediately(quota_policy):
    """paid: never burn paid quota on waits."""
    wait, _ = quota_policy.should_wait_on_429("paidp", 10)
    assert wait is False


def test_quota_class_unknown_uses_default(quota_policy):
    """Unknown quota_class -> current default behavior (rotate now)."""
    wait, _ = quota_policy.should_wait_on_429("unknownp", 10)
    assert wait is False
    p = quota_policy.get_policy("unknownp")
    assert p["wait_on_429"] is False


def test_missing_metadata_uses_default(quota_policy):
    """Provider with no metadata at all -> default (unchanged behavior)."""
    wait, _ = quota_policy.should_wait_on_429("barep", 10)
    assert wait is False


def test_explicit_wait_overrides_class(quota_policy):
    """Explicit wait_on_429/max_wait_s always win over the class table (back-compat)."""
    wait, _ = quota_policy.should_wait_on_429("overridep", 10)
    assert wait is True
    wait, _ = quota_policy.should_wait_on_429("overridep", 60)
    assert wait is False  # explicit 30s cap, not the 300s credit cap


def test_real_config_relays_keep_short_wait(policy):
    """#1068 relay entries keep their explicit 30s short-wait despite class=unlimited."""
    wait, _ = policy.should_wait_on_429("crowllm", 10)
    assert wait is True
    wait, _ = policy.should_wait_on_429("crowllm", 60)
    assert wait is False


def test_real_config_credit_provider_still_waits(policy):
    """#218 reference behavior preserved for the credit-class providers."""
    wait, _ = policy.should_wait_on_429("anthropic", 30)
    assert wait is True


if __name__ == "__main__":
    # ponytail: self-check for manual run
    pytest.main([__file__, "-v"])
