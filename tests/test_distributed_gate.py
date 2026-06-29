"""Behavior tests for distributed_gate module (task-board #233).

Covers SharedCooldownStore (shared 429 state) and ConcurrencyGate
(per-(provider, model) slot counter + min-interval). These tests do NOT
need a live gateway or litellm keys -- everything is stdlib sqlite +
threading, deterministic via injected `now=`.
"""

from __future__ import annotations

import os
import tempfile

import pytest

# Skip this whole module if yaml is missing -- ConcurrencyGate degrade-gracefully
# without yaml but the resolver logic exercising config shouldn't be tested then.
yaml = pytest.importorskip("yaml")

import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from rotator_library.distributed_gate import (  # noqa: E402
    COOLDOWN_ROW_TTL_S,
    ConcurrencyGate,
    SharedCooldownStore,
)


# --- helpers ---------------------------------------------------------------
def _tmp_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".db", prefix="cooldown_test_")
    os.close(fd)
    os.unlink(path)  # let SharedCooldownStore create it cleanly
    return path


@pytest.fixture
def db_path() -> str:
    p = _tmp_db()
    try:
        yield p
    finally:
        try:
            os.unlink(p)
        except OSError:
            pass


# ===========================================================================
# SharedCooldownStore
# ===========================================================================
class TestSharedCooldownStore:
    def test_no_cooldown_initially(self, db_path):
        s = SharedCooldownStore(db_path=db_path, machine_id="host-a")
        assert s.is_cooling_down("groq", "llama-3.3-70b") is False
        assert s.cooldown_remaining("groq", "llama-3.3-70b") == 0.0

    def test_start_then_check(self, db_path):
        s = SharedCooldownStore(db_path=db_path, machine_id="host-a")
        now = 1_000_000.0
        s.start_cooldown("groq", "llama-3.3-70b", retry_after_s=60, now=now)
        assert s.is_cooling_down("groq", "llama-3.3-70b", now=now) is True
        remaining = s.cooldown_remaining("groq", "llama-3.3-70b", now=now)
        assert abs(remaining - 60.0) < 1e-6

    def test_different_model_same_provider_independent(self, db_path):
        s = SharedCooldownStore(db_path=db_path, machine_id="host-a")
        now = 1_000_000.0
        s.start_cooldown("groq", "llama-3.3-70b", retry_after_s=60, now=now)
        # spec: per-(provider,model), not per-provider -- different model is clean
        assert s.is_cooling_down("groq", "qwen3-32b", now=now) is False

    def test_expires_naturally(self, db_path):
        s = SharedCooldownStore(db_path=db_path, machine_id="host-a")
        now = 1_000_000.0
        s.start_cooldown("groq", "llama-3.3-70b", retry_after_s=60, now=now)
        assert s.is_cooling_down("groq", "llama-3.3-70b", now=now + 61) is False

    def test_upsert_keeps_later_deadline(self, db_path):
        """Two cooldowns on the same key MUST take MAX(t1, t2). Never shorten."""
        s = SharedCooldownStore(db_path=db_path, machine_id="host-a")
        now = 1_000_000.0
        s.start_cooldown("groq", "m", retry_after_s=120, now=now)  # longer
        s.start_cooldown("groq", "m", retry_after_s=10, now=now)   # shorter
        rem = s.cooldown_remaining("groq", "m", now=now)
        assert abs(rem - 120.0) < 1e-6, "shorter cooldown must not shorten the original"

    def test_purge_expired_removes_old_rows(self, db_path):
        s = SharedCooldownStore(db_path=db_path, machine_id="host-a")
        now = 1_000_000.0
        s.start_cooldown("x", "y", retry_after_s=1, now=now)
        # purge cutoff is `now - TTL` so use now well beyond row expiry+TTL
        removed = s.purge_expired(now=now + COOLDOWN_ROW_TTL_S + 10)
        assert removed >= 1
        assert s.is_cooling_down("x", "y", now=now + COOLDOWN_ROW_TTL_S + 10) is False

    def test_recent_cooldowns_returns_rows(self, db_path):
        s = SharedCooldownStore(db_path=db_path, machine_id="host-b")
        now = 1_000_000.0
        s.start_cooldown("groq", "m1", retry_after_s=60, now=now)
        s.start_cooldown("cerebras", "m2", retry_after_s=30, now=now)
        rows = s.recent_cooldowns(limit=10)
        assert len(rows) == 2
        providers = {r["provider"] for r in rows}
        assert providers == {"groq", "cerebras"}
        # column shape -- tuple indexing works (row_factory=Row)
        for r in rows:
            assert "cooldown_until_ts" in r.keys()
            assert "source_machine" in r.keys()
            assert r["source_machine"] == "host-b"

    def test_cross_process_sharing_two_stores_same_db(self, db_path):
        """Two processes / two clients / same file -> observably shared."""
        a = SharedCooldownStore(db_path=db_path, machine_id="host-A")
        b = SharedCooldownStore(db_path=db_path, machine_id="host-B")
        now = 1_000_000.0
        a.start_cooldown("groq", "m1", retry_after_s=60, now=now)
        # B sees what A wrote (this is the WHOLE POINT of the spec)
        assert b.is_cooling_down("groq", "m1", now=now) is True
        # B's own write visible from A
        b.start_cooldown("cerebras", "m2", retry_after_s=30, now=now)
        assert a.is_cooling_down("cerebras", "m2", now=now) is True


# ===========================================================================
# ConcurrencyGate
# ===========================================================================
class TestConcurrencyGate:
    def test_defaults_when_no_config(self):
        # missing yaml -> permissive defaults (max_concurrent=2, interval=0)
        g = ConcurrencyGate(config_path="/nonexistent/__no_such_file__.yaml")
        assert g.try_acquire("p", "m") is True   # 1/2
        assert g.try_acquire("p", "m") is True   # 2/2
        assert g.try_acquire("p", "m") is False  # full -> fall back
        g.release("p", "m")
        # After release, a slot reopens
        assert g.try_acquire("p", "m") is True

    def test_release_never_goes_negative(self):
        g = ConcurrencyGate(config_path="/nonexistent/policy.yaml")
        # extra releases without acquires must NOT push in_flight negative
        for _ in range(5):
            g.release("p", "m")
        snap = g.snapshot()
        total = sum(p[2] for p in snap["active_pairs"])
        assert total >= 0

    def test_different_models_on_same_provider_independent(self):
        g = ConcurrencyGate(config_path="/nonexistent/policy.yaml")
        # fill "p"/"m" to default max=2
        assert g.try_acquire("p", "m") is True
        assert g.try_acquire("p", "m") is True
        assert g.try_acquire("p", "m") is False  # full
        # but "p"/"other" is untouched -- spec requirement
        assert g.try_acquire("p", "other") is True

    def test_snapshot_shape(self):
        g = ConcurrencyGate(config_path="/nonexistent/policy.yaml")
        assert g.snapshot() == {
            "active_pairs": [],
            "total_in_flight": 0,
            "tracked_pairs": 0,
        }
        g.try_acquire("groq", "llama-3.3-70b")
        g.try_acquire("cerebras", "gpt-oss-120b")
        snap = g.snapshot()
        assert snap["total_in_flight"] == 2
        assert snap["tracked_pairs"] == 2
        # (provider, model, in_flight, max) tuples
        keys = {p[:2] for p in snap["active_pairs"]}
        assert ("groq", "llama-3.3-70b") in keys
        assert ("cerebras", "gpt-oss-120b") in keys

    def test_snapshot_does_not_include_zero_inflight(self):
        g = ConcurrencyGate(config_path="/nonexistent/policy.yaml")
        g.try_acquire("p", "m")
        g.release("p", "m")
        snap = g.snapshot()
        # slot remains tracked but should not appear in active_pairs
        assert snap["tracked_pairs"] == 1
        assert snap["total_in_flight"] == 0
        assert snap["active_pairs"] == []

    def test_context_manager_auto_releases(self):
        g = ConcurrencyGate(config_path="/nonexistent/policy.yaml")
        with g.slot("p", "m") as ok:
            assert ok is True
            with g.slot("p", "m") as ok2:
                assert ok2 is True
                with g.slot("p", "m") as ok3:
                    # 3rd over default max 2 -> False
                    assert ok3 is False
        # all released on exit
        snap = g.snapshot()
        assert snap["total_in_flight"] == 0

    def test_min_interval_blocks_back_to_back(self):
        g = ConcurrencyGate(config_path="/nonexistent")
        # Inject config after construction -- simulates `concurrency_policy.yaml`
        g._provider_cfg = {"slow": {"max_concurrent": 5, "min_interval_ms": 200}}
        g._slots.clear()
        now = 1_000_000.0
        assert g.try_acquire("slow", "m", now=now) is True
        g.release("slow", "m")
        # 100ms < 200ms -> blocked
        assert g.try_acquire("slow", "m", now=now + 0.100) is False
        # enough time passed -> allowed
        assert g.try_acquire("slow", "m", now=now + 0.250) is True

    def test_model_override_higher_precedence_than_provider(self, tmp_path):
        cfg = tmp_path / "policy.yaml"
        cfg.write_text(
            "default:\n"
            "  max_concurrent: 2\n"
            "providers:\n"
            "  groq:\n"
            "    max_concurrent: 5\n"
            "models:\n"
            "  groq/llama-3.3-70b-versatile:\n"
            "    max_concurrent: 1\n"
        )
        g = ConcurrencyGate(config_path=str(cfg))
        # model override trumps provider override
        assert g.try_acquire("groq", "llama-3.3-70b-versatile") is True
        assert g.try_acquire("groq", "llama-3.3-70b-versatile") is False  # 1/1
        g.release("groq", "llama-3.3-70b-versatile")
        # a different model on the same provider uses provider policy (5)
        assert g.try_acquire("groq", "qwen3-32b") is True
        assert g.try_acquire("groq", "qwen3-32b") is True
