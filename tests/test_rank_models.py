#!/usr/bin/env python3
"""Tests for rank_models.py — U = A * (I^w / (C_opp * T)) scoring module.

Loads rank_models.py via importlib (matches test_reorder_chains.py pattern
so we don't depend on package install).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pytest
import unittest.mock as mock

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "rank_models.py"
spec = importlib.util.spec_from_file_location("rank_models", _SCRIPT_PATH)
rank_models = importlib.util.module_from_spec(spec)
sys.modules["rank_models"] = rank_models
spec.loader.exec_module(rank_models)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bench():
    return {
        "gemini-3-flash": rank_models.BenchmarkScore("gemini-3-flash", coding=0.747, chat=0.591),
        "gpt-5-2": rank_models.BenchmarkScore("gpt-5-2", coding=0.708, chat=0.583),
        "llama-small": rank_models.BenchmarkScore("llama-small", coding=0.30, chat=0.25),
        "untested-model": rank_models.BenchmarkScore("untested-model", coding=0.50, chat=0.50),
    }


def _make_cats():
    return {
        "nvidia": rank_models.ProviderCategory("nvidia", "free_unlimited", 40, 0),
        "groq": rank_models.ProviderCategory("groq", "free_unlimited", 30, 14400),
        "agentrouter": rank_models.ProviderCategory("agentrouter", "free_one_time", 0, 0),
        "openai": rank_models.ProviderCategory("openai", "paid", 0, 0),
        "noobrouter": rank_models.ProviderCategory("noobrouter", "free_daily", 60, 100000),
        "duckduckgo": rank_models.ProviderCategory("duckduckgo", "no_key", 0, 0),
    }


class _FakeStat:
    def __init__(self, tps=50.0, ttft=400.0):
        self.avg_tps = tps
        self.avg_ttft_ms = ttft


def _make_chain():
    return [
        {"provider": "groq", "model": "llama-small", "priority": 1},
        {"provider": "nvidia", "model": "gemini-3-flash", "priority": 2},
        {"provider": "agentrouter", "model": "gpt-5-2", "priority": 3},
        {"provider": "openai", "model": "gpt-5-2", "priority": 4},
    ]


# ---------------------------------------------------------------------------
# TestComputeU
# ---------------------------------------------------------------------------


class TestComputeU:
    def test_kill_switch_below_floor(self):
        u, reason = rank_models.compute_U(0.20, w=3, opportunity_cost=1.0, latency_s=5.0, I_floor=0.55)
        assert u == 0.0
        assert "A=0" in reason

    def test_above_floor_scales_with_w(self):
        u1, _ = rank_models.compute_U(0.7, w=1, opportunity_cost=1.0, latency_s=1.0, I_floor=0.5)
        u2, _ = rank_models.compute_U(0.7, w=2, opportunity_cost=1.0, latency_s=1.0, I_floor=0.5)
        u3, _ = rank_models.compute_U(0.7, w=3, opportunity_cost=1.0, latency_s=1.0, I_floor=0.5)
        assert u1 > 0
        assert u3 < u2 < u1  # higher w → lower U at I<1 (exponentiates intelligence)

    def test_paid_provider_caps_c_opp(self):
        u_paid, _ = rank_models.compute_U(0.7, w=2, opportunity_cost=100.0, latency_s=1.0, I_floor=0.5)
        u_free, _ = rank_models.compute_U(0.7, w=2, opportunity_cost=1.0, latency_s=1.0, I_floor=0.5)
        assert u_paid < u_free  # paid C_opp reduces U

    def test_slow_latency_reduces_u(self):
        u_fast, _ = rank_models.compute_U(0.7, w=2, opportunity_cost=1.0, latency_s=1.0, I_floor=0.5)
        u_slow, _ = rank_models.compute_U(0.7, w=2, opportunity_cost=1.0, latency_s=100.0, I_floor=0.5)
        assert u_fast > u_slow

    def test_zero_latency_clamped(self):
        # ponytail: T >= 0.1 to avoid div-by-zero
        u, _ = rank_models.compute_U(0.7, w=2, opportunity_cost=1.0, latency_s=0.0, I_floor=0.5)
        assert u > 0  # doesn't crash, clamped


# ---------------------------------------------------------------------------
# TestComputeIntelligence
# ---------------------------------------------------------------------------


class TestComputeIntelligence:
    def test_known_model_coding(self):
        bench = _make_bench()
        I = rank_models.compute_intelligence("gemini-3-flash", "coding", bench)
        assert I == pytest.approx(0.747, abs=0.001)

    def test_known_model_chat(self):
        bench = _make_bench()
        I = rank_models.compute_intelligence("gemini-3-flash", "chat", bench)
        assert I == pytest.approx(0.591, abs=0.001)

    def test_provider_prefix_stripped(self):
        bench = _make_bench()
        I = rank_models.compute_intelligence("groq/gemini-3-flash", "coding", bench)
        assert I == pytest.approx(0.747, abs=0.001)

    def test_alias_normalization_llama(self):
        bench = {
            "llama-3-3-70b-instruct": rank_models.BenchmarkScore(
                "llama-3-3-70b-instruct", coding=0.50, chat=0.40
            )
        }
        # Variants all resolve to the canonical alias.
        for variant in ["llama-3.3-70b", "llama-3.3-70b-versatile", "groq/llama-3.3-70b-versatile", "meta-llama/llama-3.3-70b-instruct"]:
            I = rank_models.compute_intelligence(variant, "coding", bench)
            assert I == pytest.approx(0.50, abs=0.001), f"failed for {variant}"

    def test_unknown_model_returns_floor(self):
        I = rank_models.compute_intelligence("brand-new-model", "coding", {})
        assert I == rank_models.NULL_I_FLOOR


# ---------------------------------------------------------------------------
# TestOpportunityCost
# ---------------------------------------------------------------------------


class TestOpportunityCost:
    def test_free_unlimited_is_one(self):
        cats = _make_cats()
        c = rank_models.compute_opportunity_cost("nvidia", cats)
        assert c == 1.0

    def test_no_key_is_one(self):
        cats = _make_cats()
        c = rank_models.compute_opportunity_cost("duckduckgo", cats)
        assert c == 1.0

    def test_paid_caps_high(self):
        cats = _make_cats()
        c = rank_models.compute_opportunity_cost("openai", cats)
        assert c == rank_models.C_OPP_CAP

    def test_free_one_time_is_scarce(self):
        cats = _make_cats()
        c = rank_models.compute_opportunity_cost("agentrouter", cats)
        assert c > 1.0  # scarce
        assert c < rank_models.C_OPP_CAP  # not as bad as paid

    def test_free_daily_rises_with_usage(self):
        cats = _make_cats()
        c0 = rank_models.compute_opportunity_cost("noobrouter", cats, telemetry_used_today=0)
        c_half = rank_models.compute_opportunity_cost("noobrouter", cats, telemetry_used_today=50000)
        c_full = rank_models.compute_opportunity_cost("noobrouter", cats, telemetry_used_today=100000)
        assert c0 == 1.0
        assert c_half > c0
        assert c_full > c_half

    def test_unknown_provider_defaults_paid(self):
        c = rank_models.compute_opportunity_cost("mystery-provider", {})
        assert c == rank_models.C_OPP_CAP


# ---------------------------------------------------------------------------
# TestComputeLatency
# ---------------------------------------------------------------------------


class TestComputeLatency:
    def test_basic_formula(self):
        T = rank_models.compute_latency(ttft_ms=1000.0, tps=50.0, n_out_tokens=2000)
        # 1.0s + 2000/50 = 1.0 + 40.0 = 41.0
        assert T == pytest.approx(41.0, abs=0.01)

    def test_missing_telemetry_uses_fallbacks(self):
        T = rank_models.compute_latency(ttft_ms=None, tps=None, n_out_tokens=2000)
        # DEFAULT_TTFT_MS_FALLBACK/1000 + 2000/DEFAULT_TPS_FALLBACK
        expected = rank_models.DEFAULT_TTFT_MS_FALLBACK / 1000.0 + 2000 / rank_models.DEFAULT_TPS_FALLBACK
        assert T == pytest.approx(expected, abs=0.01)

    def test_zero_tps_uses_fallback(self):
        T = rank_models.compute_latency(ttft_ms=500.0, tps=0.0, n_out_tokens=1000)
        assert T > 0  # doesn't div-by-zero


# ---------------------------------------------------------------------------
# TestRankChain
# ---------------------------------------------------------------------------


class TestRankChain:
    def test_high_intelligence_free_fast_wins(self):
        bench = _make_bench()
        cats = _make_cats()
        tier = rank_models.TierConfig(w=3, I_floor=0.55, n_out_tokens=4000, purpose="coding")
        telemetry = {
            ("nvidia", "gemini-3-flash"): _FakeStat(tps=50.0, ttft=400.0),
            ("groq", "llama-small"): _FakeStat(tps=250.0, ttft=200.0),
        }
        chain = _make_chain()
        new_chain, reasons = rank_models.rank_chain(chain, telemetry, {}, bench, cats, tier)
        assert new_chain[0]["provider"] == "nvidia"  # I=0.747, free, fast
        assert new_chain[-1]["provider"] == "openai"  # paid, C_opp cap

    def test_below_floor_killed(self):
        bench = _make_bench()
        cats = _make_cats()
        tier = rank_models.TierConfig(w=3, I_floor=0.55, n_out_tokens=4000, purpose="coding")
        telemetry = {("groq", "llama-small"): _FakeStat(tps=250.0, ttft=200.0)}
        chain = _make_chain()
        new_chain, reasons = rank_models.rank_chain(chain, telemetry, {}, bench, cats, tier)
        # llama-small I=0.30 < floor 0.55 → U=0
        llama_entry = next(c for c in new_chain if c["model"] == "llama-small")
        llama_reason = next(r for r in reasons if r.model == "llama-small")
        assert llama_reason.score == 0.0
        assert "A=0" in llama_reason.reason

    def test_penalty_multiplier_sinks_failing(self):
        bench = _make_bench()
        cats = _make_cats()
        tier = rank_models.TierConfig(w=2, I_floor=0.40, n_out_tokens=2000, purpose="coding")
        telemetry = {
            ("nvidia", "gemini-3-flash"): _FakeStat(tps=50.0, ttft=400.0),
            ("nvidia", "untested-model"): _FakeStat(tps=50.0, ttft=400.0),
        }
        chain = [
            {"provider": "nvidia", "model": "untested-model", "priority": 1},
            {"provider": "nvidia", "model": "gemini-3-flash", "priority": 2},
        ]
        # Penalize untested-model heavily.
        penalties = {("nvidia", "untested-model"): 50.0}
        new_chain, reasons = rank_models.rank_chain(chain, telemetry, penalties, bench, cats, tier)
        # gemini-3-flash should win (no penalty, higher I)
        assert new_chain[0]["model"] == "gemini-3-flash"

    def test_no_telemetry_preserves_original_order_at_tail(self):
        bench = _make_bench()
        cats = _make_cats()
        tier = rank_models.TierConfig(w=1, I_floor=0.10, n_out_tokens=500, purpose="coding")
        telemetry = {("nvidia", "gemini-3-flash"): _FakeStat(tps=50.0, ttft=400.0)}
        chain = [
            {"provider": "groq", "model": "llama-small", "priority": 1},
            {"provider": "agentrouter", "model": "gpt-5-2", "priority": 2},
            {"provider": "nvidia", "model": "gemini-3-flash", "priority": 3},
        ]
        new_chain, reasons = rank_models.rank_chain(chain, telemetry, {}, bench, cats, tier)
        # Scored (nvidia) first, then unscored in original order.
        assert new_chain[0]["provider"] == "nvidia"
        assert new_chain[1]["provider"] == "groq"  # original idx 0
        assert new_chain[2]["provider"] == "agentrouter"  # original idx 1

    def test_empty_chain(self):
        bench = _make_bench()
        cats = _make_cats()
        tier = rank_models.TierConfig()
        new_chain, reasons = rank_models.rank_chain([], {}, {}, bench, cats, tier)
        assert new_chain == []
        assert reasons == []


# ---------------------------------------------------------------------------
# TestTierConfigLoad
# ---------------------------------------------------------------------------


class TestTierConfigLoad:
    def test_loads_real_config(self):
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "tier_config.yaml"
        if not cfg_path.exists():
            pytest.skip("tier_config.yaml not committed yet")
        cfg = rank_models.load_tier_config(cfg_path)
        assert "default" in cfg
        assert "coding-elite" in cfg
        elite = cfg["coding-elite"]
        assert elite.w == 3
        assert elite.I_floor == 0.55
        assert elite.purpose == "coding"

    def test_missing_file_returns_default(self, tmp_path):
        cfg = rank_models.load_tier_config(tmp_path / "nope.yaml")
        assert "default" in cfg
        assert cfg["default"].w == 2

    def test_get_tier_falls_back_to_default(self):
        cfg = {"default": rank_models.TierConfig(w=2, I_floor=0.30)}
        tier = rank_models.get_tier(cfg, "nonexistent-model")
        assert tier.w == 2


# ---------------------------------------------------------------------------
# TestLoadersGraceful
# ---------------------------------------------------------------------------


class TestLoadersGraceful:
    def test_load_benchmark_missing_db_returns_empty(self, tmp_path):
        result = rank_models.load_benchmark_scores(tmp_path / "nope.db")
        assert result == {}

    def test_load_provider_categories_missing_db_returns_empty(self, tmp_path):
        result = rank_models.load_provider_categories(tmp_path / "nope.db")
        assert result == {}

    def test_load_benchmark_real_db_if_present(self):
        db = rank_models.DEFAULT_BENCHMARK_DB
        if not db.exists():
            pytest.skip("benchmark DB not on this machine")
        result = rank_models.load_benchmark_scores(db)
        assert len(result) > 0
        # Spot-check: gemini-3-flash should be present with coding score ~0.747
        score = result.get("gemini-3-flash")
        assert score is not None
        assert score.coding == pytest.approx(0.747, abs=0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
