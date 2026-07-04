#!/usr/bin/env python3
"""Tests for DynamicScoringEngine #341 changes: 80/15/5 weights,
SWE-bench threshold enforcement, and free-model baseline exclusion.

Loads scoring_engine.py via importlib to avoid package-install deps,
matching the test_rank_models.py pattern in this repo.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest

_SRC_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "rotator_library"
    / "scoring_engine.py"
)
spec = importlib.util.spec_from_file_location("scoring_engine", _SRC_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules["scoring_engine"] = mod
spec.loader.exec_module(mod)

DynamicScoringEngine = mod.DynamicScoringEngine
ModelScore = mod.ModelScore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_rankings(tmp_path: Path, models: list[dict]) -> Path:
    """Write a minimal model_rankings.yaml for testing."""
    import yaml

    path = tmp_path / "model_rankings.yaml"
    data = {"models": models}
    path.write_text(yaml.dump(data))
    return path


@pytest.fixture
def engine_no_telemetry(tmp_path):
    """Engine with no telemetry (uses fallback TPS=100, availability=1.0)."""
    models = [
        {"id": "nvidia/claude-opus-4-5", "scores": {"agentic_coding": 74.4}},
        {"id": "nvidia/gemini-3-pro", "scores": {"agentic_coding": 74.2}},
        {"id": "groq/llama-3.3-70b", "scores": {"agentic_coding": 65.2}},
        {"id": "groq/weak-model", "scores": {"agentic_coding": 20.0}},
    ]
    path = _write_rankings(tmp_path, models)
    eng = DynamicScoringEngine(telemetry_manager=None, model_rankings_path=str(path))
    eng.load_model_rankings()
    return eng


# ---------------------------------------------------------------------------
# Weight change: 80/15/5
# ---------------------------------------------------------------------------


class TestWeights80155:
    def test_coding_elite_weights_sum_and_split(self):
        w = DynamicScoringEngine.CATEGORY_WEIGHTS["coding-elite"]
        assert w["agentic"] == 0.80
        assert w["tps"] == 0.15
        assert w["availability"] == 0.05

    def test_coding_smart_weights(self):
        w = DynamicScoringEngine.CATEGORY_WEIGHTS["coding-smart"]
        assert w["agentic"] == 0.80
        assert w["tps"] == 0.15
        assert w["availability"] == 0.05

    def test_chat_elite_has_availability(self):
        w = DynamicScoringEngine.CATEGORY_WEIGHTS["chat-elite"]
        assert w["availability"] == 0.05
        assert w["intelligence"] == 0.80

    def test_no_legacy_75_15_10(self):
        """Ensure old 75/15/10 weights are gone."""
        w = DynamicScoringEngine.CATEGORY_WEIGHTS["coding-elite"]
        assert w["agentic"] != 0.75

    def test_availability_contributes_to_score(self, engine_no_telemetry):
        """Availability (default 1.0 with no telemetry) adds 5% of 100 = 5."""
        score = engine_no_telemetry.calculate_coding_score(
            "nvidia", "claude-opus-4-5", "coding-elite"
        )
        # agentic 74.4 * 0.80 = 59.52
        # tps fallback 100 -> 100/20=5.0 *0.15 = 0.75
        # availability 1.0*100=100 * 0.05 = 5.0
        # hallucination default 10.0 -> 10/20=0.5 *0.10 = -0.05
        assert score.total_score == pytest.approx(59.52 + 0.75 + 5.0 - 0.05, abs=0.5)


# ---------------------------------------------------------------------------
# Threshold enforcement (coding-elite 70, coding-smart 65)
# ---------------------------------------------------------------------------


class TestThresholdEnforcement:
    def test_elite_threshold_is_70(self):
        assert DynamicScoringEngine.THRESHOLDS["coding-elite"] == 70.0

    def test_smart_threshold_is_65(self):
        assert DynamicScoringEngine.THRESHOLDS["coding-smart"] == 65.0

    def test_below_elite_threshold_meets_false(self, engine_no_telemetry):
        """llama-3.3-70b (65.2) < 70 elite threshold -> meets_threshold False."""
        score = engine_no_telemetry.calculate_coding_score(
            "groq", "llama-3.3-70b", "coding-elite"
        )
        assert score.agentic_score == pytest.approx(65.2)
        assert score.meets_threshold is False

    def test_above_elite_threshold_meets_true(self, engine_no_telemetry):
        """claude-opus-4-5 (74.4) >= 70 -> meets_threshold True."""
        score = engine_no_telemetry.calculate_coding_score(
            "nvidia", "claude-opus-4-5", "coding-elite"
        )
        assert score.meets_threshold is True

    def test_smart_threshold_65_passes_llama(self, engine_no_telemetry):
        """llama-3.3-70b (65.2) >= 65 smart threshold -> meets True."""
        score = engine_no_telemetry.calculate_coding_score(
            "groq", "llama-3.3-70b", "coding-smart"
        )
        assert score.meets_threshold is True

    def test_below_threshold_demoted_to_tail_in_rank(self, engine_no_telemetry):
        """When ranking, below-threshold models sort after above-threshold."""
        candidates = [
            ("groq", "llama-3.3-70b"),  # 65.2 < 70 elite
            ("nvidia", "claude-opus-4-5"),  # 74.4 >= 70
        ]
        ranked = engine_no_telemetry.rank_models_for_virtual("coding-elite", candidates)
        # opus first (meets threshold), llama second (demoted)
        assert ranked[0].model == "claude-opus-4-5"
        assert ranked[1].model == "llama-3.3-70b"
        assert ranked[0].meets_threshold is True
        assert ranked[1].meets_threshold is False

    def test_chat_unaffected_by_swe_threshold(self, engine_no_telemetry):
        """chat-elite has threshold 0.0, so weak model still meets_threshold."""
        score = engine_no_telemetry.calculate_coding_score(
            "groq", "weak-model", "chat-elite"
        )
        assert score.meets_threshold is True  # threshold 0.0


# ---------------------------------------------------------------------------
# Free-model baseline exclusion
# ---------------------------------------------------------------------------


class TestFreeBaselineExclusion:
    def test_baseline_computed_at_load(self, tmp_path):
        """Free baseline = worst free-model agentic score, >= hard floor 30."""
        models = [
            {"id": "nvidia/strong-free", "scores": {"agentic_coding": 74.0}},
            {"id": "groq/weak-free", "scores": {"agentic_coding": 45.0}},
            {"id": "openai/paid-model", "scores": {"agentic_coding": 90.0}},
        ]
        path = _write_rankings(tmp_path, models)
        eng = DynamicScoringEngine(
            telemetry_manager=None, model_rankings_path=str(path)
        )
        eng.load_model_rankings()
        # free scores: nvidia 74, groq 45 -> min=45 -> baseline=max(45,30)=45
        assert eng._free_baseline == 45.0

    def test_below_baseline_score_zeroed(self, tmp_path):
        """Model below free baseline gets total_score=0 + meets_threshold False."""
        models = [
            {"id": "nvidia/good-free", "scores": {"agentic_coding": 60.0}},
            {"id": "openai/terrible-paid", "scores": {"agentic_coding": 10.0}},
        ]
        path = _write_rankings(tmp_path, models)
        eng = DynamicScoringEngine(
            telemetry_manager=None, model_rankings_path=str(path)
        )
        eng.load_model_rankings()
        # baseline = max(min([60]), 30) = 60. terrible-paid 10 < 60 -> zeroed.
        score = eng.calculate_coding_score(
            "openai", "terrible-paid", "coding-elite"
        )
        assert score.total_score == 0.0
        assert score.meets_threshold is False

    def test_baseline_floor_enforced(self, tmp_path):
        """If all free models score low, baseline floors at 30."""
        models = [
            {"id": "groq/weak-1", "scores": {"agentic_coding": 5.0}},
            {"id": "groq/weak-2", "scores": {"agentic_coding": 10.0}},
        ]
        path = _write_rankings(tmp_path, models)
        eng = DynamicScoringEngine(
            telemetry_manager=None, model_rankings_path=str(path)
        )
        eng.load_model_rankings()
        assert eng._free_baseline == 30.0  # floor

    def test_baseline_recomputed_on_reload(self, tmp_path):
        """Reloading rankings after file change recomputes baseline (#341 AC5)."""
        import yaml

        path = tmp_path / "model_rankings.yaml"
        path.write_text(
            yaml.dump(
                {"models": [{"id": "nvidia/m", "scores": {"agentic_coding": 40.0}}]}
            )
        )
        eng = DynamicScoringEngine(
            telemetry_manager=None, model_rankings_path=str(path)
        )
        eng.load_model_rankings()
        assert eng._free_baseline == 40.0

        # Overwrite with higher scores
        path.write_text(
            yaml.dump(
                {"models": [{"id": "nvidia/m", "scores": {"agentic_coding": 80.0}}]}
            )
        )
        eng.load_model_rankings()
        assert eng._free_baseline == 80.0


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
