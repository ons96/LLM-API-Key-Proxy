"""Quality scores in live chain ordering (#1122).

The scorer that actually runs is DynamicScoringEngine.rank_models_for_virtual;
these tests pin the four acceptance criteria:

1. a model with a known benchmark score outranks one without, at equal tier;
2. the quality/speed blend differs per virtual model (fast is speed-weighted);
3. unscored models get a neutral prior instead of sinking to the bottom;
4. no dormant second scorer is left behind (ModelRanker removed).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from rotator_library.quality_scores import load_quality_scores, normalize
from rotator_library.scoring_engine import DynamicScoringEngine

SRC = Path(__file__).resolve().parent.parent / "src"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def write_rankings(tmp_path: Path, models: list[dict]) -> str:
    path = tmp_path / "model_rankings.yaml"
    path.write_text(yaml.safe_dump({"models": models}))
    return str(path)


def write_quality(tmp_path: Path, payload) -> str:
    path = tmp_path / "quality_scores.json"
    path.write_text(json.dumps(payload))
    return str(path)


def engine_for(tmp_path: Path, models: list[dict], quality=None) -> DynamicScoringEngine:
    quality_path = (
        write_quality(tmp_path, quality) if quality is not None else str(tmp_path / "absent.json")
    )
    engine = DynamicScoringEngine(
        model_rankings_path=write_rankings(tmp_path, models),
        quality_scores_path=quality_path,
    )
    engine.load_model_rankings()
    return engine


def with_tps(engine: DynamicScoringEngine, speeds: dict) -> DynamicScoringEngine:
    """Telemetry is absent in unit tests; inject per-model speed directly."""
    engine.get_tps_estimate = lambda provider, model: speeds.get(f"{provider}/{model}", 100.0)
    return engine


# ---------------------------------------------------------------------------
# AC1 -- a known score outranks an unknown one
# ---------------------------------------------------------------------------


def test_known_benchmark_outranks_unscored_at_equal_tier(tmp_path):
    engine = engine_for(
        tmp_path,
        [{"id": "alpha/benchmarked", "scores": {"agentic_coding": 82.0}}],
    )

    ranked = engine.rank_models_for_virtual(
        "coding-smart", [("beta", "mystery"), ("alpha", "benchmarked")]
    )

    assert [s.model for s in ranked] == ["benchmarked", "mystery"]
    assert ranked[0].quality_known is True
    assert ranked[1].quality_known is False


# ---------------------------------------------------------------------------
# AC2 -- the blend differs per virtual model, and stays quality-majority
# ---------------------------------------------------------------------------


def test_fast_tier_is_speed_weighted_smart_tier_is_quality_weighted():
    fast = DynamicScoringEngine.CATEGORY_WEIGHTS["coding-fast"]
    smart = DynamicScoringEngine.CATEGORY_WEIGHTS["coding-smart"]

    assert fast["tps"] > smart["tps"]
    assert fast["agentic"] < smart["agentic"]
    # quality keeps the majority even in the latency tier
    assert fast["agentic"] > fast["tps"]
    assert fast["agentic"] + fast["tps"] + fast["availability"] == pytest.approx(1.0)


def test_fast_model_wins_fast_tier_but_loses_smart_tier(tmp_path):
    engine = with_tps(
        engine_for(
            tmp_path,
            [
                {"id": "slow/strong", "scores": {"agentic_coding": 88.0}},
                {"id": "quick/weak", "scores": {"agentic_coding": 58.0}},
            ],
        ),
        {"slow/strong": 20.0, "quick/weak": 900.0},
    )

    fast = engine.rank_models_for_virtual(
        "coding-fast", [("slow", "strong"), ("quick", "weak")]
    )
    smart = engine.rank_models_for_virtual(
        "coding-smart", [("slow", "strong"), ("quick", "weak")]
    )

    assert [s.model for s in fast] == ["weak", "strong"], "fast tier should follow speed"
    assert [s.model for s in smart] == ["strong", "weak"], "smart tier should follow quality"


def test_fast_but_weak_never_leads_a_coding_chain(tmp_path):
    """Note in #1122: speed must not let a weak model head a coding chain at equal tps."""
    engine = with_tps(
        engine_for(
            tmp_path,
            [
                {"id": "good/model", "scores": {"agentic_coding": 85.0}},
                {"id": "bad/model", "scores": {"agentic_coding": 12.0}},
            ],
        ),
        {"good/model": 100.0, "bad/model": 100.0},
    )

    for virtual_model in ("coding-smart", "coding-elite", "coding-fast"):
        ranked = engine.rank_models_for_virtual(
            virtual_model, [("bad", "model"), ("good", "model")]
        )
        assert ranked[0].model == "model" and ranked[0].provider == "good", virtual_model


# ---------------------------------------------------------------------------
# AC3 -- neutral prior, not the bottom
# ---------------------------------------------------------------------------


def test_unscored_model_uses_median_prior_not_zero(tmp_path):
    engine = engine_for(
        tmp_path,
        [
            {"id": "a/low", "scores": {"agentic_coding": 40.0}},
            {"id": "a/mid", "scores": {"agentic_coding": 60.0}},
            {"id": "a/high", "scores": {"agentic_coding": 80.0}},
        ],
    )

    assert engine.neutral_prior == pytest.approx(60.0)
    score = engine.calculate_coding_score("b", "unknown", "coding-smart")
    assert score.agentic_score == pytest.approx(60.0)
    assert score.quality_known is False


def test_unscored_model_ranks_above_a_weak_known_model(tmp_path):
    engine = engine_for(
        tmp_path,
        [
            {"id": "a/low", "scores": {"agentic_coding": 40.0}},
            {"id": "a/mid", "scores": {"agentic_coding": 60.0}},
            {"id": "a/high", "scores": {"agentic_coding": 80.0}},
            {"id": "weak/known", "scores": {"agentic_coding": 12.0}},
        ],
    )

    ranked = engine.rank_models_for_virtual(
        "coding-fast", [("weak", "known"), ("ghost", "model")]
    )

    assert ranked[0].provider == "ghost", "unknown must not be treated as worst"
    assert ranked[-1].provider == "weak"


def test_neutral_prior_is_clamped_to_a_sane_band(tmp_path):
    engine = engine_for(
        tmp_path,
        [{"id": f"a/m{i}", "scores": {"agentic_coding": 95.0}} for i in range(3)],
    )
    assert engine.neutral_prior == pytest.approx(DynamicScoringEngine.NEUTRAL_PRIOR_MAX)

    low_dir = tmp_path / "low"
    low_dir.mkdir()
    low = engine_for(
        low_dir,
        [{"id": f"a/m{i}", "scores": {"agentic_coding": 5.0}} for i in range(3)],
    )
    assert low.neutral_prior == pytest.approx(DynamicScoringEngine.NEUTRAL_PRIOR_MIN)


def test_threshold_still_demotes_below_floor_models(tmp_path):
    """AC3 must not defeat the tier floor: weak known models stay demoted."""
    engine = engine_for(
        tmp_path,
        [
            {"id": "a/strong", "scores": {"agentic_coding": 85.0}},
            {"id": "a/weak", "scores": {"agentic_coding": 30.0}},
        ],
    )

    ranked = engine.rank_models_for_virtual(
        "coding-smart", [("a", "weak"), ("a", "strong")]
    )

    assert ranked[0].model == "strong"
    assert ranked[-1].meets_threshold is False


# ---------------------------------------------------------------------------
# catalog export fills the gap the rankings file leaves
# ---------------------------------------------------------------------------


def test_catalog_export_scores_a_model_with_no_ranking_row(tmp_path):
    engine = engine_for(
        tmp_path,
        [{"id": "a/ranked", "scores": {"agentic_coding": 50.0}}],
        quality={"models": {"freetheai/bbl/gemini-3.5-flash": {"quality_score": 72.4}}},
    )

    assert engine.get_model_score("freetheai", "bbl/gemini-3.5-flash") == pytest.approx(72.4)
    assert engine.get_model_score("nobody", "nowhere") is None


def test_missing_quality_export_is_not_fatal(tmp_path):
    engine = engine_for(tmp_path, [{"id": "a/ranked", "scores": {"agentic_coding": 70.0}}])
    assert engine._catalog_quality == {}
    ranked = engine.rank_models_for_virtual("coding-smart", [("a", "ranked")])
    assert ranked[0].total_score > 0


@pytest.mark.parametrize(
    "payload",
    [
        {"models": {"p/m": {"quality_score": 61.0}}},
        {"models": {"p/m": 61.0}},
        {"p/m": {"quality_score": 61.0}},
        {"p/m": 61.0},
        [{"id": "p/m", "quality_score": 61.0}],
    ],
)
def test_quality_export_shapes_normalize(payload):
    assert normalize(payload) == {"p/m": pytest.approx(0.61)}


def test_quality_export_drops_unusable_values():
    assert normalize(
        {
            "models": {
                "P/M": 88.0,
                "p/bool": True,
                "p/str": "70",
                "p/negative": -5.0,
                "p/over": 101.0,
                "p/none": None,
                "no-slash": 50.0,
            }
        }
    ) == {"p/m": pytest.approx(0.88)}


def test_quality_export_reloads_when_reexported(tmp_path):
    path = write_quality(tmp_path, {"models": {"p/m": 40.0}})
    assert load_quality_scores(path) == {"p/m": pytest.approx(0.40)}

    Path(path).write_text(json.dumps({"models": {"p/m": 90.0}}))
    # mtime granularity guard: make the rewrite visible to the cache key
    stat = Path(path).stat()
    assert load_quality_scores(path) == {"p/m": pytest.approx(0.90)}
    assert stat.st_size != 0


def test_corrupt_quality_export_returns_empty(tmp_path):
    path = tmp_path / "quality_scores.json"
    path.write_text("{not json")
    assert load_quality_scores(str(path)) == {}


# ---------------------------------------------------------------------------
# AC4 -- no dormant second scorer
# ---------------------------------------------------------------------------


def test_dormant_model_ranker_is_gone():
    assert importlib.util.find_spec("proxy_app.model_ranker") is None
    source = (SRC / "proxy_app" / "router_core.py").read_text()
    assert "model_ranker" not in source
    assert "ModelRanker" not in source


def test_router_imports_without_the_removed_module():
    sys.modules.pop("proxy_app.router_core", None)
    import proxy_app.router_core  # noqa: F401  -- import must not explode

    assert "ModelRanker" not in dir(proxy_app.router_core)
