"""Tests for chain_selector.py (#480)."""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proxy_app.routing.chain_selector import (  # noqa: E402
    ChainCandidate,
    ChainSelector,
    SelectResult,
    load_capabilities,
    load_virtual_chain,
    on_error,
    to_debug_header,
)
from proxy_app.routing.latency_predictor import LatencyPredictor, Prediction  # noqa: E402
from proxy_app.routing.request_features import (  # noqa: E402
    Capabilities,
    RequestFeatures,
    TaskClass,
)
from proxy_app.routing.tier_classifier import Tier  # noqa: E402


def _features(**kw):
    defaults = dict(
        capabilities=Capabilities.TEXT,
        task_class=TaskClass.SHORT_QA,
        estimated_input_tokens=100,
        image_count=0,
        file_count=0,
        tool_count=0,
        prompt_length_bucket="SHORT",
        text_char_count=80,
    )
    defaults.update(kw)
    return RequestFeatures(**defaults)


class _FakePredictor:
    """LatencyPredictor stand-in returning canned predictions."""

    def __init__(self, preds):
        self._preds = preds

    def predict_many(self, candidates, output_tokens=200):
        out = []
        for provider, model in candidates:
            p = self._preds.get((provider, model))
            if p:
                out.append(
                    Prediction(
                        provider=provider,
                        model=model,
                        e_time_ms=p[0],
                        ttft_ms=100,
                        tps=10.0,
                        p_fail=0.05,
                        confidence=p[1],
                        source="telemetry" if p[1] >= 0.7 else "default",
                    )
                )
        return out


CANDIDATES = [
    {"provider": "groq", "model": "llama-3.1-8b-instant", "priority": 1},
    {"provider": "cerebras", "model": "gpt-oss-120b", "priority": 2},
    {"provider": "openai", "model": "gpt-4o-mini", "priority": 3},
    {"provider": "anthropic", "model": "claude-haiku-4-5", "priority": 4},
]


def test_vision_request_filters_non_vision_first_candidate():
    caps = {
        "groq/llama-3.1-8b-instant": Capabilities.TEXT | Capabilities.TOOL_CALLING,
        "cerebras/gpt-oss-120b": Capabilities.TEXT,
        "openai/gpt-4o-mini": Capabilities.TEXT | Capabilities.VISION,
        "anthropic/claude-haiku-4-5": Capabilities.TEXT | Capabilities.VISION,
    }
    sel = ChainSelector(
        model_scores={k: 0.5 for k in caps},
        capabilities=caps,
        predictor=None,
    )
    res = sel.select(
        _features(capabilities=Capabilities.TEXT | Capabilities.VISION),
        Tier.T2,
        CANDIDATES,
    )
    assert res.chain
    for c in res.chain:
        assert c.capabilities & Capabilities.VISION
    assert res.chain[0].model == "gpt-4o-mini"  # lowest priority vision model


def test_tools_request_prefers_tool_capable():
    caps = {
        "groq/llama-3.1-8b-instant": Capabilities.TEXT | Capabilities.TOOL_CALLING,
        "cerebras/gpt-oss-120b": Capabilities.TEXT,
    }
    sel = ChainSelector(model_scores={}, capabilities=caps, predictor=None)
    res = sel.select(
        _features(capabilities=Capabilities.TEXT | Capabilities.TOOL_CALLING),
        Tier.T1,
        CANDIDATES[:2],
    )
    assert [c.model for c in res.chain] == ["llama-3.1-8b-instant"]


def test_attachment_requires_file_parsing():
    caps = {
        "groq/llama-3.1-8b-instant": Capabilities.TEXT | Capabilities.TOOL_CALLING,
        "openai/gpt-4o-mini": Capabilities.TEXT | Capabilities.VISION | Capabilities.FILE_PARSING,
    }
    sel = ChainSelector(
        model_scores={k: 0.5 for k in caps},
        capabilities=caps,
        predictor=None,
    )
    res = sel.select(
        _features(capabilities=Capabilities.TEXT | Capabilities.FILE_PARSING, file_count=1),
        Tier.T2,
        CANDIDATES,
    )
    assert [c.model for c in res.chain] == ["gpt-4o-mini"]


def test_tier_floor_filters_weak_models():
    scores = {
        "groq/llama-3.1-8b-instant": 0.10,  # below T2 floor 0.40
        "cerebras/gpt-oss-120b": 0.55,  # meets T2
    }
    sel = ChainSelector(model_scores=scores, capabilities={}, predictor=None)
    res = sel.select(_features(), Tier.T2, CANDIDATES[:2])
    assert [c.model for c in res.chain] == ["gpt-oss-120b"]


def test_t0_keeps_everything():
    scores = {"groq/llama-3.1-8b-instant": 0.0}
    sel = ChainSelector(model_scores=scores, capabilities={}, predictor=None)
    res = sel.select(_features(), Tier.T0, CANDIDATES[:1])
    assert len(res.chain) == 1


def test_e_time_sort_with_confident_predictions():
    sel = ChainSelector(
        model_scores={},
        capabilities={},
        predictor=_FakePredictor(
            {
                ("groq", "llama-3.1-8b-instant"): (5000, 1.0),
                ("cerebras", "gpt-oss-120b"): (1000, 1.0),
            }
        ),
    )
    res = sel.select(_features(), Tier.T1, CANDIDATES[:2])
    assert not res.static_fallback
    assert [c.model for c in res.chain] == ["gpt-oss-120b", "llama-3.1-8b-instant"]


def test_static_fallback_without_telemetry():
    sel = ChainSelector(model_scores={}, capabilities={}, predictor=None)
    res = sel.select(_features(), Tier.T1, CANDIDATES)
    assert res.static_fallback
    # input order preserved
    assert [c.model for c in res.chain] == [c["model"] for c in CANDIDATES]


def test_low_confidence_predictions_fall_back_static():
    sel = ChainSelector(
        model_scores={},
        capabilities={},
        predictor=_FakePredictor(
            {
                ("groq", "llama-3.1-8b-instant"): (5000, 0.2),
                ("cerebras", "gpt-oss-120b"): (1000, 0.2),
            }
        ),
    )
    res = sel.select(_features(), Tier.T1, CANDIDATES[:2])
    assert res.static_fallback
    assert [c.model for c in res.chain] == [c["model"] for c in CANDIDATES[:2]]


def test_pins_honored_at_head():
    sel = ChainSelector(
        model_scores={},
        capabilities={},
        predictor=None,
        pins=[{"provider": "cerebras", "model": "gpt-oss-120b"}],
    )
    res = sel.select(_features(), Tier.T1, CANDIDATES)
    assert res.chain[0].model == "gpt-oss-120b"
    assert res.chain[0].reason == "PINNED"
    assert res.chain[0].priority == 1


def test_pin_dropped_when_capability_mismatch():
    caps = {
        "cerebras/gpt-oss-120b": Capabilities.TEXT,
        "openai/gpt-4o-mini": Capabilities.TEXT | Capabilities.VISION,
    }
    sel = ChainSelector(
        model_scores={k: 0.5 for k in caps},
        capabilities=caps,
        predictor=None,
        pins=[{"provider": "cerebras", "model": "gpt-oss-120b"}],
    )
    res = sel.select(
        _features(capabilities=Capabilities.TEXT | Capabilities.VISION),
        Tier.T2,
        CANDIDATES,
    )
    assert res.chain[0].model == "gpt-4o-mini"  # pin filtered out


def test_escalation_pool_contains_stronger_models():
    scores = {
        "groq/llama-3.1-8b-instant": 0.10,  # below T2 floor -> dropped from chain
        "cerebras/gpt-oss-120b": 0.30,  # below T2 floor -> dropped
        "openai/gpt-4o-mini": 0.45,  # meets T2 floor -> chain
        "anthropic/claude-haiku-4-5": 0.80,  # meets T2 and T3 floors
    }
    sel = ChainSelector(model_scores=scores, capabilities={}, predictor=None)
    res = sel.select(_features(), Tier.T2, CANDIDATES)
    assert [c.model for c in res.chain] == ["gpt-4o-mini", "claude-haiku-4-5"]
    # T3 is the next tier: the stronger candidate must be available for
    # capability failures or other escalation-triggering errors.
    assert [c.model for c in res.escalation] == ["claude-haiku-4-5"]


def test_on_error_capability_escalates():
    chain = [ChainCandidate(provider="a", model="m1", priority=1)]
    esc = [ChainCandidate(provider="b", model="m2", priority=1, reason="escalation")]
    assert on_error(chain, esc, 422) is esc[0]
    assert on_error(chain, esc, 400, "context length exceeded") is esc[0]
    assert on_error(chain, esc, 404) is esc[0]


def test_on_error_transient_moves_to_next_candidate():
    chain = [
        ChainCandidate(provider="a", model="m1", priority=1),
        ChainCandidate(provider="b", model="m2", priority=2),
    ]
    esc = [ChainCandidate(provider="c", model="m3", priority=1, reason="escalation")]
    assert on_error(chain, esc, 429) is chain[1]
    assert on_error(chain, esc, 503) is chain[1]
    assert on_error(chain, esc, 429, "upstream timed out") is chain[1]


def test_on_error_transient_single_candidate_escalates():
    chain = [ChainCandidate(provider="a", model="m1", priority=1)]
    esc = [ChainCandidate(provider="b", model="m2", priority=1, reason="escalation")]
    assert on_error(chain, esc, 429) is esc[0]


def test_on_error_unknown_returns_none():
    chain = [ChainCandidate(provider="a", model="m1", priority=1)]
    esc = [ChainCandidate(provider="b", model="m2", priority=1, reason="escalation")]
    assert on_error(chain, esc, 200) is None
    assert on_error(chain, esc, None, "") is None


def test_sticky_session_escalate_only():
    scores = {
        "groq/llama-3.1-8b-instant": 0.45,
        "cerebras/gpt-oss-120b": 0.45,
    }
    caps = {
        "groq/llama-3.1-8b-instant": Capabilities.TEXT,
        "cerebras/gpt-oss-120b": Capabilities.TEXT,
    }
    sel = ChainSelector(model_scores=scores, capabilities=caps, predictor=None)
    now = 1000.0
    r1 = sel.select(_features(), Tier.T2, CANDIDATES[:2], session_key="s1", now=now)
    assert r1.chain[0].model == "llama-3.1-8b-instant"
    # Same session, now a T1 request: stored T2 stickiness must hold
    r2 = sel.select(_features(), Tier.T1, CANDIDATES[:2], session_key="s1", now=now + 10)
    assert r2.chain[0].model == "llama-3.1-8b-instant"


def test_sticky_expires():
    sel = ChainSelector(model_scores={}, capabilities={}, predictor=None)
    r1 = sel.select(_features(), Tier.T2, CANDIDATES, session_key="s2", now=1000.0)
    r2 = sel.select(_features(), Tier.T1, CANDIDATES, session_key="s2", now=2000.0)
    assert r2.chain[0].model == CANDIDATES[0]["model"]  # stickiness expired


def test_to_debug_header_format():
    sel = ChainSelector(model_scores={}, capabilities={}, predictor=None)
    res = sel.select(_features(), Tier.T1, CANDIDATES[:2])
    header = to_debug_header(res, "coding-smart")
    assert header.startswith("[coding-smart] #1 groq/llama-3.1-8b-instant: ")
    assert "static-fallback" in header


def test_header_with_pin_reason():
    sel = ChainSelector(
        model_scores={},
        capabilities={},
        predictor=None,
        pins=[{"provider": "groq", "model": "llama-3.1-8b-instant"}],
    )
    res = sel.select(_features(), Tier.T1, CANDIDATES[:2])
    header = to_debug_header(res, "coding-smart")
    assert "PINNED" in header


def test_load_virtual_chain_missing_file():
    assert load_virtual_chain("/nonexistent/vm.yaml", "coding-smart") == []


def test_load_capabilities_missing_files():
    assert load_capabilities(None, None) == {}


def test_capability_overrides_win_over_provider_database(tmp_path):
    providers = tmp_path / "providers.yaml"
    providers.write_text(
        "providers:\n"
        "  - id: demo\n"
        "    capabilities: [vision]\n"
        "    free_models:\n"
        "      - id: model\n"
        "        capabilities: [vision]\n",
        encoding="utf-8",
    )
    overrides = tmp_path / "overrides.yaml"
    overrides.write_text(
        "demo/model: [tools]\n",
        encoding="utf-8",
    )

    caps = load_capabilities(str(providers), str(overrides))
    assert caps["demo/model"] == Capabilities.TEXT | Capabilities.TOOL_CALLING
    assert not caps["demo/model"] & Capabilities.VISION


def test_duplicate_candidates_are_stably_deduplicated():
    sel = ChainSelector(model_scores={"p/a": 0.45, "p/b": 0.80}, capabilities={})
    candidates = [
        {"provider": "p", "model": "a", "priority": 1},
        {"provider": "p", "model": "a", "priority": 2},
        {"provider": "p", "model": "b", "priority": 3},
        {"provider": "p", "model": "b", "priority": 4},
    ]

    result = sel.select(_features(), Tier.T2, candidates)

    assert [(c.provider, c.model) for c in result.chain] == [("p", "a"), ("p", "b")]
    assert [c.priority for c in result.chain] == [1, 3]
    assert [c.model for c in result.escalation] == ["b"]


def test_duplicate_pins_produce_one_pinned_candidate():
    sel = ChainSelector(
        model_scores={},
        capabilities={},
        pins=[{"provider": "p", "model": "a"}, {"provider": "p", "model": "a"}],
    )
    result = sel.select(
        _features(),
        Tier.T1,
        [
            {"provider": "p", "model": "a", "priority": 2},
            {"provider": "p", "model": "a", "priority": 3},
        ],
    )

    assert [(c.provider, c.model) for c in result.chain] == [("p", "a")]
    assert result.chain[0].reason == "PINNED"
    assert result.chain[0].priority == 1


def test_media_only_model_excluded():
    caps = {}
    sel = ChainSelector(model_scores={}, capabilities=caps, predictor=None)
    cands = [
        {"provider": "openai", "model": "text-embedding-3-small", "priority": 1},
        {"provider": "groq", "model": "llama-3.1-8b-instant", "priority": 2},
    ]
    res = sel.select(_features(), Tier.T1, cands)
    assert [c.model for c in res.chain] == ["llama-3.1-8b-instant"]


def test_perf_50_candidates():
    sel = ChainSelector(model_scores={}, capabilities={}, predictor=None)
    cands = [
        {"provider": f"p{i}", "model": f"m{i}", "priority": i + 1} for i in range(50)
    ]
    start = time.monotonic()
    res = sel.select(_features(), Tier.T1, cands)
    elapsed = (time.monotonic() - start) * 1000
    assert len(res.chain) == 50
    assert elapsed < 100.0


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_coding_chains_select_live_vision_candidate():
    """The real coding chains must select a verified vision model (#486 AC1)."""
    import yaml

    from proxy_app.routing.tier_classifier import classify_request, load_model_scores

    db_path = REPO_ROOT / "config" / "providers_database.yaml"
    caps = load_capabilities(
        str(db_path), str(REPO_ROOT / "config" / "model_capabilities.yaml")
    )
    scores = load_model_scores(str(REPO_ROOT / "config" / "model_rankings.yaml"))
    data = yaml.safe_load((REPO_ROOT / "config" / "virtual_models.yaml").read_text())
    verified = {
        ("mistral", "pixtral-12b-2409"),
        ("nvidia", "nvidia/nemotron-3.5-super-vl-120b-a12b"),
    }
    features = _features(
        capabilities=Capabilities.TEXT | Capabilities.VISION, image_count=1
    )
    tier = classify_request(features)

    for vm in ("coding-smart", "coding-fast"):
        chain = data["virtual_models"][vm]["fallback_chain"]
        pairs = {(entry["provider"], entry["model"]) for entry in chain}
        assert verified <= pairs, f"{vm} missing verified vision entries: {verified - pairs}"

        selector = ChainSelector(model_scores=scores, capabilities=caps, predictor=None)
        result = selector.select(features, tier, chain)
        chosen = {(c.provider, c.model) for c in result.chain}
        assert result.chain, f"{vm} produced an empty vision chain at {tier}"
        assert all(c.capabilities & Capabilities.VISION for c in result.chain)
        assert chosen & verified, f"{vm} did not select a verified vision model"


def test_real_catalog_has_a_live_vision_candidate():
    """A vision request must reach a model that accepts images (#486 AC1).

    gemini free ids are dead/quota'd and cerebras has no vision model, so the
    catalog has to carry at least one live vision path or case 3 of the smoke
    matrix can only ever fail with no candidates.
    """
    import yaml

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import chain_policy

    db_path = REPO_ROOT / "config" / "providers_database.yaml"
    caps = load_capabilities(str(db_path), str(REPO_ROOT / "config" / "model_capabilities.yaml"))
    policy = chain_policy.load_policy()
    database = yaml.safe_load(db_path.read_text())

    vision = []
    for provider in database["providers"]:
        if not provider.get("enabled", True):
            continue
        pid = provider["id"]
        for model in provider.get("free_models") or []:
            mid = model["id"]
            key = f"{pid}/{mid}"
            if not (caps.get(key, Capabilities.TEXT) & Capabilities.VISION):
                continue
            if chain_policy.blocked_reason(pid, mid, policy):
                continue
            vision.append((pid, mid))

    assert ("nvidia", "nvidia/nemotron-3.5-super-vl-120b-a12b") in vision
    assert ("mistral", "pixtral-12b-2409") in vision

    sel = ChainSelector(
        model_scores={f"{p}/{m}": 0.5 for p, m in vision},
        capabilities=caps,
        predictor=None,
    )
    candidates = [
        {"provider": p, "model": m, "priority": i + 1} for i, (p, m) in enumerate(vision)
    ]
    res = sel.select(
        _features(capabilities=Capabilities.TEXT | Capabilities.VISION),
        Tier.T2,
        candidates,
    )
    assert res.chain
    assert all(c.capabilities & Capabilities.VISION for c in res.chain)
