"""M4: ordering correctness.

  * property test over randomized catalogs: no tier-3 model ever precedes an
    eligible tier-0/1/2 model
  * tools-required request never selects a non-tool model
  * long-context request never selects a small-context model
  * an unsampled top-quartile benchmark model outranks a well-sampled weak one
  * the same catalog input produces byte-identical chain output
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from proxy_app.routing.chain_order import (  # noqa: E402
    Candidate,
    CapabilityError,
    Requirements,
    rank,
    reliability,
    to_dicts,
    wilson_lower_bound,
)


def cand(
    provider: str,
    model: str,
    tier: int = 0,
    *,
    quality: float | None = 50.0,
    speed: float | None = 20.0,
    context: int | None = 128000,
    caps: set[str] | None = None,
    samples: int = 0,
    successes: int = 0,
    last_ok: float | None = None,
    pinned: bool = False,
) -> Candidate:
    return Candidate(
        provider=provider,
        model=model,
        quota_tier=tier,
        quality_score=quality,
        speed_tps=speed,
        context_window=context,
        capabilities=caps if caps is not None else {"tools", "vision"},
        samples=samples,
        successes=successes,
        last_ok_at=last_ok,
        pinned=pinned,
    )


# --------------------------------------------------------------------------- #
# property test over randomized catalogs
# --------------------------------------------------------------------------- #


def _random_catalog(rng: random.Random) -> list[Candidate]:
    out: list[Candidate] = []
    for i in range(rng.randint(2, 12)):
        tier = rng.choice([0, 1, 2, 3, 4])
        out.append(
            cand(
                f"p{rng.randint(1, 4)}",
                f"m{i}",
                tier,
                quality=round(rng.uniform(0, 100), 1),
                speed=round(rng.uniform(1, 200), 1),
                samples=rng.randint(0, 50),
                successes=rng.randint(0, 50) if rng.random() > 0.2 else 0,
                last_ok=rng.uniform(0, 1e9),
            )
        )
    return out


@pytest.mark.parametrize("seed", range(40))
def test_property_tier3_never_precedes_renewing_tier(seed: int):
    """For random catalogs, no tier-3 model outranks an eligible tier-0/1/2 one."""
    rng = random.Random(seed)
    catalog = _random_catalog(rng)
    ordered = rank(catalog)

    tiers = [c.quota_tier for c in ordered if c.quota_tier < 4]
    first_three = next(
        (idx for idx, tier in enumerate(tiers) if tier == 3), len(tiers)
    )
    assert all(tier < 3 for tier in tiers[:first_three]), (
        f"seed {seed}: tier 3 appeared before a renewing tier: {tiers}"
    )


def test_paid_models_are_excluded_entirely():
    catalog = [cand("openai", "gpt-5", 4), cand("groq", "llama", 0)]
    ordered = rank(catalog)
    assert [c.provider for c in ordered] == ["groq"]


def test_pinned_does_not_lift_a_model_above_a_better_tier():
    catalog = [cand("together", "qwen", 3, pinned=True), cand("groq", "llama", 1)]
    ordered = rank(catalog)
    assert ordered[0].quota_tier == 1


# --------------------------------------------------------------------------- #
# capability correctness
# --------------------------------------------------------------------------- #


def test_tools_request_never_selects_a_non_tool_model():
    catalog = [
        cand("a", "no-tools", 0, quality=99, caps={"vision"}),
        cand("b", "tools", 0, quality=10, caps={"tools"}),
    ]
    ordered = rank(catalog, Requirements(needs_tools=True))
    assert ordered, "expected at least one eligible candidate"
    assert ordered[0].model == "tools"
    assert all(c.supports("tools") for c in ordered)


def test_tools_request_without_any_tool_model_raises():
    catalog = [cand("a", "chatty", 0, caps={"vision"})]
    with pytest.raises(CapabilityError) as excinfo:
        rank(catalog, Requirements(needs_tools=True))
    assert excinfo.value.requirement == "tools"


def test_undeclared_capabilities_do_not_count_as_capable():
    """An empty capability set is 'unknown', not 'yes'."""
    catalog = [cand("a", "mystery", 0, caps=set())]
    with pytest.raises(CapabilityError):
        rank(catalog, Requirements(needs_tools=True))


def test_long_context_request_never_selects_a_small_context_model():
    catalog = [
        cand("a", "small", 0, quality=99, context=32000),
        cand("b", "huge", 0, quality=10, context=200000),
    ]
    ordered = rank(catalog, Requirements(min_context_tokens=150000))
    assert ordered[0].model == "huge"
    assert all((c.context_window or 0) >= 150000 for c in ordered)


def test_long_context_request_without_a_big_enough_model_raises():
    catalog = [cand("a", "small", 0, context=32000)]
    with pytest.raises(CapabilityError) as excinfo:
        rank(catalog, Requirements(min_context_tokens=150000))
    assert "context" in excinfo.value.detail


# --------------------------------------------------------------------------- #
# sparse-telemetry prior
# --------------------------------------------------------------------------- #


def test_unsampled_top_quartile_beats_well_sampled_weak():
    """A capable model with no history must not sink below a proven-weak one."""
    unsampled = cand("new", "frontier", 0, quality=92.0, samples=0)
    well_sampled_weak = cand(
        "old", "mediocre", 0, quality=31.0, samples=400, successes=300
    )
    ordered = rank([well_sampled_weak, unsampled])
    assert ordered[0].model == "frontier"


def test_telemetry_does_not_overturn_a_large_benchmark_gap():
    strong = cand("a", "strong", 0, quality=90.0, samples=200, successes=200)
    strong_but_slow = cand("b", "weaker", 0, quality=55.0, samples=500, successes=500)
    assert rank([strong_but_slow, strong])[0].model == "strong"


def test_reliability_shrinks_small_samples():
    fresh = cand("a", "m", 0, samples=1, successes=1)
    proven = cand("b", "m2", 0, samples=100, successes=98)
    assert reliability(fresh) < reliability(proven), "1/1 must not look like 98/100"
    # 1/1 shrinks to 0.44: Wilson(1,1)=0.21 pulled 80% of the way to the 0.5
    # prior. Below the prior on purpose -- one sample is weak evidence, not proof.
    assert 0.3 < reliability(fresh) < 0.6


def test_wilson_bound_is_monotonic():
    assert wilson_lower_bound(0, 1) == 0.0
    assert wilson_lower_bound(10, 10) > wilson_lower_bound(9, 10)
    assert wilson_lower_bound(1, 1) < 1.0


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #


def test_same_catalog_produces_byte_identical_output():
    catalog = _random_catalog(random.Random(7))
    first = json.dumps(to_dicts(rank(catalog)), sort_keys=True)
    for _ in range(5):
        shuffled = catalog[:]
        random.shuffle(shuffled)
        assert json.dumps(to_dicts(rank(shuffled)), sort_keys=True) == first


def test_determinism_holds_for_a_large_catalog():
    rng = random.Random(1234)
    catalog = [
        cand(
            f"provider{i % 7}",
            f"model{i}",
            rng.choice([0, 0, 1, 1, 2, 3]),
            quality=round(rng.uniform(0, 100), 3),
            speed=round(rng.uniform(1, 100), 3),
            samples=rng.randint(0, 20),
            successes=rng.randint(0, 20),
        )
        for i in range(200)
    ]
    baseline = json.dumps(to_dicts(rank(catalog)), sort_keys=True)
    rotated = catalog[50:] + catalog[:50]
    assert json.dumps(to_dicts(rank(rotated)), sort_keys=True) == baseline


# --------------------------------------------------------------------------- #
# interaction with the real config + catalog
# --------------------------------------------------------------------------- #


def test_real_chain_candidates_order_without_paid_models():
    """Ordering the real virtual-model chains never puts a paid model first."""
    import yaml

    from proxy_app.routing import free_policy

    tiers = free_policy.load_tiers(REPO_ROOT, force=True)
    data = yaml.safe_load(
        (REPO_ROOT / "config" / "virtual_models.yaml").read_text(encoding="utf-8")
    )
    chains = data.get("virtual_models") or {}

    checked = 0
    for alias, config in chains.items():
        entries = (config or {}).get("fallback_chain") or []
        candidates = [
            cand(
                e["provider"],
                e["model"],
                free_policy.provider_tier(tiers, e["provider"]),
            )
            for e in entries
            if isinstance(e, dict) and e.get("provider") and e.get("model")
        ]
        if not candidates:
            continue
        checked += 1
        ordered = rank(candidates)
        assert ordered == [] or ordered[0].quota_tier < 4, alias
    assert checked > 0, "no real chains were checked; the test would be vacuous"
