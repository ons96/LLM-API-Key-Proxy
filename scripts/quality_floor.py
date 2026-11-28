#!/usr/bin/env python3
"""Quality-floor check (#1422).

Verifies that the quality floor configuration stays enforceable:

* every virtual model has a valid intelligence floor (``I_floor`` in ``[0, 1]``)
  either explicitly or through the tier-config default;
* every virtual model chain keeps at least one candidate that meets its floor
  whenever benchmark data exists for that chain.

A chain without any benchmark row is reported as unverified, not failed: unknown
quality is not the same as low quality. Exit code 0 means the floors hold,
1 means at least one floor is broken.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TIER_CONFIG_PATH = REPO_ROOT / "config" / "tier_config.yaml"
VIRTUAL_MODELS_PATH = REPO_ROOT / "config" / "virtual_models.yaml"
RANKINGS_PATH = REPO_ROOT / "config" / "model_rankings.yaml"
SCORE_METRIC = "composite_score"
SCORE_SCALE = 100.0


def load_yaml(path: Path) -> Dict:
    """Load a YAML mapping from disk."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def known_scores(rankings: Dict) -> Dict[str, float]:
    """Index benchmark scores by model id and by bare model name."""
    scores: Dict[str, float] = {}
    rows = rankings.get("models", []) if isinstance(rankings, dict) else rankings
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id", ""))
        score = (row.get("scores") or {}).get(SCORE_METRIC)
        if not model_id or not isinstance(score, (int, float)) or score < 0:
            continue
        scores[model_id.lower()] = float(score)
        scores[model_id.lower().split("/")[-1]] = float(score)
    return scores


def lookup_score(model: str, scores: Dict[str, float]) -> float | None:
    """Return the benchmark score for a model, if it is known."""
    key = model.strip().lower()
    if key in scores:
        return scores[key]
    bare = key.split("/")[-1]
    return scores.get(bare)


def chain_models(chain: object) -> List[str]:
    """Extract model names from a fallback chain definition."""
    models = []
    if not isinstance(chain, list):
        return models
    for entry in chain:
        if isinstance(entry, dict):
            model = entry.get("model")
            if isinstance(model, str):
                models.append(model)
        elif isinstance(entry, str):
            models.append(entry)
    return models


def floor_for(tier_config: Dict, virtual_model: str) -> float | None:
    """Resolve the intelligence floor for a virtual model."""
    default = tier_config.get("default") or {}
    floors = tier_config.get("virtual_models") or {}
    entry = floors.get(virtual_model)
    source = entry if isinstance(entry, dict) else default
    if not isinstance(source, dict):
        return None
    floor = source.get("I_floor")
    if not isinstance(floor, (int, float)):
        return None
    return float(floor)


def run() -> int:
    """Run the quality-floor check."""
    try:
        tier_config = load_yaml(TIER_CONFIG_PATH)
        virtual_models = load_yaml(VIRTUAL_MODELS_PATH)
        rankings = load_yaml(RANKINGS_PATH)
    except FileNotFoundError as exc:
        print(f"quality floor: missing configuration {exc}", file=sys.stderr)
        return 1

    scores = known_scores(rankings)
    chains = virtual_models.get("virtual_models") or {}
    if not isinstance(chains, dict) or not chains:
        print("quality floor: no virtual models configured", file=sys.stderr)
        return 1

    violations: List[str] = []
    unverified: List[str] = []

    for virtual_model, config in sorted(chains.items()):
        floor = floor_for(tier_config, virtual_model)
        if floor is None:
            violations.append(f"{virtual_model}: no valid I_floor in tier_config.yaml")
            continue
        if not 0.0 <= floor <= 1.0:
            violations.append(f"{virtual_model}: I_floor {floor} outside [0, 1]")
            continue
        models = chain_models((config or {}).get("fallback_chain"))
        if not models:
            violations.append(f"{virtual_model}: empty fallback chain")
            continue
        known = [
            score
            for score in (lookup_score(model, scores) for model in models)
            if score is not None
        ]
        if not known:
            unverified.append(virtual_model)
            continue
        threshold = floor * SCORE_SCALE
        if max(known) < threshold:
            violations.append(
                f"{virtual_model}: best known score {max(known):.2f} below floor "
                f"{threshold:.2f}"
            )

    for name in unverified:
        print(f"quality floor: {name}: no benchmark row, floor unverified")

    if violations:
        print("quality floor: FAILED", file=sys.stderr)
        for line in violations:
            print(f"  - {line}", file=sys.stderr)
        return 1

    print(f"quality floor: OK ({len(chains)} virtual models checked)")
    return 0


def main(argv: List[str]) -> int:
    """CLI entry point."""
    return run()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
