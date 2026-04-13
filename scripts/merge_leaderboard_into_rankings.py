#!/usr/bin/env python3
"""Merge llm-leaderboard-aggregate data into config/model_rankings.yaml.

Reads:
  - leaderboards/gateway_fallback_free_only.csv (from llm-leaderboard-aggregate, downloaded by scrape_leaderboards.yml)
  - leaderboards/agentic_coding_leaderboard.csv
  - config/model_rankings.yaml (existing)

Updates model_rankings.yaml with:
  - agentic_coding scores for models present in leaderboard CSVs
  - Adds new models found in leaderboard that aren't in rankings yet
  - Does NOT remove existing entries

Usage:
  python scripts/merge_leaderboard_into_rankings.py
  python scripts/merge_leaderboard_into_rankings.py --dry-run
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("merge_leaderboard")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
LEADERBOARDS_DIR = CONFIG_DIR / "leaderboards"  # downloaded by scrape_leaderboards.yml
RANKINGS_PATH = CONFIG_DIR / "model_rankings.yaml"

# Also check fallbacks/ directory and sibling repo (for local dev)
FALLBACKS_DIR = CONFIG_DIR / "fallbacks"
SIBLING_LEADERBOARDS_DIR = PROJECT_ROOT.parent / "llm-leaderboard-aggregate" / "leaderboards"


def _slug_to_id(model_id: str) -> str:
    """Normalize model slug for matching."""
    return model_id.lower().replace("_", "-").replace(" ", "-")


def _load_agentic_leaderboard(csv_path: Path) -> dict[str, float]:
    """Load agentic_coding_leaderboard.csv -> {model_id: score}."""
    scores: dict[str, float] = {}
    if not csv_path.exists():
        logger.warning(f"Not found: {csv_path}")
        return scores
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            mid = row.get("model_id", "").strip()
            score_str = row.get("avg_agentic_coding_score", "").strip()
            if mid and score_str:
                try:
                    scores[_slug_to_id(mid)] = float(score_str)
                except ValueError:
                    pass
    logger.info(f"Loaded {len(scores)} scores from {csv_path.name}")
    return scores


def _load_free_provider_leaderboard(csv_path: Path) -> dict[str, dict]:
    """Load gateway_fallback_free_only.csv -> {provider/model_id: row}."""
    rows: dict[str, dict] = {}
    if not csv_path.exists():
        logger.warning(f"Not found: {csv_path}")
        return rows
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            provider = row.get("provider_name", "").strip()
            model_id = row.get("model_id", "").strip()
            if provider and model_id:
                key = f"{provider}/{_slug_to_id(model_id)}"
                # Keep highest-scoring entry per provider/model
                score_str = row.get("avg_agentic_coding_score", "").strip()
                if key not in rows or (score_str and float(score_str or 0) > float(rows[key].get("avg_agentic_coding_score") or 0)):
                    rows[key] = row
    logger.info(f"Loaded {len(rows)} free-provider entries from {csv_path.name}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load existing rankings
    if not RANKINGS_PATH.exists():
        logger.error(f"Rankings file not found: {RANKINGS_PATH}")
        sys.exit(1)

    with open(RANKINGS_PATH) as f:
        rankings = yaml.safe_load(f) or {}

    models = rankings.get("models", [])
    existing_ids = {_slug_to_id(m.get("id", "")): i for i, m in enumerate(models)}
    logger.info(f"Existing model_rankings.yaml: {len(models)} entries")

    # Load leaderboard data - check config/, fallbacks/, and sibling repo
    def _find_csv(name: str) -> Path:
        for d in [LEADERBOARDS_DIR, FALLBACKS_DIR, SIBLING_LEADERBOARDS_DIR]:
            p = d / name
            if p.exists():
                return p
        return LEADERBOARDS_DIR / name  # return non-existent path; caller handles missing

    agentic_path = _find_csv("agentic_coding_leaderboard.csv")
    free_path = _find_csv("gateway_fallback_free_only.csv")

    agentic_scores = _load_agentic_leaderboard(agentic_path)
    free_providers = _load_free_provider_leaderboard(free_path)

    if not agentic_scores and not free_providers:
        logger.warning("No leaderboard data found. Nothing to merge.")
        sys.exit(0)

    updated = 0
    added = 0

    # Update existing model scores
    for model in models:
        mid_raw = model.get("id", "")
        mid = _slug_to_id(mid_raw)

        # Try to find score: exact match, then bare model name
        score = agentic_scores.get(mid)
        if score is None and "/" in mid:
            score = agentic_scores.get(mid.split("/", 1)[1])

        if score is not None:
            old = model.get("scores", {}).get("agentic_coding", -1)
            if abs(old - score) > 0.1:  # Only update if changed significantly
                model.setdefault("scores", {})["agentic_coding"] = round(score, 2)
                # Also update composite and humaneval if they were derived from agentic
                model["scores"]["swe_bench"] = round(score, 2)
                model["scores"]["humaneval"] = round(score, 2)
                updated += 1
                logger.debug(f"Updated {mid_raw}: {old:.1f} -> {score:.1f}")

    # Add new models from agentic leaderboard not in rankings
    for slug, score in agentic_scores.items():
        if slug in existing_ids:
            continue
        # Only add models with scores worth tracking (> 40)
        if score < 40.0:
            continue
        # Build a provider/model ID - slugs from leaderboard are bare model names
        new_entry = {
            "id": slug,
            "name": slug.replace("-", " ").title(),
            "scores": {
                "speed_tps": -1,
                "livebench_coding": -1,
                "aider": -1,
                "swe_bench": round(score, 2),
                "swe_bench_verified": -1,
                "swe_rebench": -1,
                "humaneval": round(score, 2),
                "bigcodebench": -1,
                "gso_bench": -1,
                "ts_bench": -1,
                "vals_ai": -1,
                "agentic_coding": round(score, 2),
                "composite_score": round(score * 0.95, 2),  # rough estimate
            },
            "best_for": ["coding-smart", "coding-elite"] if score >= 65 else ["coding-smart"],
        }
        models.append(new_entry)
        existing_ids[slug] = len(models) - 1
        added += 1
        logger.debug(f"Added new model: {slug} score={score:.1f}")

    logger.info(f"Updated {updated} existing entries, added {added} new entries")

    if args.dry_run:
        logger.info("Dry run - not writing changes")
        return

    if updated > 0 or added > 0:
        rankings["models"] = models
        with open(RANKINGS_PATH, "w") as f:
            yaml.dump(rankings, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        logger.info(f"Written {RANKINGS_PATH} ({len(models)} total entries)")
    else:
        logger.info("No changes needed")


if __name__ == "__main__":
    main()
