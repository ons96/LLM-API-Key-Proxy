#!/usr/bin/env python3
"""Unified Free LLM Ranking — combines all data sources into one ranked list.

Data sources (in priority order for TPS):
  1. config/benchmark_results.csv  — measured TPS from benchmark_providers.py (most accurate)
  2. config/providers_database.yaml free_models[].tps — declared TPS per model
  3. config/provider_speeds.json   — legacy static TPS per provider (fallback)
  4. Telemetry DB                  — live EWMA TPS from actual gateway calls

Data sources for quality score:
  1. config/model_rankings.yaml agentic_coding score (from leaderboard merge)
  2. Hardcoded fallback 0.5 if model unknown

Rate limit factor:
  - Models from providers with low daily limits get a penalty
  - Formula: rate_limit_factor = min(1.0, daily_requests / 1440)  (1440 = 1 req/min)
  - Providers with unlimited daily get factor = 1.0

Outputs:
  - config/unified_model_ranking.yaml  (replaces static provider_speeds.json over time)
  - Printed ranked table per virtual model

Usage:
  python scripts/unified_ranking.py               # Full ranking
  python scripts/unified_ranking.py --virtual coding-fast  # Single virtual model
  python scripts/unified_ranking.py --update-speeds        # Update provider_speeds.json
"""

import argparse
import csv
import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("unified_ranking")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
ROUTER_CONFIG = CONFIG_DIR / "router_config.yaml"
PROVIDERS_DB = CONFIG_DIR / "providers_database.yaml"
MODEL_RANKINGS = CONFIG_DIR / "model_rankings.yaml"
VIRTUAL_MODELS = CONFIG_DIR / "virtual_models.yaml"
PROVIDER_SPEEDS = CONFIG_DIR / "provider_speeds.json"
BENCHMARK_CSV = CONFIG_DIR / "benchmark_results.csv"
TELEMETRY_DB = Path("/tmp/llm_proxy_telemetry.db")
OUTPUT_YAML = CONFIG_DIR / "unified_model_ranking.yaml"


def load_agentic_scores() -> dict[str, float]:
    """Load agentic coding scores from model_rankings.yaml."""
    if not MODEL_RANKINGS.exists():
        return {}
    with open(MODEL_RANKINGS) as f:
        data = yaml.safe_load(f) or {}
    scores = {}
    for m in data.get("models", []):
        mid = m.get("id", "").lower()
        score = m.get("scores", {}).get("agentic_coding", -1)
        if score > 0:
            scores[mid] = score
            if "/" in mid:
                scores[mid.split("/", 1)[1]] = score
    return scores


def load_benchmark_tps() -> dict[str, float]:
    """Load measured TPS from benchmark_results.csv (most accurate source)."""
    if not BENCHMARK_CSV.exists():
        return {}
    results: dict[str, list[float]] = {}
    with open(BENCHMARK_CSV) as f:
        for row in csv.DictReader(f):
            if row.get("success") != "True":
                continue
            key = f"{row['provider']}/{row['model']}"
            tps_str = row.get("tps", "")
            if tps_str:
                results.setdefault(key, []).append(float(tps_str))
    return {k: sum(v) / len(v) for k, v in results.items()}


def load_telemetry_tps() -> dict[str, float]:
    """Load recent average TPS from telemetry SQLite."""
    if not TELEMETRY_DB.exists():
        return {}
    try:
        conn = sqlite3.connect(str(TELEMETRY_DB))
        rows = conn.execute("""
            SELECT provider, model, AVG(tokens_per_second)
            FROM api_calls
            WHERE success = 1
              AND tokens_per_second IS NOT NULL
              AND timestamp >= datetime('now', '-7 days')
            GROUP BY provider, model
            HAVING COUNT(*) >= 3
        """).fetchall()
        conn.close()
        return {f"{r[0]}/{r[1]}": float(r[2]) for r in rows}
    except Exception:
        return {}


def load_provider_rate_limits() -> dict[str, dict]:
    """Load rate limits from providers_database.yaml."""
    if not PROVIDERS_DB.exists():
        return {}
    with open(PROVIDERS_DB) as f:
        db = yaml.safe_load(f) or {}
    limits = {}
    for p in db.get("providers", []):
        if not isinstance(p, dict):
            continue
        pid = p.get("id", "")
        rl = p.get("rate_limits", {})
        # Declared TPS per model
        model_tps = {}
        for m in p.get("free_models", []):
            if isinstance(m, dict) and m.get("id") and m.get("tps"):
                model_tps[m["id"]] = m["tps"]
        limits[pid] = {
            "rpm": rl.get("rpm", 30),
            "daily": rl.get("daily"),  # None = unlimited
            "model_tps": model_tps,
        }
    return limits


def load_legacy_speeds() -> dict[str, float]:
    """Load legacy provider_speeds.json."""
    if not PROVIDER_SPEEDS.exists():
        return {}
    with open(PROVIDER_SPEEDS) as f:
        data = json.load(f)
    return {k: v.get("tps", 0) for k, v in data.items() if isinstance(v, dict)}


def compute_rate_limit_factor(daily: Optional[int]) -> float:
    """Compute rate limit penalty factor (0.0-1.0).
    Providers with low daily limits get penalized in fallback ordering.
    Unlimited = 1.0, 14400/day (Groq) = ~0.5, 1500/day (Gemini) = ~0.1.
    """
    if daily is None:  # unlimited
        return 1.0
    # 1440 requests/day = 1/min sustained = "reasonable" floor
    # Scale: 14400 -> 0.87, 1500 -> 0.51, 200 -> 0.12, 100 -> 0.07
    import math
    return min(1.0, math.log10(max(daily, 1)) / math.log10(86400))  # 86400 = 1/sec


def get_tps(provider: str, model: str,
           benchmark_tps: dict, telemetry_tps: dict,
           rate_limits: dict, legacy_speeds: dict) -> float:
    """Get best available TPS estimate for a provider/model."""
    key = f"{provider}/{model}"
    # Priority: benchmark measured > telemetry live > declared per-model > legacy provider
    if key in benchmark_tps:
        return benchmark_tps[key]
    if key in telemetry_tps:
        return telemetry_tps[key]
    # Declared per-model TPS from providers_database.yaml
    model_tps = rate_limits.get(provider, {}).get("model_tps", {})
    if model in model_tps:
        return float(model_tps[model])
    # Legacy provider-level TPS
    if provider in legacy_speeds:
        return legacy_speeds[provider]
    return 50.0  # conservative default


def rank_candidates(candidates: list[dict], agentic_scores: dict,
                   benchmark_tps: dict, telemetry_tps: dict,
                   rate_limits: dict, legacy_speeds: dict,
                   mode: str = "virtual") -> list[dict]:
    """Rank a list of provider/model candidates by composite score."""
    ranked = []
    for entry in candidates:
        provider = entry.get("provider", "")
        model = entry.get("model", "")
        model_lower = model.lower()

        # Agentic score (0-1)
        ag = agentic_scores.get(model_lower, agentic_scores.get(f"{provider}/{model_lower}", 50.0))
        ag_norm = min(ag / 100.0, 1.0)

        # TPS score (0-1, normalized to 3000 max)
        tps = get_tps(provider, model, benchmark_tps, telemetry_tps, rate_limits, legacy_speeds)
        tps_norm = min(tps / 3000.0, 1.0)

        # Rate limit penalty
        daily = rate_limits.get(provider, {}).get("daily")
        rl_factor = compute_rate_limit_factor(daily)

        # Composite score
        if mode == "fast":
            # Speed-dominant: TPS 60% + agentic 30% + rate_limit 10%
            score = (tps_norm * 0.60) + (ag_norm * 0.30) + (rl_factor * 0.10)
        elif mode == "elite":
            # Quality-dominant: agentic 75% + TPS 20% + rate_limit 5%
            score = (ag_norm * 0.75) + (tps_norm * 0.20) + (rl_factor * 0.05)
        else:  # smart / virtual (default)
            # Balanced: agentic 65% + TPS 25% + rate_limit 10%
            score = (ag_norm * 0.65) + (tps_norm * 0.25) + (rl_factor * 0.10)

        ranked.append({
            **entry,
            "agentic_score": round(ag, 1),
            "tps": round(tps, 0),
            "rl_factor": round(rl_factor, 2),
            "composite_score": round(score, 4),
            "tps_source": (
                "benchmark" if f"{provider}/{model}" in benchmark_tps else
                "telemetry" if f"{provider}/{model}" in telemetry_tps else
                "declared" if model in rate_limits.get(provider, {}).get("model_tps", {}) else
                "legacy"
            ),
        })

    return sorted(ranked, key=lambda x: -x["composite_score"])


def load_virtual_models() -> dict:
    """Load virtual models with their fallback chains."""
    if not VIRTUAL_MODELS.exists():
        return {}
    with open(VIRTUAL_MODELS) as f:
        data = yaml.safe_load(f) or {}
    return data.get("virtual_models", {})


def main():
    parser = argparse.ArgumentParser(description="Unified LLM ranking across all data sources")
    parser.add_argument("--virtual", help="Only rank this virtual model (e.g. coding-fast)")
    parser.add_argument("--update-speeds", action="store_true",
                        help="Update provider_speeds.json with benchmark/telemetry data")
    parser.add_argument("--output", action="store_true",
                        help="Write unified_model_ranking.yaml")
    args = parser.parse_args()

    logger.info("Loading data sources...")
    agentic_scores = load_agentic_scores()
    benchmark_tps = load_benchmark_tps()
    telemetry_tps = load_telemetry_tps()
    rate_limits = load_provider_rate_limits()
    legacy_speeds = load_legacy_speeds()
    virtual_models = load_virtual_models()

    logger.info(f"  Agentic scores: {len(agentic_scores)} models")
    logger.info(f"  Benchmark TPS:  {len(benchmark_tps)} measurements")
    logger.info(f"  Telemetry TPS:  {len(telemetry_tps)} measurements")
    logger.info(f"  Rate limits:    {len(rate_limits)} providers")

    output_data = {"virtual_models": {}}

    target_vms = [args.virtual] if args.virtual else list(virtual_models.keys())

    for vm_name in target_vms:
        vm_def = virtual_models.get(vm_name)
        if not vm_def:
            logger.warning(f"Virtual model '{vm_name}' not found")
            continue

        candidates = vm_def.get("fallback_chain") or vm_def.get("candidates", [])
        if not candidates:
            continue

        # Determine scoring mode from name
        if "fast" in vm_name:
            mode = "fast"
        elif "elite" in vm_name:
            mode = "elite"
        else:
            mode = "smart"

        ranked = rank_candidates(
            candidates, agentic_scores,
            benchmark_tps, telemetry_tps,
            rate_limits, legacy_speeds,
            mode=mode,
        )

        print(f"\n{'='*70}")
        print(f"  {vm_name} ({mode} mode) — {len(ranked)} candidates")
        print(f"{'='*70}")
        print(f"  {'#':>2}  {'Provider/Model':45} {'Score':>7} {'TPS':>6} {'Agentic':>8} {'RL':>5} {'Source':10}")
        print(f"  {'-'*2}  {'-'*45} {'-'*7} {'-'*6} {'-'*8} {'-'*5} {'-'*10}")
        for i, r in enumerate(ranked[:15], 1):
            label = f"{r['provider']}/{r['model']}"
            print(f"  {i:>2}. {label:45} {r['composite_score']:7.4f} "
                  f"{r['tps']:6.0f} {r['agentic_score']:8.1f} {r['rl_factor']:5.2f} {r['tps_source']:10}")

        output_data["virtual_models"][vm_name] = {
            "description": vm_def.get("description", ""),
            "scoring_mode": mode,
            "ranked_candidates": [
                {
                    "provider": r["provider"],
                    "model": r["model"],
                    "composite_score": r["composite_score"],
                    "tps": r["tps"],
                    "agentic_score": r["agentic_score"],
                    "rl_factor": r["rl_factor"],
                    "tps_source": r["tps_source"],
                }
                for r in ranked
            ],
        }

    if args.output:
        with open(OUTPUT_YAML, "w") as f:
            yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Written: {OUTPUT_YAML}")

    if args.update_speeds:
        # Update provider_speeds.json with best measured/declared TPS per provider
        speeds = {}
        for provider, rl_data in rate_limits.items():
            model_tps = rl_data.get("model_tps", {})
            if model_tps:
                best_tps = max(model_tps.values())
            elif provider in legacy_speeds:
                best_tps = legacy_speeds[provider]
            else:
                best_tps = 50
            # Override with benchmark data if available
            for key, tps in benchmark_tps.items():
                if key.startswith(f"{provider}/"):
                    best_tps = max(best_tps, tps)
            speeds[provider] = {"tps": int(best_tps), "ttft": 0.5}
        with open(PROVIDER_SPEEDS, "w") as f:
            json.dump(speeds, f, indent=2)
        logger.info(f"Updated {PROVIDER_SPEEDS} ({len(speeds)} providers)")


if __name__ == "__main__":
    main()
