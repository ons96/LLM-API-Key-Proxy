import csv
import json
import math
import yaml
from pathlib import Path
import re

# Configuration
LEADERBOARD_FILE = "aggregated_leaderboard.csv"
OUTPUT_FILE = "config/aliases.yaml"

# Manual Speed Map for models missing TPS
FALLBACK_SPEEDS = {
    "claude-3-5-sonnet": 70.0,
    "gpt-4o": 80.0,
    "gemini-1-5-pro": 50.0,
    "deepseek-v3": 60.0,
    "llama-3-1-70b": 100.0,
    "llama-3-3-70b": 110.0,
}

# Providers Map
PROVIDER_MAP = {
    "DeepSeek": ["deepseek", "together", "g4f", "deepinfra"],
    "Anthropic": ["anthropic", "g4f", "together"],
    "OpenAI": ["openai", "g4f"],
    "Google": ["gemini", "g4f", "google"],
    "Meta": ["groq", "cerebras", "together", "deepinfra", "g4f"],
    "Mistral": ["mistral", "deepinfra", "g4f", "together"],
    "Perplexity": ["perplexity"],
    "xAI": ["xai", "g4f"],
    "Qwen": ["alibaba", "deepinfra", "g4f"],
    "Alibaba": ["alibaba", "deepinfra", "g4f"],
    "Cohere": ["cohere"],
    "01.AI": ["deepinfra"],
    "NVIDIA": ["nvidia", "deepinfra"],
    "Microsoft Azure": ["azure"],
    "Minimax": ["minimax", "g4f"],
}


def normalize_slug(name):
    """Normalize model name to slug."""
    clean_name = re.sub(r"[^\w\s\-\.]", "", name)
    slug = (
        clean_name.lower()
        .replace(" ", "-")
        .replace(".", "-")
        .replace("(", "")
        .replace(")", "")
    )
    return slug


def load_aggregated_data():
    """Load models from aggregated multi-source CSV."""
    models_data = []
    try:
        root_dir = Path(__file__).resolve().parent.parent.parent
        csv_path = root_dir / "llm-leaderboard" / LEADERBOARD_FILE

        # Fallback to original CSV if aggregated doesn't exist
        if not csv_path.exists():
            csv_path = root_dir / "llm-leaderboard" / "artificial_analysis_models.csv"

        if not csv_path.exists():
            print(f"Error: Leaderboard file not found at {csv_path}")
            return []

        print(f"Reading data from {csv_path}...")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("name", row.get("Model", ""))
                if not name:
                    continue

                slug = row.get("slug", normalize_slug(name))

                # New aggregated format
                agentic = float(row.get("agentic_coding") or 0)
                coding = float(row.get("coding") or 0)
                overall = float(row.get("overall") or 0)
                livecodebench_raw = float(row.get("livecodebench") or 0)
                # livecodebench in aggregated CSV is already a percentage (0-100)
                # If it's <= 1, it's a decimal and needs conversion
                livecodebench = (
                    livecodebench_raw * 100
                    if livecodebench_raw <= 1
                    else livecodebench_raw
                )

                # Fallback to old format
                if agentic == 0:
                    agentic = float(
                        row.get("eval_artificial_analysis_coding_index") or 0
                    )
                    coding = float(row.get("coding") or 0)
                    overall = float(row.get("overall") or 0)
                    livecodebench = float(row.get("eval_livecodebench") or 0) * 100

                # Priority Chain
                quality_score = max(agentic, coding, livecodebench, overall)

                # TPS Extraction
                tps = float(row.get("tps") or 0)
                if tps == 0:
                    # Try old format
                    tps = float(row.get("median_output_tokens_per_second") or 0)

                # Fallback TPS
                if tps == 0:
                    for k, v in FALLBACK_SPEEDS.items():
                        if k in slug:
                            tps = v
                            break
                    if tps == 0:
                        tps = 20.0

                if quality_score > 0:
                    creator = row.get("creator", "Unknown")
                    if creator == "Unknown":
                        if "gpt" in slug:
                            creator = "OpenAI"
                        elif "claude" in slug:
                            creator = "Anthropic"
                        elif "gemini" in slug:
                            creator = "Google"
                        elif "deepseek" in slug:
                            creator = "DeepSeek"

                    models_data.append(
                        {
                            "name": name,
                            "slug": slug,
                            "creator": creator,
                            "score": quality_score,
                            "tps": tps,
                        }
                    )
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        import traceback

        traceback.print_exc()

    return models_data


def calculate_effective_score(quality_score, tps, slug=""):
    """Score models by quality alone (tps parameter retained for signature compatibility)."""
    if quality_score < 30:
        return 0

    # Tie-breaker boost for model class (Opus > Sonnet > Haiku)
    tier_boost = 0
    if "opus" in slug:
        tier_boost = 2.0
    elif "sonnet" in slug:
        tier_boost = 1.0

    # Version boost (Newer > Older)
    if "4-5" in slug or "4.5" in slug:
        tier_boost += 0.5
    elif "3-7" in slug or "3.7" in slug:
        tier_boost += 0.3

    adjusted_quality = quality_score + tier_boost
    return adjusted_quality**2


def generate_aliases():
    models = load_aggregated_data()

    ranked_candidates = []

    for m in models:
        eff_score = calculate_effective_score(m["score"], m["tps"], m["slug"])

        valid_providers = PROVIDER_MAP.get(m["creator"], ["g4f"])

        for provider in valid_providers:
            if provider == "groq" and "llama" not in m["slug"]:
                continue

            ranked_candidates.append(
                {
                    "provider": provider,
                    "model": m["slug"],
                    "real_name": m["name"],
                    "base_score": m["score"],
                    "tps": m["tps"],
                    "effective_score": eff_score,
                }
            )

    ranked_candidates.sort(key=lambda x: x["effective_score"], reverse=True)

    unique_candidates = []
    seen = set()
    for c in ranked_candidates:
        base_slug = c["model"].replace("-instruct", "")
        key = (c["provider"], base_slug)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    ranked_candidates = unique_candidates

    aliases_config = {
        "aliases": {
            "coding": {
                "description": "Aggregated Leaderboard Top Coding Models",
                "strategy": "chain_fallback",
                "candidates": [],
            },
            "smart": {
                "description": "Highest Agentic Coding Score",
                "strategy": "chain_fallback",
                "candidates": [],
            },
            "fast": {
                "description": "Highest TPS (Quality > 30)",
                "strategy": "chain_fallback",
                "candidates": [],
            },
        }
    }

    for c in ranked_candidates[:40]:
        aliases_config["aliases"]["coding"]["candidates"].append(
            {"provider": c["provider"], "model": c["model"]}
        )

    smart_candidates = sorted(
        ranked_candidates, key=lambda x: x["base_score"], reverse=True
    )
    for c in smart_candidates[:20]:
        aliases_config["aliases"]["smart"]["candidates"].append(
            {"provider": c["provider"], "model": c["model"]}
        )

    fast_candidates = sorted(
        [c for c in ranked_candidates if c["base_score"] > 40],
        key=lambda x: x["tps"],
        reverse=True,
    )
    for c in fast_candidates[:20]:
        aliases_config["aliases"]["fast"]["candidates"].append(
            {"provider": c["provider"], "model": c["model"]}
        )

    out_path = Path(__file__).parent.parent / OUTPUT_FILE
    with open(out_path, "w") as f:
        yaml.dump(aliases_config, f, sort_keys=False)

    print(f"Generated {len(ranked_candidates)} unique candidates into {out_path}")
    print("\nTop 5 'Coding' Models (Aggregated):")
    for c in ranked_candidates[:5]:
        safe_name = c["real_name"].encode("ascii", "ignore").decode("ascii")
        print(f"  {safe_name} ({c['provider']})")
        print(
            f"    Score: {c['effective_score']:.0f} | Quality: {c['base_score']} | TPS: {c['tps']}"
        )


if __name__ == "__main__":
    generate_aliases()
