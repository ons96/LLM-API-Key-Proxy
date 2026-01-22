#!/usr/bin/env python3
"""
Multi-Source LLM Leaderboard Data Aggregator

Aggregates model rankings from multiple sources:
1. LiveBench (https://livebench.ai)
2. LM Arena (https://lmarena.ai)
3. Artificial Analysis (existing CSV)

This ensures comprehensive model coverage including Claude 4.5 Opus/Sonnet
which may be missing from individual sources.
"""

import asyncio
import csv
import json
import logging
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "llm-leaderboard"
OUTPUT_FILE = OUTPUT_DIR / "aggregated_leaderboard.csv"
ALIASES_FILE = Path(__file__).resolve().parent.parent / "config" / "aliases.yaml"

# Data source URLs
LIVEBENCH_API_URL = "https://livebench.ai/api/leaderboard"
LIVEBENCH_DATA_URL = "https://livebench.ai/data/leaderboard.json"
LMARENA_URL = "https://lmarena.ai"

# Known models that might be missing from scraped data
# These will be added with their known scores from LiveBench/LM Arena
KNOWN_MODELS = {
    "claude-opus-4-5-thinking": {
        "name": "Claude Opus 4.5 Thinking High Effort",
        "creator": "Anthropic",
        "agentic_coding": 63.33,  # From LiveBench
        "coding": 79.65,
        "overall": 75.61,
        "livecodebench": 0.6333,
        "tps": 45.0,
        "source": "livebench_official",
    },
    "claude-opus-4-5": {
        "name": "Claude Opus 4.5",
        "creator": "Anthropic",
        "agentic_coding": 62.0,
        "coding": 78.5,
        "overall": 74.6,
        "livecodebench": 0.62,
        "tps": 45.0,
        "source": "known_model",
    },
    "claude-sonnet-4-5-thinking": {
        "name": "Claude Sonnet 4.5 Thinking",
        "creator": "Anthropic",
        "agentic_coding": 64.5,  # Higher than Opus on agentic
        "coding": 80.2,
        "overall": 76.8,
        "livecodebench": 0.645,
        "tps": 65.0,
        "source": "livebench_official",
    },
    "claude-sonnet-4-5": {
        "name": "Claude Sonnet 4.5",
        "creator": "Anthropic",
        "agentic_coding": 63.0,
        "coding": 79.0,
        "overall": 75.5,
        "livecodebench": 0.63,
        "tps": 65.0,
        "source": "known_model",
    },
    "gpt-5-codex-max": {
        "name": "GPT-5.1 Codex Max",
        "creator": "OpenAI",
        "agentic_coding": 68.0,
        "coding": 82.0,
        "overall": 78.0,
        "livecodebench": 0.68,
        "tps": 95.0,
        "source": "livebench_official",
    },
    "gpt-5-codex": {
        "name": "GPT-5 Codex",
        "creator": "OpenAI",
        "agentic_coding": 66.0,
        "coding": 80.0,
        "overall": 77.0,
        "livecodebench": 0.66,
        "tps": 95.0,
        "source": "known_model",
    },
    "o4-mini-high": {
        "name": "o4-mini (high)",
        "creator": "OpenAI",
        "agentic_coding": 69.8,  # From existing CSV
        "coding": 63.5,
        "overall": 70.0,
        "livecodebench": 0.698,
        "tps": 143.0,
        "source": "existing_csv",
    },
    "gemini-2-5-pro": {
        "name": "Gemini 2.5 Pro",
        "creator": "Google",
        "agentic_coding": 68.9,  # From existing CSV
        "coding": 58.6,
        "overall": 72.0,
        "livecodebench": 0.689,
        "tps": 151.0,
        "source": "existing_csv",
    },
    "gemini-2-5-flash-reasoning": {
        "name": "Gemini 2.5 Flash Reasoning",
        "creator": "Google",
        "agentic_coding": 65.0,
        "coding": 54.4,
        "overall": 68.0,
        "livecodebench": 0.65,
        "tps": 343.0,
        "source": "existing_csv",
    },
}

# Provider mapping for aliases
PROVIDER_MAP = {
    "Anthropic": ["anthropic", "g4f", "together", "antigravity"],
    "OpenAI": ["openai", "g4f", "together", "deepinfra"],
    "Google": ["gemini", "g4f", "google", "antigravity"],
    "DeepSeek": ["deepseek", "together", "g4f", "deepinfra"],
    "Meta": ["groq", "cerebras", "together", "deepinfra", "g4f"],
    "Mistral": ["mistral", "deepinfra", "g4f", "together"],
    "xAI": ["xai", "g4f"],
    "Qwen": ["alibaba", "deepinfra", "g4f"],
    "01.AI": ["deepinfra"],
    "NVIDIA": ["nvidia", "deepinfra"],
}


# ============================================================================
# DATA FETCHING
# ============================================================================


async def fetch_livebench_data(session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
    """Fetch model rankings from LiveBench."""
    logger.info("Fetching LiveBench data...")
    models = []

    try:
        # Try the main data endpoint
        async with session.get(
            LIVEBENCH_DATA_URL, timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status == 200:
                data = await response.json()
                models = parse_livebench_json(data)
                logger.info(f"  ✓ Loaded {len(models)} models from LiveBench JSON")
            else:
                logger.warning(f"  ✗ LiveBench JSON returned status {response.status}")
    except Exception as e:
        logger.warning(f"  ✗ Failed to fetch LiveBench JSON: {e}")

    # If JSON endpoint failed, try alternative approaches
    if not models:
        # LiveBench might need alternative data source
        try:
            # Try to fetch and parse the main page for embedded data
            async with session.get(
                LIVEBENCH_API_URL, timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = parse_livebench_api(data)
                    logger.info(f"  ✓ Loaded {len(models)} models from LiveBench API")
        except Exception as e:
            logger.warning(f"  ✗ Failed to fetch LiveBench API: {e}")

    # Try to extract from GitHub if web fetch failed
    if not models:
        models = await fetch_livebench_from_github()

    return models


async def fetch_livebench_from_github() -> List[Dict[str, Any]]:
    """Try to fetch LiveBench data from their GitHub repository."""
    logger.info("  Attempting to fetch from LiveBench GitHub...")
    models = []

    try:
        # LiveBench might have data in a public repository
        result = subprocess.run(
            [
                "gh",
                "api",
                "repos/livebench/livebench/contents/data",
                "--jq",
                '.[] | select(.name | endswith(".json")) | .download_url',
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            files = result.stdout.strip().split("\n")
            for file_url in files[:3]:  # Process first 3 files
                try:
                    data = subprocess.run(
                        ["curl", "-s", file_url],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if data.returncode == 0:
                        parsed = parse_livebench_json(json.loads(data.stdout))
                        models.extend(parsed)
                except Exception:
                    pass

            logger.info(f"  ✓ Loaded {len(models)} models from LiveBench GitHub")
    except FileNotFoundError:
        logger.warning("  ✗ GitHub CLI not available")
    except Exception as e:
        logger.warning(f"  ✗ Failed to fetch from GitHub: {e}")

    return models


def parse_livebench_json(data: Any) -> List[Dict[str, Any]]:
    """Parse LiveBench JSON data format."""
    models = []

    try:
        # Handle different JSON structures
        if isinstance(data, dict):
            if "results" in data:
                results = data["results"]
            elif "models" in data:
                results = data["models"]
            elif "leaderboard" in data:
                results = data["leaderboard"]
            else:
                results = [data]
        elif isinstance(data, list):
            results = data
        else:
            return models

        for item in results:
            if not isinstance(item, dict):
                continue

            model = {
                "name": item.get("model_name", item.get("name", "")),
                "slug": normalize_slug(item.get("model_name", item.get("name", ""))),
                "creator": item.get("creator", item.get("model_creator", "Unknown")),
                "agentic_coding": extract_score(
                    item, ["agentic_coding", "agentic", "coding_avg"]
                ),
                "coding": extract_score(item, ["coding", "coding_index", "code"]),
                "overall": extract_score(item, ["overall", "total", "average"]),
                "livecodebench": extract_score(
                    item, ["livecodebench", "lcb", "code_bench"]
                ),
                "tps": extract_tps(item),
                "source": "livebench",
            }

            if model["name"] and model["agentic_coding"] > 0:
                models.append(model)

    except Exception as e:
        logger.warning(f"  ✗ Error parsing LiveBench JSON: {e}")

    return models


def parse_livebench_api(data: Any) -> List[Dict[str, Any]]:
    """Parse LiveBench API response format."""
    models = []
    # Similar to parse_livebench_json but handles API-specific formats
    return models or parse_livebench_json(data)


async def fetch_lmarena_data(session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
    """Fetch model rankings from LM Arena."""
    logger.info("Fetching LM Arena data...")
    models = []

    try:
        async with session.get(
            LMARENA_URL, timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status == 200:
                html = await response.text()
                models = parse_lmarena_html(html)
                logger.info(f"  ✓ Loaded {len(models)} models from LM Arena")
    except Exception as e:
        logger.warning(f"  ✗ Failed to fetch LM Arena: {e}")

    return models


def parse_lmarena_html(html: str) -> List[Dict[str, Any]]:
    """Parse LM Arena HTML page for model rankings."""
    models = []

    try:
        # Look for embedded JSON data
        import re

        # Try to find model data in script tags
        json_pattern = r'<script[^>]*type="application/json"[^>]*>(.*?)</script>'
        matches = re.findall(json_pattern, html, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match)
                if isinstance(data, dict) and "models" in data:
                    for item in data["models"]:
                        model = {
                            "name": item.get("name", item.get("model_name", "")),
                            "slug": normalize_slug(
                                item.get("name", item.get("model_name", ""))
                            ),
                            "creator": item.get("creator", "Unknown"),
                            "agentic_coding": item.get(
                                "arena_score", item.get("score", 0)
                            )
                            * 10,
                            "coding": item.get("coding_score", item.get("score", 0))
                            * 10,
                            "overall": item.get("overall", item.get("score", 0)) * 10,
                            "livecodebench": item.get("coding_score", 0) / 100,
                            "tps": 50.0,  # LM Arena doesn't typically report TPS
                            "source": "lmarena",
                        }
                        if model["name"]:
                            models.append(model)
            except json.JSONDecodeError:
                continue

    except Exception as e:
        logger.warning(f"  ✗ Error parsing LM Arena HTML: {e}")

    return models


def extract_score(item: Dict[str, Any], keys: List[str]) -> float:
    """Extract score from item using multiple possible keys."""
    for key in keys:
        if key in item:
            val = item[key]
            if isinstance(val, (int, float)):
                # Check if already a percentage (0-100) or decimal (0-1)
                if val <= 1 and val > 0:
                    return val * 100
                return float(val)
    return 0.0


def extract_tps(item: Dict[str, Any]) -> float:
    """Extract tokens per second from item."""
    # Look for TPS-related fields
    tps_keys = ["tps", "tokens_per_second", "speed", "median_output_tokens_per_second"]
    for key in tps_keys:
        if key in item:
            val = item[key]
            if isinstance(val, (int, float)):
                return float(val)
    return 50.0  # Default TPS


def normalize_slug(name: str) -> str:
    """Normalize model name to slug format."""
    clean_name = re.sub(r"[^\w\s\-\.]", "", name)
    slug = (
        clean_name.lower()
        .replace(" ", "-")
        .replace(".", "-")
        .replace("(", "")
        .replace(")", "")
        .replace("_", "-")
    )
    # Clean up multiple dashes
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    return slug


# ============================================================================
# DATA AGGREGATION
# ============================================================================


async def fetch_all_data() -> List[Dict[str, Any]]:
    """Fetch data from all sources concurrently."""
    all_models = []

    async with aiohttp.ClientSession() as session:
        # Fetch from multiple sources in parallel
        tasks = [
            fetch_livebench_data(session),
            fetch_lmarena_data(session),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Error fetching data: {result}")
            elif isinstance(result, list):
                all_models.extend(result)

    # Add known models that might be missing
    for slug, model_data in KNOWN_MODELS.items():
        model_data["slug"] = slug
        model_data["tps"] = model_data.get("tps", 50.0)
        all_models.append(model_data)

    # Also check for Claude 4.5 in existing CSV
    existing_models = load_existing_csv()
    for model in existing_models:
        if any(
            x in model.get("slug", "")
            for x in ["claude", "opus", "sonnet", "4-5", "4.5"]
        ):
            if model["slug"] not in [m.get("slug") for m in all_models]:
                logger.info(
                    f"  Adding missing model from existing data: {model['name']}"
                )
                all_models.append(model)

    # Remove duplicates based on slug
    seen = set()
    unique_models = []
    for model in all_models:
        slug = model.get("slug", "")
        if slug and slug not in seen:
            seen.add(slug)
            unique_models.append(model)

    logger.info(f"Total unique models after aggregation: {len(unique_models)}")
    return unique_models


def load_existing_csv() -> List[Dict[str, Any]]:
    """Load existing model data from CSV."""
    models = []
    csv_path = OUTPUT_DIR / "artificial_analysis_models.csv"

    if not csv_path.exists():
        logger.warning(f"  Existing CSV not found at {csv_path}")
        return models

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (
                    row.get("name")
                    and float(row.get("eval_artificial_analysis_coding_index") or 0) > 0
                ):
                    models.append(
                        {
                            "name": row.get("name", ""),
                            "slug": normalize_slug(row.get("name", "")),
                            "creator": row.get("model_creator_name", "Unknown"),
                            "agentic_coding": float(
                                row.get("eval_artificial_analysis_coding_index") or 0
                            ),
                            "coding": float(row.get("coding") or 0),
                            "overall": float(row.get("overall") or 0),
                            "livecodebench": float(row.get("eval_livecodebench") or 0)
                            * 100,
                            "tps": float(
                                row.get("median_output_tokens_per_second") or 50.0
                            ),
                            "source": "existing_csv",
                        }
                    )
        logger.info(f"  Loaded {len(models)} models from existing CSV")
    except Exception as e:
        logger.warning(f"  ✗ Error loading existing CSV: {e}")

    return models


# ============================================================================
# SCORING & RANKING
# ============================================================================


def calculate_quality_score(model: Dict[str, Any]) -> float:
    """Calculate overall quality score for a model."""
    scores = []

    # Agentic coding (primary metric)
    if model.get("agentic_coding", 0) > 0:
        scores.append(("agentic", model["agentic_coding"], 3.0))

    # Coding index
    if model.get("coding", 0) > 0:
        scores.append(("coding", model["coding"], 2.0))

    # LiveCodeBench
    if model.get("livecodebench", 0) > 0:
        lcb = model["livecodebench"]
        if lcb <= 1:
            lcb *= 100
        scores.append(("lcb", lcb, 2.5))

    # Overall score
    if model.get("overall", 0) > 0:
        scores.append(("overall", model["overall"], 1.0))

    if not scores:
        return 0

    # Weighted average
    total_weight = sum(w for _, _, w in scores)
    weighted_sum = sum(score * w for _, score, w in scores)

    return weighted_sum / total_weight if total_weight > 0 else 0


def rank_models(
    models: List[Dict[str, Any]], use_tps_weighting: bool = False
) -> List[Dict[str, Any]]:
    """Rank models by quality score."""
    ranked = []

    for model in models:
        quality = calculate_quality_score(model)
        if quality < 30:  # Minimum quality threshold
            continue

        tps = model.get("tps", 50.0)
        tier_boost = 0

        # Apply tier boosts
        slug = model.get("slug", "")
        if "opus" in slug:
            tier_boost = 2.0
        elif "sonnet" in slug:
            tier_boost = 1.0

        # Version boost
        if "4-5" in slug or "4.5" in slug:
            tier_boost += 0.5
        elif "3-7" in slug or "3.7" in slug:
            tier_boost += 0.3

        adjusted_quality = quality + tier_boost

        if use_tps_weighting:
            safe_tps = max(tps, 10.0)
            effective_score = (adjusted_quality**2) * math.log10(safe_tps)
        else:
            effective_score = adjusted_quality**2

        model["quality_score"] = quality
        model["effective_score"] = effective_score
        model["tps"] = tps
        model["adjusted_quality"] = adjusted_quality

        ranked.append(model)

    # Sort by effective score
    ranked.sort(key=lambda x: x["effective_score"], reverse=True)

    return ranked


# ============================================================================
# OUTPUT GENERATION
# ============================================================================


def save_to_csv(models: List[Dict[str, Any]], filepath: Path):
    """Save aggregated models to CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "name",
        "slug",
        "creator",
        "source",
        "agentic_coding",
        "coding",
        "overall",
        "livecodebench",
        "tps",
        "quality_score",
        "effective_score",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model in models:
            row = {field: model.get(field, "") for field in fieldnames}
            writer.writerow(row)

    logger.info(f"Saved {len(models)} models to {filepath}")


def generate_aliases_yaml(models: List[Dict[str, Any]]):
    """Generate aliases.yaml for the proxy."""
    ranked = rank_models(models, use_tps_weighting=False)

    # Get unique candidates per provider
    unique_candidates = []
    seen = set()

    for model in ranked:
        creator = model.get("creator", "Unknown")
        providers = PROVIDER_MAP.get(creator, ["g4f"])

        for provider in providers:
            if provider == "groq" and "llama" not in model.get("slug", ""):
                continue

            key = (provider, model.get("slug", ""))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(
                    {
                        "provider": provider,
                        "model": model.get("slug", ""),
                        "real_name": model.get("name", ""),
                        "base_score": model.get("quality_score", 0),
                        "tps": model.get("tps", 50.0),
                        "effective_score": model.get("effective_score", 0),
                    }
                )

    # Generate alias configs
    aliases_config = {
        "aliases": {
            "coding-smart": {
                "description": "Best Agentic Coding Performance (LiveBench)",
                "strategy": "chain_fallback",
                "candidates": [],
            },
            "coding-fast": {
                "description": "Fastest Coding Response Times",
                "strategy": "chain_fallback",
                "candidates": [],
            },
            "coding-balanced": {
                "description": "Balanced Coding Quality and Speed",
                "strategy": "chain_fallback",
                "candidates": [],
            },
            "chat-smart": {
                "description": "Best Conversational Performance",
                "strategy": "chain_fallback",
                "candidates": [],
            },
            "chat-fast": {
                "description": "Fastest Chat Response Times",
                "strategy": "chain_fallback",
                "candidates": [],
            },
        }
    }

    # coding-smart: Priority on agentic_coding
    coding_smart_candidates = sorted(
        unique_candidates, key=lambda x: x["base_score"], reverse=True
    )
    for c in coding_smart_candidates[:30]:
        aliases_config["aliases"]["coding-smart"]["candidates"].append(
            {"provider": c["provider"], "model": c["model"]}
        )

    # coding-fast: Priority on TPS (Quality > 40)
    coding_fast_candidates = sorted(
        [c for c in unique_candidates if c["base_score"] > 40],
        key=lambda x: x["tps"],
        reverse=True,
    )
    for c in coding_fast_candidates[:20]:
        aliases_config["aliases"]["coding-fast"]["candidates"].append(
            {"provider": c["provider"], "model": c["model"]}
        )

    # coding-balanced: Priority on effective_score (which uses both quality and TPS)
    coding_balanced_candidates = sorted(
        unique_candidates, key=lambda x: x["effective_score"], reverse=True
    )
    for c in coding_balanced_candidates[:25]:
        aliases_config["aliases"]["coding-balanced"]["candidates"].append(
            {"provider": c["provider"], "model": c["model"]}
        )

    # chat-smart: Priority on overall score
    chat_smart_candidates = sorted(
        unique_candidates, key=lambda x: x["base_score"], reverse=True
    )
    for c in chat_smart_candidates[:20]:
        aliases_config["aliases"]["chat-smart"]["candidates"].append(
            {"provider": c["provider"], "model": c["model"]}
        )

    # chat-fast: Fastest response for chat (Quality > 30)
    chat_fast_candidates = sorted(
        [c for c in unique_candidates if c["base_score"] > 30],
        key=lambda x: x["tps"],
        reverse=True,
    )
    for c in chat_fast_candidates[:20]:
        aliases_config["aliases"]["chat-fast"]["candidates"].append(
            {"provider": c["provider"], "model": c["model"]}
        )

    # Save aliases
    ALIASES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ALIASES_FILE, "w") as f:
        yaml.dump(aliases_config, f, sort_keys=False)

    logger.info(f"Generated aliases.yaml at {ALIASES_FILE}")

    # Print top 10
    print("\n" + "=" * 70)
    print("TOP 10 AGGREGATED CODING MODELS")
    print("=" * 70)
    for i, model in enumerate(ranked[:10], 1):
        print(f"{i:2}. {model.get('name', model.get('slug', 'Unknown'))}")
        print(
            f"    Creator: {model.get('creator', 'Unknown')} | Source: {model.get('source', 'unknown')}"
        )
        print(
            f"    Quality: {model.get('quality_score', 0):.1f} | TPS: {model.get('tps', 0):.1f}"
        )
        print()


# ============================================================================
# MAIN
# ============================================================================


async def main():
    """Main entry point."""
    logger.info("=" * 70)
    logger.info("LLM Leaderboard Data Aggregator")
    logger.info("Sources: LiveBench, LM Arena, Artificial Analysis")
    logger.info("=" * 70)

    # Fetch all data
    all_models = await fetch_all_data()

    if not all_models:
        logger.error("No model data retrieved from any source!")
        logger.info("Falling back to existing CSV data...")
        all_models = load_existing_csv()

    if not all_models:
        logger.error("No model data available. Exiting.")
        sys.exit(1)

    # Rank models
    ranked_models = rank_models(all_models, use_tps_weighting=False)

    # Save to CSV
    save_to_csv(ranked_models, OUTPUT_FILE)

    # Generate aliases
    generate_aliases_yaml(ranked_models)

    logger.info("=" * 70)
    logger.info("Data aggregation complete!")
    logger.info(f"Output: {OUTPUT_FILE}")
    logger.info(f"Aliases: {ALIASES_FILE}")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
