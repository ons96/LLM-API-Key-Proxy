"""
UGI Leaderboard Scraper

Fetches uncensored model rankings from the UGI Leaderboard on HuggingFace.
UGI = Uncensored General Intelligence - measures how uncensored/willing a model is.
"""

import requests
import csv
import re
import logging
from typing import List, Dict, Tuple, Optional
from io import StringIO

logger = logging.getLogger(__name__)

UGI_CSV_URL = "https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard/resolve/main/ugi-leaderboard-data.csv"


def normalize_model_name(name: str) -> str:
    """Normalize model name to a common format."""
    if not name:
        return ""
    parts = name.split("/")
    if len(parts) == 2:
        author, model = parts
        return model.lower().replace("-", "-")
    return name.lower().replace("/", "-").replace("_", "-")


def scrape_ugi_leaderboard(
    min_ugi: float = 20.0, min_writing: float = 0.0
) -> List[Dict]:
    """
    Scrape UGI leaderboard for uncensored model rankings.

    Returns list of dicts with:
        - model: normalized model name
        - full_name: original author/model name
        - ugi_score: uncensored score (higher = more uncensored)
        - writing: writing quality score
        - willingness: W/10 score
        - natint: natural intelligence score
        - params: active parameters (in billions)
        - nsfw_score: NSFW capability score
    """
    try:
        logger.info(f"Fetching UGI leaderboard from {UGI_CSV_URL}")
        response = requests.get(UGI_CSV_URL, timeout=60)
        response.raise_for_status()

        content = response.text
        reader = csv.DictReader(StringIO(content))

        models = []
        for row in reader:
            try:
                full_name = row.get("author/model_name", "")
                if not full_name:
                    continue

                ugi_score = float(row.get("UGI 🏆", 0) or 0)
                writing = float(row.get("Writing ✍️", 0) or 0)
                willingness = float(row.get("W/10 👍", 0) or 0)
                natint = float(row.get("NatInt 💡", 0) or 0)
                params = float(row.get("Active Parameters", 0) or 0)
                nsfw_score = float(row.get("avg_nsfw_score", 0) or 0)

                if ugi_score < min_ugi:
                    continue
                if writing < min_writing:
                    continue

                model = normalize_model_name(full_name)

                models.append(
                    {
                        "model": model,
                        "full_name": full_name,
                        "ugi_score": ugi_score,
                        "writing": writing,
                        "willingness": willingness,
                        "natint": natint,
                        "params": params,
                        "nsfw_score": nsfw_score,
                    }
                )

            except (ValueError, TypeError) as e:
                continue

        models.sort(key=lambda x: x["ugi_score"], reverse=True)

        logger.info(f"✓ Found {len(models)} uncensored models (UGI >= {min_ugi})")
        return models

    except Exception as e:
        logger.error(f"Failed to scrape UGI leaderboard: {e}")
        return []


def get_top_rp_models(limit: int = 30, min_ugi: float = 25.0) -> List[Dict]:
    """
    Get top models for RP/uncensored use cases.

    Filters for:
        - High UGI score (uncensored)
        - Good writing quality
        - Reasonable intelligence
    """
    models = scrape_ugi_leaderboard(min_ugi=min_ugi)

    scored = []
    for m in models:
        score = (
            m["ugi_score"] * 0.5
            + m["writing"] * 0.25
            + m["natint"] * 0.15
            + m["willingness"] * 5 * 0.1
        )
        scored.append({**m, "rp_score": score})

    scored.sort(key=lambda x: x["rp_score"], reverse=True)
    return scored[:limit]


def map_to_providers(models: List[Dict], provider: str = "g4f") -> List[Dict]:
    """
    Map UGI models to provider-specific model IDs.

    Most UGI models are open-source and available via:
        - g4f (primary for RP models)
        - together
        - deepinfra
        - local
    """
    mapped = []
    for m in models:
        mapped.append(
            {
                "provider": provider,
                "model": m["model"],
                "full_name": m["full_name"],
                "ugi_score": m["ugi_score"],
                "writing": m["writing"],
            }
        )
    return mapped


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    models = get_top_rp_models(limit=50)

    print("\n" + "=" * 70)
    print("TOP 20 UNCOVERSED MODELS FOR RP")
    print("=" * 70)

    for i, m in enumerate(models[:20], 1):
        print(f"{i:2}. {m['full_name']}")
        print(
            f"    UGI: {m['ugi_score']:.1f} | Writing: {m['writing']:.1f} | Intel: {m['natint']:.1f}"
        )
        print()
