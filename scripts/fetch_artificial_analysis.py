#!/usr/bin/env python3
"""Fetch model data from Artificial Analysis API.

Outputs CSV matching the format expected by aggregate_leaderboard_data.py.

Usage:
    export AA_API_KEY="your_key_here"  # Get free at https://artificialanalysis.ai/insights
    python scripts/fetch_artificial_analysis.py
    python scripts/fetch_artificial_analysis.py --output data/aa_models.csv

Requires: pip install httpx
Attribution: https://artificialanalysis.ai/
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fetch-aa")

API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent.parent
    / "llm-leaderboard"
    / "artificial_analysis_models.csv"
)


def fetch_models(api_key: str) -> Optional[List[Dict[str, Any]]]:
    headers = {"x-api-key": api_key, "Accept": "application/json"}
    try:
        resp = httpx.get(API_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        logger.info(f"Fetched {len(models)} models from AA API")
        return models
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            logger.error(
                "Invalid API key. Get one free at https://artificialanalysis.ai/insights"
            )
        elif e.response.status_code == 429:
            logger.error("Rate limited (1,000 req/day). Cache responses or wait.")
        else:
            logger.error(f"HTTP {e.response.status_code}: {e}")
    except Exception as e:
        logger.error(f"Failed to fetch: {e}")
    return None


def convert_to_csv_rows(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for m in models:
        evals = m.get("evaluations") or {}
        pricing = m.get("pricing") or {}
        creator = m.get("model_creator") or {}
        rows.append(
            {
                "name": m.get("name", ""),
                "slug": m.get("slug", ""),
                "model_id": m.get("id", ""),
                "model_creator_name": creator.get("name", "Unknown"),
                "model_creator_id": creator.get("id", ""),
                "eval_artificial_analysis_intelligence_index": evals.get(
                    "artificial_analysis_intelligence_index"
                ),
                "eval_artificial_analysis_coding_index": evals.get(
                    "artificial_analysis_coding_index"
                ),
                "eval_artificial_analysis_math_index": evals.get(
                    "artificial_analysis_math_index"
                ),
                "eval_mmlu_pro": evals.get("mmlu_pro"),
                "eval_gpqa": evals.get("gpqa"),
                "eval_hle": evals.get("hle"),
                "eval_livecodebench": evals.get("livecodebench"),
                "eval_scicode": evals.get("scicode"),
                "eval_math_500": evals.get("math_500"),
                "eval_aime": evals.get("aime"),
                "price_1m_input_tokens": pricing.get("price_1m_input_tokens"),
                "price_1m_output_tokens": pricing.get("price_1m_output_tokens"),
                "price_1m_blended": pricing.get("price_1m_blended_3_to_1"),
                "median_output_tokens_per_second": m.get(
                    "median_output_tokens_per_second"
                ),
                "median_time_to_first_token_seconds": m.get(
                    "median_time_to_first_token_seconds"
                ),
                "fetched_at": datetime.utcnow().isoformat(),
            }
        )
    logger.info(f"Converted {len(rows)} models to CSV format")
    return rows


def save_csv(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        logger.warning("No data to save")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch Artificial Analysis model data")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path"
    )
    parser.add_argument("--json", action="store_true", help="Also output raw JSON")
    args = parser.parse_args()

    api_key = os.environ.get("AA_API_KEY")
    if not api_key:
        logger.error(
            "AA_API_KEY not set. Get one free at https://artificialanalysis.ai/insights"
        )
        logger.error("  export AA_API_KEY='your_key_here'")
        sys.exit(1)

    models = fetch_models(api_key)
    if not models:
        sys.exit(1)

    rows = convert_to_csv_rows(models)
    save_csv(rows, args.output)

    if args.json:
        json_path = args.output.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)
        logger.info(f"Saved raw JSON to {json_path}")

    if rows:
        print(
            f"\nDone. {len(rows)} models from Artificial Analysis saved to {args.output}"
        )
        print("Attribution: https://artificialanalysis.ai/")


if __name__ == "__main__":
    main()
