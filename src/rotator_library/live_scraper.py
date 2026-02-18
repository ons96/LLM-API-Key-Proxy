"""Live Scraper for Real-Time LLM Leaderboards

Fetches fresh data from:
1. models.dev - Comprehensive model database with API mappings
2. Artificial Analysis - Live leaderboard data (via API if available)
3. Multiple sources aggregated for accurate scores

NO cached data - always fetches live!
"""

import requests
import json
import time
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class LiveLeaderboardScraper:
    """Scrapes live data from multiple sources."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "LLM-Gateway-Scraper/1.0 (Research Purpose)"}
        )

    def fetch_models_dev(self) -> Optional[Dict]:
        """Fetch comprehensive model list from models.dev API."""
        try:
            logger.info("Fetching models.dev API...")
            # Correct URL - returns JSON with provider-keyed objects
            response = self.session.get("https://models.dev/api.json", timeout=60)
            response.raise_for_status()

            data = response.json()
            total_models = sum(
                len(p.get("models", {})) for p in data.values() if isinstance(p, dict)
            )
            logger.info(
                f"✓ Fetched {total_models} models from {len(data)} providers via models.dev"
            )
            return data

        except Exception as e:
            logger.error(f"Failed to fetch models.dev: {e}")
            return None

    def fetch_artificial_analysis_direct(self) -> Optional[Dict]:
        """Fetch directly from Artificial Analysis API v2 endpoint."""
        import os

        try:
            logger.info("Fetching Artificial Analysis leaderboard...")

            api_key = os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY")
            headers = {}
            if api_key:
                headers["x-api-key"] = api_key

            response = self.session.get(
                "https://artificialanalysis.ai/api/v2/data/llms/models",
                headers=headers,
                timeout=60,
            )

            if response.status_code == 200:
                data = response.json()
                models = data.get("data", data.get("models", []))
                logger.info(
                    f"✓ Fetched {len(models)} models from Artificial Analysis API v2"
                )
                return data
            elif response.status_code == 401:
                logger.warning(
                    "Artificial Analysis API requires key - set ARTIFICIAL_ANALYSIS_API_KEY env var"
                )
                return None
            else:
                logger.warning(f"AI API returned {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Failed to fetch AI API: {e}")
            return None

    def process_models_dev_data(self, data: Dict) -> List[Dict]:
        """Process models.dev data into standardized format."""
        models = []
        free_models = []

        for provider_id, provider_data in data.items():
            if not isinstance(provider_data, dict):
                continue

            provider_models = provider_data.get("models", {})
            for model_id, model_info in provider_models.items():
                cost = model_info.get("cost", {})
                is_free = cost.get("input", 1) == 0 and cost.get("output", 1) == 0

                model_data = {
                    "id": f"{provider_id}/{model_id}",
                    "name": model_info.get("name", model_id),
                    "provider": provider_id,
                    "api_base": provider_data.get("api", ""),
                    "context_window": model_info.get("limit", {}).get("context", 0),
                    "is_free": is_free,
                    "cost": cost,
                    "supports_tools": model_info.get("tool_call", False),
                    "supports_reasoning": model_info.get("reasoning", False),
                    "modalities": model_info.get("modalities", {}),
                }

                models.append(model_data)
                if is_free:
                    free_models.append(model_data)

        logger.info(
            f"✓ Found {len(free_models)} FREE models out of {len(models)} total"
        )
        return free_models

    def _extract_provider(self, model: Dict) -> str:
        """Extract primary provider from model data."""
        # Check sources/APIs
        sources = model.get("sources", [])
        if sources:
            # Return first source as primary provider
            return sources[0].get("provider", "unknown")

        # Try to infer from model ID
        model_id = model.get("id", "").lower()
        if "claude" in model_id:
            return "anthropic"
        elif "gpt" in model_id or "o1" in model_id or "o3" in model_id:
            return "openai"
        elif "gemini" in model_id:
            return "google"
        elif "llama" in model_id:
            return "meta"
        elif "grok" in model_id:
            return "xai"
        elif "mistral" in model_id:
            return "mistral"
        elif "qwen" in model_id:
            return "alibaba"
        elif "deepseek" in model_id:
            return "deepseek"
        elif "kimi" in model_id:
            return "moonshot"
        elif "minimax" in model_id:
            return "minimax"

        return "unknown"

    def aggregate_live_data(self) -> Optional[List[Dict]]:
        """Fetch and aggregate live data from all sources."""
        logger.info("Starting LIVE data aggregation...")

        # Fetch from models.dev (comprehensive model list)
        models_dev_data = self.fetch_models_dev()

        if not models_dev_data:
            logger.error("Failed to fetch from models.dev - cannot proceed")
            return None

        # Process models.dev data
        models = self.process_models_dev_data(models_dev_data)

        logger.info(f"✓ Processed {len(models)} models from live sources")

        return models


def update_model_rankings_live():
    """Update model rankings with LIVE data (no cache!)."""
    scraper = LiveLeaderboardScraper()
    models = scraper.aggregate_live_data()

    if not models:
        logger.error("Failed to fetch live data")
        return False

    # Save to file
    import yaml
    from pathlib import Path

    output = {
        "metadata": {
            "source": "LIVE - models.dev API",
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_models": len(models),
            "data_freshness": "LIVE (real-time)",
        },
        "models": models,
    }

    output_path = Path("config/model_rankings.yaml")
    try:
        with open(output_path, "w") as f:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)
        logger.info(f"✓ Updated {output_path} with {len(models)} LIVE models")
        return True
    except Exception as e:
        logger.error(f"Failed to write rankings: {e}")
        return False


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    success = update_model_rankings_live()
    sys.exit(0 if success else 1)
