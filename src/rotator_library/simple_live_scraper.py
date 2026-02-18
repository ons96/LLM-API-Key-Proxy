"""Simple Live Scraper

Tries multiple approaches to get live data:
1. Direct API endpoints
2. Page scraping with requests/BeautifulSoup
3. JSON extraction from page
"""

import requests
import json
import re
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SimpleLiveScraper:
    """Simple scraper that doesn't require browser."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

    def scrape_artificial_analysis_page(self) -> Optional[List[Dict]]:
        """Scrape the Artificial Analysis leaderboard page directly."""
        url = "https://artificialanalysis.ai/leaderboards/providers?deprecation=all"

        try:
            logger.info(f"Fetching {url}...")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            html = response.text

            # Try to find JSON data embedded in page
            # Look for __NEXT_DATA__ or similar
            patterns = [
                r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                r"window\.__DATA__\s*=\s*(\{.*?\});",
                r'<script[^>]*>.*?(\{[^}]*"models"[^}]*\}).*?</script>',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, html, re.DOTALL)
                for match in matches:
                    try:
                        data = json.loads(match)
                        if "models" in data or "props" in data:
                            logger.info("✓ Found embedded JSON data")
                            return self._extract_models_from_json(data)
                    except json.JSONDecodeError:
                        continue

            logger.warning("Could not find embedded JSON in page")
            return None

        except Exception as e:
            logger.error(f"Failed to scrape page: {e}")
            return None

    def _extract_models_from_json(self, data: Dict) -> List[Dict]:
        """Extract model list from JSON structure."""
        # Try different paths to find models
        paths = [
            data.get("models", []),
            data.get("props", {}).get("pageProps", {}).get("models", []),
            data.get("pageProps", {}).get("models", []),
        ]

        for models in paths:
            if models:
                return models

        return []

    def fetch_fresh_data(self) -> Optional[List[Dict]]:
        """Fetch fresh data using available methods."""
        logger.info("Attempting to fetch LIVE data...")

        # Try page scraping
        models = self.scrape_artificial_analysis_page()
        if models:
            return models

        logger.error("All live fetch methods failed")
        return None


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    scraper = SimpleLiveScraper()
    models = scraper.fetch_fresh_data()

    if models:
        print(f"\n✓ Successfully fetched {len(models)} models")
        print("\nFirst 5 models:")
        for i, m in enumerate(models[:5], 1):
            print(f"{i}. {m.get('name', 'Unknown')}")
    else:
        print("\n✗ Failed to fetch live data")
        sys.exit(1)
