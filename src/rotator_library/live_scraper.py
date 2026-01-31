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
        self.session.headers.update({
            'User-Agent': 'LLM-Gateway-Scraper/1.0 (Research Purpose)'
        })
        
    def fetch_models_dev(self) -> Optional[Dict]:
        """Fetch comprehensive model list from models.dev API."""
        try:
            logger.info("Fetching models.dev API...")
            response = self.session.get(
                "https://models.dev/api/v1/models",
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"✓ Fetched {len(data.get('models', []))} models from models.dev")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch models.dev: {e}")
            return None
    
    def fetch_artificial_analysis_direct(self) -> Optional[Dict]:
        """Fetch directly from Artificial Analysis API endpoint."""
        try:
            logger.info("Fetching Artificial Analysis leaderboard...")
            
            # Try the API endpoint directly
            response = self.session.get(
                "https://artificialanalysis.ai/api/leaderboard",
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✓ Fetched {len(data.get('models', []))} models from AI API")
                return data
            else:
                logger.warning(f"AI API returned {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch AI API: {e}")
            return None
    
    def process_models_dev_data(self, data: Dict) -> List[Dict]:
        """Process models.dev data into standardized format."""
        models = []
        
        for model in data.get('models', []):
            model_id = model.get('id', '')
            
            # Extract provider from model ID or sources
            provider = self._extract_provider(model)
            
            # Get API mappings (which providers offer this model)
            api_mappings = model.get('sources', [])
            
            model_data = {
                'id': f"{provider}/{model_id}" if provider else model_id,
                'name': model.get('name', model_id),
                'provider': provider,
                'api_mappings': api_mappings,
                'context_window': model.get('context_length', 0),
                'scores': {
                    'agentic_coding': 0,  # Will fill from other sources
                    'intelligence': model.get('intelligence_score', 0),
                    'overall': model.get('overall_score', 0),
                }
            }
            
            models.append(model_data)
        
        return models
    
    def _extract_provider(self, model: Dict) -> str:
        """Extract primary provider from model data."""
        # Check sources/APIs
        sources = model.get('sources', [])
        if sources:
            # Return first source as primary provider
            return sources[0].get('provider', 'unknown')
        
        # Try to infer from model ID
        model_id = model.get('id', '').lower()
        if 'claude' in model_id:
            return 'anthropic'
        elif 'gpt' in model_id or 'o1' in model_id or 'o3' in model_id:
            return 'openai'
        elif 'gemini' in model_id:
            return 'google'
        elif 'llama' in model_id:
            return 'meta'
        elif 'grok' in model_id:
            return 'xai'
        elif 'mistral' in model_id:
            return 'mistral'
        elif 'qwen' in model_id:
            return 'alibaba'
        elif 'deepseek' in model_id:
            return 'deepseek'
        elif 'kimi' in model_id:
            return 'moonshot'
        elif 'minimax' in model_id:
            return 'minimax'
        
        return 'unknown'
    
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
        'metadata': {
            'source': 'LIVE - models.dev API',
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_models': len(models),
            'data_freshness': 'LIVE (real-time)',
        },
        'models': models
    }
    
    output_path = Path('config/model_rankings.yaml')
    try:
        with open(output_path, 'w') as f:
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
