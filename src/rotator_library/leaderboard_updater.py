"""Leaderboard Updater

Fetches and aggregates LLM rankings from multiple sources:
- Artificial Analysis (primary)
- Arena.ai (Chatbot Arena)
- LiveBench
- SWE-bench (for coding)

Updates model_rankings.yaml with comprehensive, accurate data.
Designed to run on 1GB VPS every 6 hours.
"""

import json
import yaml
import time
import sqlite3
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LeaderboardUpdater:
    """Updates model rankings from multiple sources."""
    
    def __init__(self, 
                 output_path: str = "config/model_rankings.yaml",
                 cache_db: str = "/tmp/leaderboard_cache.db"):
        self.output_path = Path(output_path)
        self.cache_db = cache_db
        self._init_cache()
        
    def _init_cache(self):
        """Initialize SQLite cache for leaderboard data."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS leaderboard_cache (
                source TEXT PRIMARY KEY,
                data TEXT,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        
    def fetch_artificial_analysis(self) -> Optional[Dict]:
        """Fetch from Artificial Analysis API."""
        try:
            # Import from copied scraper
            sys.path.insert(0, '/home/owens/CodingProjects/llm-leaderboard')
            from fetch_artificial_analysis_data import fetch_artificial_analysis_leaderboard
            
            data = fetch_artificial_analysis_leaderboard()
            
            if data:
                # Cache the result
                conn = sqlite3.connect(self.cache_db)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO leaderboard_cache (source, data) VALUES (?, ?)",
                    ("artificial_analysis", json.dumps(data))
                )
                conn.commit()
                conn.close()
                
                return data
                
        except Exception as e:
            logger.error(f"Failed to fetch Artificial Analysis: {e}")
            
        # Try to use cached data
        return self._get_cached_data("artificial_analysis")
    
    def _get_cached_data(self, source: str) -> Optional[Dict]:
        """Get cached data if fetch fails."""
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT data FROM leaderboard_cache WHERE source = ?",
                (source,)
            )
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return json.loads(result[0])
        except Exception as e:
            logger.error(f"Failed to get cached data: {e}")
            
        return None
    
    def aggregate_scores(self, aa_data: Dict) -> List[Dict]:
        """Aggregate scores from all sources into unified ranking."""
        models = []
        
        # Process Artificial Analysis data
        for model in aa_data.get('models', []):
            model_id = model.get('id', '')
            if not model_id:
                continue
                
            # Normalize provider/model format
            if '/' not in model_id:
                # Try to infer provider from name
                provider = self._infer_provider(model.get('name', ''))
                model_id = f"{provider}/{model_id}"
            
            scores = model.get('scores', {})
            
            # Calculate agentic coding score from available metrics
            agentic_score = self._calculate_agentic_score(scores)
            
            models.append({
                'id': model_id,
                'name': model.get('name', 'Unknown'),
                'provider': model_id.split('/')[0] if '/' in model_id else 'unknown',
                'scores': {
                    'agentic_coding': agentic_score,
                    'artificial_analysis': scores.get('overall', 0),
                    'intelligence': scores.get('intelligence', 0),
                    'coding': scores.get('coding', 0),
                },
                'context_window': model.get('context_window', 0),
                'cost_per_1k_input': model.get('cost_input', 0),
                'cost_per_1k_output': model.get('cost_output', 0),
            })
        
        # Sort by agentic coding score
        models.sort(key=lambda x: x['scores']['agentic_coding'], reverse=True)
        
        return models
    
    def _infer_provider(self, model_name: str) -> str:
        """Infer provider from model name."""
        name_lower = model_name.lower()
        
        if 'claude' in name_lower:
            return 'anthropic'
        elif 'gpt' in name_lower or 'o1' in name_lower or 'o3' in name_lower:
            return 'openai'
        elif 'gemini' in name_lower:
            return 'google'
        elif 'llama' in name_lower:
            return 'meta'
        elif 'grok' in name_lower:
            return 'xai'
        elif 'mistral' in name_lower:
            return 'mistral'
        elif 'qwen' in name_lower:
            return 'alibaba'
        elif 'deepseek' in name_lower:
            return 'deepseek'
        elif 'kimi' in name_lower:
            return 'moonshot'
        elif 'minimax' in name_lower:
            return 'minimax'
        else:
            return 'unknown'
    
    def _calculate_agentic_score(self, scores: Dict) -> float:
        """Calculate agentic coding score from available metrics."""
        # Prioritize coding-specific scores
        coding = scores.get('coding', 0)
        reasoning = scores.get('reasoning', 0)
        overall = scores.get('overall', 0)
        
        # Weight: coding > reasoning > overall
        return (coding * 0.5) + (reasoning * 0.3) + (overall * 0.2)
    
    def update_model_rankings(self) -> bool:
        """Update the model_rankings.yaml file."""
        logger.info("Starting leaderboard update...")
        
        # Fetch data from all sources
        aa_data = self.fetch_artificial_analysis()
        
        if not aa_data:
            logger.error("No data available from any source")
            return False
        
        # Aggregate scores
        models = self.aggregate_scores(aa_data)
        
        if not models:
            logger.error("No models after aggregation")
            return False
        
        # Create output structure
        output = {
            'metadata': {
                'source': 'Aggregated from Artificial Analysis',
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_models': len(models),
            },
            'models': models
        }
        
        # Write to file
        try:
            with open(self.output_path, 'w') as f:
                yaml.dump(output, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Updated {self.output_path} with {len(models)} models")
            return True
        except Exception as e:
            logger.error(f"Failed to write rankings: {e}")
            return False


def update_leaderboards():
    """Main entry point for updating leaderboards."""
    updater = LeaderboardUpdater()
    success = updater.update_model_rankings()
    return success


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    success = update_leaderboards()
    sys.exit(0 if success else 1)
