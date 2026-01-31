"""Dynamic Model Scoring Engine

Calculates real-time scores for models based on agentic coding performance,
tokens per second, and availability with configurable weights and thresholds.
"""

import time
import sqlite3
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelScore:
    """Represents a calculated score for a model/provider combination."""
    provider: str
    model: str
    agentic_score: float
    tps: float
    availability: float
    total_score: float
    meets_threshold: bool


class DynamicScoringEngine:
    """Calculates dynamic scores for virtual model fallback ordering.
    
    Coding Models: Agentic(80%) + TPS(15%) + Availability(5%)
    Chat Models: Intelligence(70%) + TPS(20%) + Availability(10%)
    """
    
    CODING_WEIGHTS = {
        'agentic': 0.80,
        'tps': 0.15,
        'availability': 0.05
    }
    
    CHAT_WEIGHTS = {
        'intelligence': 0.70,
        'tps': 0.20,
        'availability': 0.10
    }
    
    THRESHOLDS = {
        'coding-elite': 70.0,
        'coding-smart': 65.0,
        'coding-fast': 0.0,
        'chat-smart': 0.0,
        'chat-fast': 0.0
    }
    
    def __init__(self, telemetry_manager=None, model_rankings_path: str = "config/model_rankings.yaml"):
        self.telemetry = telemetry_manager
        self.model_rankings_path = Path(model_rankings_path)
        self._rankings_cache: Dict[str, Dict] = {}
        self._last_refresh = 0
        
    def load_model_rankings(self) -> Dict[str, Dict]:
        """Load model rankings from YAML file."""
        import yaml
        
        try:
            with open(self.model_rankings_path) as f:
                data = yaml.safe_load(f)
                
            rankings = {}
            for model in data.get('models', []):
                model_id = model.get('id', '')
                if '/' in model_id:
                    provider, model_name = model_id.split('/', 1)
                    key = f"{provider}/{model_name}"
                    rankings[key] = model
                    
            self._rankings_cache = rankings
            self._last_refresh = time.time()
            return rankings
            
        except Exception as e:
            logger.error(f"Failed to load model rankings: {e}")
            return {}
    
    def get_model_score(self, provider: str, model: str) -> Optional[float]:
        """Get agentic coding score for a model."""
        if time.time() - self._last_refresh > 3600:
            self.load_model_rankings()
            
        key = f"{provider}/{model}"
        model_data = self._rankings_cache.get(key)
        
        if not model_data:
            for k, v in self._rankings_cache.items():
                if k.endswith(f"/{model}") or v.get('name', '').lower() == model.lower():
                    model_data = v
                    break
        
        if model_data:
            scores = model_data.get('scores', {})
            return scores.get('agentic_coding', scores.get('swe_bench', 0.0))
            
        return None
    
    def get_tps_estimate(self, provider: str, model: str) -> float:
        """Get TPS from telemetry or return default estimate."""
        if not self.telemetry:
            return 100.0
            
        try:
            conn = sqlite3.connect(self.telemetry.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT AVG(tps) FROM tps_metrics 
                WHERE provider = ? AND model = ? 
                AND timestamp > datetime('now', '-1 hour')
            """, (provider, model))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                return float(result[0])
                
        except Exception as e:
            logger.debug(f"Failed to get TPS from telemetry: {e}")
            
        return 100.0
    
    def get_availability(self, provider: str, model: str) -> float:
        """Get availability score (0-1) based on recent health."""
        if not self.telemetry:
            return 1.0
            
        try:
            conn = sqlite3.connect(self.telemetry.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(CASE WHEN error_message IS NULL THEN 1 END) as success,
                    COUNT(*) as total
                FROM telemetry
                WHERE provider = ? AND model = ?
                AND timestamp > datetime('now', '-30 minutes')
            """, (provider, model))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[1] > 0:
                return result[0] / result[1]
                
        except Exception as e:
            logger.debug(f"Failed to get availability: {e}")
            
        return 1.0
    
    def calculate_coding_score(self, provider: str, model: str,
                                virtual_model_type: str = "coding-elite") -> ModelScore:
        """Calculate score for coding models with 80/15/5 weights."""
        agentic_score = self.get_model_score(provider, model) or 0.0
        tps = self.get_tps_estimate(provider, model)
        availability = self.get_availability(provider, model)
        
        tps_normalized = min(tps / 20.0, 100.0)
        
        weights = self.CODING_WEIGHTS
        total_score = (
            (agentic_score * weights['agentic']) +
            (tps_normalized * weights['tps']) +
            (availability * 100 * weights['availability'])
        )
        
        threshold = self.THRESHOLDS.get(virtual_model_type, 0.0)
        meets_threshold = agentic_score >= threshold
        
        return ModelScore(
            provider=provider,
            model=model,
            agentic_score=agentic_score,
            tps=tps,
            availability=availability,
            total_score=total_score,
            meets_threshold=meets_threshold
        )
    
    def rank_models_for_virtual(self, virtual_model_type: str, 
                                 candidates: List[tuple]) -> List[ModelScore]:
        """Rank candidate models by score for a virtual model.
        
        Args:
            virtual_model_type: e.g., 'coding-elite', 'coding-smart'
            candidates: List of (provider, model) tuples
            
        Returns:
            List of ModelScore objects sorted by total_score descending
        """
        scores = []
        
        for provider, model in candidates:
            score = self.calculate_coding_score(provider, model, virtual_model_type)
            scores.append(score)
        
        # Sort by total_score descending
        scores.sort(key=lambda s: s.total_score, reverse=True)
        
        return scores


def get_scoring_engine(telemetry_manager=None) -> DynamicScoringEngine:
    """Factory function to get scoring engine instance."""
    return DynamicScoringEngine(telemetry_manager)
