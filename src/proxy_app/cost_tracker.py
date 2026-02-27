"""Cost tracking for LLM requests.

Phase 3.2 Usage Analytics - Cost calculation per request.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Optional
from decimal import Decimal, ROUND_HALF_UP

logger = logging.getLogger(__name__)


class CostTracker:
    """Tracks and calculates costs for model requests based on config/model_rankings.yaml."""
    
    def __init__(self, rankings_path: str = "config/model_rankings.yaml"):
        self.rankings_path = Path(rankings_path)
        self._costs: Dict[str, Dict[str, float]] = {}
        self._load_costs()
    
    def _load_costs(self):
        """Load cost data from model rankings."""
        try:
            if not self.rankings_path.exists():
                logger.warning(f"Rankings file not found: {self.rankings_path}")
                return
            
            with open(self.rankings_path) as f:
                data = yaml.safe_load(f)
            
            models = data.get("models", [])
            for model in models:
                model_id = model.get("id")
                cost_data = model.get("cost", {})
                
                if model_id and cost_data:
                    # Costs stored as per-million-token rates
                    self._costs[model_id] = {
                        "input": float(cost_data.get("input", 0)),
                        "output": float(cost_data.get("output", 0))
                    }
            
            logger.info(f"Loaded cost data for {len(self._costs)} models")
            
        except Exception as e:
            logger.error(f"Failed to load cost data: {e}")
    
    def get_cost_per_million(self, model_id: str) -> Dict[str, float]:
        """Get cost per million tokens for a model."""
        return self._costs.get(model_id, {"input": 0.0, "output": 0.0})
    
    def calculate(
        self, 
        model_id: str, 
        prompt_tokens: int, 
        completion_tokens: int
    ) -> Dict:
        """
        Calculate actual cost for a request.
        
        Args:
            model_id: Full model identifier (e.g., "openai/gpt-4", "anthropic/claude-3-opus")
            prompt_tokens: Number of input/prompt tokens used
            completion_tokens: Number of output/completion tokens used
            
        Returns:
            Dict with detailed cost breakdown
        """
        rates = self.get_cost_per_million(model_id)
        
        # Calculate costs (convert from per-million to actual)
        input_cost = (prompt_tokens / 1_000_000) * rates["input"]
        output_cost = (completion_tokens / 1_000_000) * rates["output"]
        total_cost = input_cost + output_cost
        
        # Round to 6 decimal places for micro-dollar precision
        total = Decimal(str(total_cost)).quantize(
            Decimal("0.000001"), 
            rounding=ROUND_HALF_UP
        )
        
        return {
            "model_id": model_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": float(total),
            "rates_per_million": rates
        }
    
    def refresh_costs(self):
        """Reload cost data from disk (call when rankings are updated)."""
        self._load_costs()
        logger.info("Cost data refreshed")
