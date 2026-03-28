"""
Scoring engine for dynamic provider/model ranking.
Combines agentic coding scores and TPS into unified scores.
Availability is used as a yes/no filter, not a scoring factor.
"""

import logging
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ScoringEngine:
    """
    Ranks provider+model combinations using:
    - Virtual models: Agentic(70%) + TPS(30%)
    - Specific models: TPS(70%) + Quality(30%)
    - Availability is a yes/no filter (excluded providers)
    """

    def __init__(self, telemetry=None):
        self._telemetry = telemetry  # lazy-loaded if None
        self.agentic_scores = self._load_agentic_scores()
        self.max_observed_tps = 3000  # Will be updated from telemetry

    @property
    def telemetry(self):
        if self._telemetry is None:
            try:
                from rotator_library.telemetry import get_telemetry_manager
                self._telemetry = get_telemetry_manager()
            except ImportError:
                try:
                    from telemetry import get_telemetry_manager
                    self._telemetry = get_telemetry_manager()
                except ImportError:
                    pass
        return self._telemetry

    def _load_agentic_scores(self) -> Dict[str, float]:
        """Load SWE-bench agentic coding scores from model_rankings.yaml.
        Falls back to hardcoded scores if file not found.
        """
        # Resolve path relative to this file
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "model_rankings.yaml"

        if config_path.exists():
            try:
                with open(config_path) as f:
                    data = yaml.safe_load(f)

                scores: Dict[str, float] = {}
                for model in data.get("models", []):
                    model_id = model.get("id", "")
                    ag = model.get("scores", {}).get("agentic_coding", -1)
                    if ag > 0 and model_id:
                        # Store both full id (provider/model) and bare model name
                        scores[model_id] = ag
                        # Also store just the model part for fuzzy matching
                        if "/" in model_id:
                            bare = model_id.split("/", 1)[1]
                            scores[bare] = ag

                if scores:
                    logger.info(f"Loaded {len(scores)} agentic scores from {config_path}")
                    return scores

            except Exception as e:
                logger.warning(f"Failed to load model_rankings.yaml: {e}")

        # Fallback: hardcoded scores
        logger.warning("Using hardcoded agentic scores (model_rankings.yaml not found)")
        return {
            "claude-opus-4-5": 74.4,
            "claude-opus-4.5": 74.4,
            "gemini-3-pro": 74.2,
            "gpt-5-2": 71.8,
            "gpt-5.2": 71.8,
            "claude-sonnet-4-5": 70.6,
            "claude-sonnet-4.5": 70.6,
            "gpt-4": 68.5,
            "gpt-4o": 68.3,
            "gemini-1.5-pro": 67.2,
            "llama-3.3-70b": 65.2,
            "llama-3.3-70b-versatile": 65.2,
            "codestral": 63.5,
        }

    def reload_scores(self):
        """Reload agentic scores from disk (call after leaderboard update)."""
        self.agentic_scores = self._load_agentic_scores()

    def get_agentic_score(self, model: str) -> float:
        """Get agentic coding score for a model (normalized 0-1)."""
        model_lower = model.lower().strip()

        # Exact match first
        if model_lower in self.agentic_scores:
            return self.agentic_scores[model_lower] / 100.0

        # Fuzzy match
        for key, score in self.agentic_scores.items():
            if key.lower() in model_lower or model_lower in key.lower():
                return score / 100.0

        return 0.5

    def get_average_tps(self, provider: str, model: str, minutes: int = 60) -> float:
        """Get average TPS from telemetry data."""
        if self.telemetry is None:
            return 0.0
        try:
            import sqlite3
            from datetime import datetime, timedelta
            since = (datetime.now() - timedelta(minutes=minutes)).isoformat()

            conn = sqlite3.connect(self.telemetry.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT AVG(tokens_per_second)
                FROM api_calls
                WHERE provider = ? AND model = ? AND timestamp >= ? AND success = 1
                AND tokens_per_second IS NOT NULL
                """,
                (provider, model, since),
            )
            result = cursor.fetchone()[0]
            conn.close()
            return float(result) if result else 0.0
        except Exception as e:
            logger.error(f"Failed to get TPS for {provider}/{model}: {e}")
            return 0.0

    def is_provider_available(
        self,
        provider: str,
        model: Optional[str] = None,
    ) -> bool:
        """Check if provider is available (not rate-limited, not unhealthy)."""
        if self.telemetry is None:
            return True
        try:
            health = self.telemetry.get_provider_health(provider, model)
            if not health.get("is_healthy", True):
                return False
            if health.get("consecutive_failures", 0) > 5:
                return False
            is_limited, _ = self.telemetry.check_rate_limit(provider, model or "*")
            if is_limited:
                return False
        except Exception:
            pass
        return True

    def calculate_virtual_model_score(self, provider: str, model: str) -> float:
        """Score = Agentic(70%) + TPS(30%). Returns 0.0 if unavailable."""
        if not self.is_provider_available(provider, model):
            return 0.0
        agentic_score = self.get_agentic_score(model)
        tps = self.get_average_tps(provider, model, minutes=60)
        tps_score = min(tps / 3000.0, 1.0) if tps else 0.1
        return (agentic_score * 0.70) + (tps_score * 0.30)

    def calculate_specific_model_score(self, provider: str, model: str) -> float:
        """Score = TPS(70%) + Agentic(30%). Returns 0.0 if unavailable."""
        if not self.is_provider_available(provider, model):
            return 0.0
        tps = self.get_average_tps(provider, model, minutes=60)
        tps_score = min(tps / 3000.0, 1.0) if tps else 0.1
        agentic_score = self.get_agentic_score(model)
        return (tps_score * 0.70) + (agentic_score * 0.30)

    def rank_virtual_models(
        self,
        model_candidates: List[str],
        providers_per_model: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """Rank provider+model combinations for virtual models."""
        candidates = []
        for model in model_candidates:
            for provider in providers_per_model.get(model, []):
                score = self.calculate_virtual_model_score(provider, model)
                if score > 0:
                    candidates.append({"provider": provider, "model": model, "score": score})
        return sorted(candidates, key=lambda x: x["score"], reverse=True)

    def rank_specific_model(
        self,
        model: str,
        providers: List[str],
    ) -> List[Dict[str, Any]]:
        """Rank providers for a specific model."""
        candidates = []
        for provider in providers:
            score = self.calculate_specific_model_score(provider, model)
            if score > 0:
                candidates.append({"provider": provider, "model": model, "score": score})
        return sorted(candidates, key=lambda x: x["score"], reverse=True)


_scoring_engine: Optional[ScoringEngine] = None


def get_scoring_engine() -> ScoringEngine:
    """Get or create the global scoring engine instance."""
    global _scoring_engine
    if _scoring_engine is None:
        _scoring_engine = ScoringEngine()
    return _scoring_engine
