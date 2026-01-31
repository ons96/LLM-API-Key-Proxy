"""
Scoring engine for dynamic provider/model ranking.
Combines agentic coding scores and TPS into unified scores.
Availability is used as a yes/no filter, not a scoring factor.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from telemetry import TelemetryManager, get_telemetry_manager

logger = logging.getLogger(__name__)


class ScoringEngine:
    """
    Ranks provider+model combinations using:
    - Virtual models: Agentic(70%) + TPS(30%)
    - Specific models: TPS(70%) + Quality(30%)
    - Availability is a yes/no filter (excluded providers)
    """

    def __init__(self, telemetry: Optional[TelemetryManager] = None):
        self.telemetry = telemetry or get_telemetry_manager()
        self.agentic_scores = self._load_agentic_scores()
        self.max_observed_tps = 3000  # Will be updated from telemetry

    def _load_agentic_scores(self) -> Dict[str, float]:
        """Load SWE-bench agentic coding scores for models."""
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

    def get_agentic_score(self, model: str) -> float:
        """Get agentic coding score for a model (normalized 0-1)."""
        model_lower = model.lower().strip()

        for key, score in self.agentic_scores.items():
            if key.lower() in model_lower or model_lower in key.lower():
                return score / 100.0

        return 0.5

    def get_average_tps(self, provider: str, model: str, minutes: int = 60) -> float:
        """Get average TPS from telemetry data."""
        try:
            since = datetime.now() - timedelta(minutes=minutes)

            conn = self.telemetry._connect()
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

            return result or 0.0

        except Exception as e:
            logger.error(f"Failed to get TPS for {provider}/{model}: {e}")
            return 0.0

    def is_provider_available(
        self,
        provider: str,
        model: Optional[str] = None,
    ) -> bool:
        """
        Check if provider is available (not rate-limited, not unhealthy).
        This is a yes/no filter, not a scoring factor.
        """
        health = self.telemetry.get_provider_health(provider, model)

        if not health.get("is_healthy", True):
            logger.debug(f"{provider}/{model or '*'} not available: unhealthy")
            return False

        if health.get("consecutive_failures", 0) > 5:
            logger.debug(f"{provider}/{model or '*'} not available: too many failures")
            return False

        is_limited, _ = self.telemetry.check_rate_limit(provider, model or "*")
        if is_limited:
            logger.debug(f"{provider}/{model or '*'} not available: rate limited")
            return False

        return True

    def calculate_virtual_model_score(
        self,
        provider: str,
        model: str,
    ) -> float:
        """
        Calculate score for virtual models:
        Score = Agentic(70%) + TPS(30%)
        Availability is a yes/no filter (excluded if not available)
        """
        if not self.is_provider_available(provider, model):
            return 0.0

        agentic_score = self.get_agentic_score(model)
        tps = self.get_average_tps(provider, model, minutes=60)
        tps_score = min(tps / 3000.0, 1.0) if tps else 0.1

        final_score = (agentic_score * 0.70) + (tps_score * 0.30)

        logger.debug(
            f"Virtual model score for {provider}/{model}: Agentic={agentic_score:.3f}, TPS={tps_score:.3f}, "
            f"Final={final_score:.3f}"
        )

        return final_score

    def calculate_specific_model_score(
        self,
        provider: str,
        model: str,
    ) -> float:
        """
        Calculate score for specific models:
        Score = TPS(70%) + Agentic(30%)
        Availability is a yes/no filter (excluded if not available)
        """
        if not self.is_provider_available(provider, model):
            return 0.0

        tps = self.get_average_tps(provider, model, minutes=60)
        tps_score = min(tps / 3000.0, 1.0) if tps else 0.1

        agentic_score = self.get_agentic_score(model)

        final_score = (tps_score * 0.70) + (agentic_score * 0.30)

        logger.debug(
            f"Specific model score for {provider}/{model}: TPS={tps_score:.3f}, "
            f"Agentic={agentic_score:.3f}, Final={final_score:.3f}"
        )

        return final_score

    def rank_virtual_models(
        self,
        model_candidates: List[str],
        providers_per_model: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Rank provider+model combinations for virtual models.
        Returns sorted list of {provider, model, score}
        """
        candidates = []

        for model in model_candidates:
            providers = providers_per_model.get(model, [])

            for provider in providers:
                score = self.calculate_virtual_model_score(provider, model)

                if score > 0:  # Only include available providers
                    candidates.append(
                        {
                            "provider": provider,
                            "model": model,
                            "score": score,
                            "agentic_score": self.get_agentic_score(model),
                        }
                    )

        ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)

        logger.info(f"Ranked {len(ranked)} candidates for virtual models")
        return ranked

    def rank_specific_model(
        self,
        model: str,
        providers: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Rank providers for a specific model (fastest first with some quality).
        Returns sorted list of {provider, model, score}
        """
        candidates = []

        for provider in providers:
            score = self.calculate_specific_model_score(provider, model)

            if score > 0:  # Only include available providers
                candidates.append(
                    {
                        "provider": provider,
                        "model": model,
                        "score": score,
                    }
                )

        ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)

        logger.info(f"Ranked {len(ranked)} providers for specific model {model}")
        return ranked


# Global scoring engine instance
_scoring_engine: Optional[ScoringEngine] = None


def get_scoring_engine() -> ScoringEngine:
    """Get or create the global scoring engine instance."""
    global _scoring_engine
    if _scoring_engine is None:
        _scoring_engine = ScoringEngine()
    return _scoring_engine
