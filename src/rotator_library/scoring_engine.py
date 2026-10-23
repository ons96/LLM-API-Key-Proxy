"""Dynamic Model Scoring Engine

Calculates real-time scores for models based on agentic coding performance,
tokens per second, and availability with configurable weights and thresholds.

Quality sources, in order (#1122):
  1. `config/model_rankings.yaml` row (agentic_coding, else swe_bench)
  2. `config/quality_scores.json` -- the gateway catalog export, which scores
     far more models than the rankings file has rows for
  3. neutral prior -- the median of all known scores. Models with no benchmark
     row are *unknown*, not worthless: they rank mid-pack instead of being
     pinned to the bottom of every chain.
"""

import os
import time
import sqlite3
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

from .quality_scores import load_quality_scores

logger = logging.getLogger(__name__)


@dataclass
class ModelScore:
    provider: str
    model: str
    agentic_score: float
    tps: float
    availability: float
    hallucination_rate: float
    web_search_capable: bool
    total_score: float
    meets_threshold: bool
    # False when the quality figure is the neutral prior rather than a
    # measured benchmark (no rankings row and no catalog score).
    quality_known: bool = True


class DynamicScoringEngine:
    """Calculates dynamic scores for virtual model fallback ordering.

    Models are ordered purely by performance benchmarks + TPS + hallucination.
    Rate limits and availability are NOT used for ordering — rate-limited
    providers are skipped at runtime instead.
    """

    # ponytail: 80/15/5 per DESIGN.md #341 (agentic/tps/availability).
    # Hallucination is a subtractive penalty, NOT part of the 80/15/5 split,
    # so weight sums below exclude it. Availability IS now scored (5%).
    # #1122: the blend differs by tier instead of being one 80/15/5 everywhere.
    #   elite / smart -- quality first (80/15/5, per #341 / DESIGN.md)
    #   fast          -- latency tier: speed rises to 40%, quality keeps the
    #                    majority so a quick-but-weak model still cannot lead
    #                    a coding chain
    # Availability stays a constant 5% floor everywhere; hallucination is
    # subtractive and sits outside the blend.
    CATEGORY_WEIGHTS = {
        "coding-elite": {
            "agentic": 0.80,
            "tps": 0.15,
            "availability": 0.05,
            "hallucination_penalty": 0.10,
        },
        "coding-smart": {
            "agentic": 0.80,
            "tps": 0.15,
            "availability": 0.05,
            "hallucination_penalty": 0.15,
        },
        "coding-fast": {
            "agentic": 0.55,
            "tps": 0.40,
            "availability": 0.05,
            "hallucination_penalty": 0.15,
        },
        "chat-elite": {
            "intelligence": 0.80,
            "tps": 0.15,
            "availability": 0.05,
            "hallucination_penalty": 0.20,
        },
        "chat-smart": {
            "intelligence": 0.80,
            "tps": 0.15,
            "availability": 0.05,
            "hallucination_penalty": 0.20,
        },
        "chat-fast": {
            "intelligence": 0.55,
            "tps": 0.40,
            "availability": 0.05,
            "hallucination_penalty": 0.20,
        },
    }

    CODING_WEIGHTS = {
        "agentic": 0.80,
        "tps": 0.15,
        "availability": 0.05,
        "hallucination_penalty": 0.10,
    }

    CHAT_WEIGHTS = {
        "intelligence": 0.80,
        "tps": 0.15,
        "availability": 0.05,
        "hallucination_penalty": 0.15,
    }

    # ponytail: SWE-bench floors per #341. Below-floor models demoted to
    # last-resort priority slot, NOT excluded (chain stays resilient).
    THRESHOLDS = {
        "coding-elite": 70.0,
        "coding-smart": 65.0,
        "coding-fast": 0.0,  # fast tier: no SWE floor, speed-first
        "chat-elite": 0.0,
        "chat-smart": 0.0,
        "chat-fast": 0.0,
    }

    # ponytail: free-model baseline = worst available free model's agentic
    # score. Models below baseline excluded from coding-* chains entirely.
    # Upgraded to per-tier if FREEDOT_BASELINES set, else computed at load.
    FREE_BASELINE_FLOOR = 30.0  # hard floor; computed baseline >= this

    # #1122: neutral prior for models nobody has scored. The median of known
    # scores, clamped so a degenerate export cannot move it out of band.
    NEUTRAL_PRIOR_MIN = 30.0
    NEUTRAL_PRIOR_MAX = 70.0
    NEUTRAL_PRIOR_FALLBACK = 50.0

    def __init__(
        self,
        telemetry_manager=None,
        model_rankings_path: str = "config/model_rankings.yaml",
        quality_scores_path: Optional[str] = None,
    ):
        self.telemetry = telemetry_manager
        self.model_rankings_path = Path(model_rankings_path)
        self.quality_scores_path = quality_scores_path or os.environ.get(
            "QUALITY_SCORES_PATH", "config/quality_scores.json"
        )
        self._rankings_cache: Dict[str, Dict] = {}
        self._catalog_quality: Dict[str, float] = {}
        self._neutral_prior = self.NEUTRAL_PRIOR_FALLBACK
        self._last_refresh = 0

    def load_model_rankings(self) -> Dict[str, Dict]:
        """Load model rankings from YAML file."""
        import yaml

        try:
            with open(self.model_rankings_path) as f:
                data = yaml.safe_load(f)

            rankings = {}
            for model in data.get("models", []):
                model_id = model.get("id", "")
                if "/" in model_id:
                    provider, model_name = model_id.split("/", 1)
                    key = f"{provider}/{model_name}"
                    rankings[key] = model

            self._rankings_cache = rankings
            self._last_refresh = time.time()
            self._compute_free_baseline(rankings)
            self._load_catalog_quality()
            self._compute_neutral_prior()
            return rankings

        except Exception as e:
            logger.error(f"Failed to load model rankings: {e}")
            # The catalog export can still score models without the rankings
            # file, so load it either way (#1122).
            self._load_catalog_quality()
            self._compute_neutral_prior()
            return {}

    def _compute_free_baseline(self, rankings: Dict[str, Dict]) -> None:
        """Compute the free-model baseline: worst agentic score among free
        providers. Models scoring below this are excluded from coding chains.
        Runs at load + every reload (per #341 acceptance criterion 5).
        """
        # ponytail: FREE_PROVIDERS matches generate_virtual_models.py list.
        free_providers = {
            "groq", "cerebras", "gemini", "together", "g4f",
            "nvidia", "github-models", "kilo", "modal",
        }
        free_scores = []
        for key, model_data in rankings.items():
            provider = key.split("/", 1)[0] if "/" in key else ""
            if provider not in free_providers:
                continue
            scores = model_data.get("scores", {})
            ag = scores.get("agentic_coding", scores.get("swe_bench", 0.0))
            if ag > 0:
                free_scores.append(ag)

        if free_scores:
            self._free_baseline = max(min(free_scores), self.FREE_BASELINE_FLOOR)
            logger.info(f"Free-model baseline: {self._free_baseline}")
        else:
            self._free_baseline = self.FREE_BASELINE_FLOOR

    def _load_catalog_quality(self) -> None:
        """Catalog export -> {model id: 0..1}; {} when the file is absent."""
        self._catalog_quality = load_quality_scores(self.quality_scores_path)

    def _compute_neutral_prior(self) -> None:
        """Median of every known quality score, clamped to a sane band."""
        known = [
            float(score)
            for entry in self._rankings_cache.values()
            for score in [
                entry.get("scores", {}).get(
                    "agentic_coding", entry.get("scores", {}).get("swe_bench")
                )
            ]
            if isinstance(score, (int, float)) and score > 0
        ]
        known.extend(value * 100.0 for value in self._catalog_quality.values())

        if not known:
            self._neutral_prior = self.NEUTRAL_PRIOR_FALLBACK
            return

        known.sort()
        mid = len(known) // 2
        median = known[mid] if len(known) % 2 else (known[mid - 1] + known[mid]) / 2.0
        self._neutral_prior = min(
            max(median, self.NEUTRAL_PRIOR_MIN), self.NEUTRAL_PRIOR_MAX
        )
        logger.info(
            "Neutral quality prior: %.1f (from %d scored models)",
            self._neutral_prior,
            len(known),
        )

    @property
    def neutral_prior(self) -> float:
        """Quality assumed for a model with no benchmark row at all."""
        return self._neutral_prior

    def _below_free_baseline(self, agentic_score: float) -> bool:
        """True if model scores below the free-model baseline."""
        baseline = getattr(self, "_free_baseline", self.FREE_BASELINE_FLOOR)
        return 0 < agentic_score < baseline

    def get_model_score(self, provider: str, model: str) -> Optional[float]:
        """Get agentic coding score for a model."""
        if time.time() - self._last_refresh > 3600:
            self.load_model_rankings()

        key = f"{provider}/{model}"
        model_data = self._rankings_cache.get(key)

        if not model_data:
            for k, v in self._rankings_cache.items():
                if (
                    k.endswith(f"/{model}")
                    or v.get("name", "").lower() == model.lower()
                ):
                    model_data = v
                    break

        ranking_score = None
        if model_data:
            scores = model_data.get("scores", {})
            ranking_score = scores.get("agentic_coding", scores.get("swe_bench"))
        if isinstance(ranking_score, (int, float)) and ranking_score > 0:
            return float(ranking_score)

        # No usable rankings row (279 of 4592 catalog models have one): fall
        # back to the catalog export before giving up.
        catalog = self._catalog_quality.get(key)
        if catalog is None:
            suffix = f"/{model.lower()}"
            for candidate_key, value in self._catalog_quality.items():
                if candidate_key.endswith(suffix):
                    catalog = value
                    break
        if catalog is not None:
            return catalog * 100.0

        return None  # caller substitutes the neutral prior

    def get_hallucination_rate(self, provider: str, model: str) -> float:
        if time.time() - self._last_refresh > 3600:
            self.load_model_rankings()

        key = f"{provider}/{model}"
        model_data = self._rankings_cache.get(key)

        if not model_data:
            for k, v in self._rankings_cache.items():
                if (
                    k.endswith(f"/{model}")
                    or v.get("name", "").lower() == model.lower()
                ):
                    model_data = v
                    break

        if model_data:
            scores = model_data.get("scores", {})
            return scores.get("hallucination_rate", 10.0)

        return 10.0

    def get_web_search_capable(self, provider: str, model: str) -> bool:
        if time.time() - self._last_refresh > 3600:
            self.load_model_rankings()

        key = f"{provider}/{model}"
        model_data = self._rankings_cache.get(key)

        if not model_data:
            for k, v in self._rankings_cache.items():
                if (
                    k.endswith(f"/{model}")
                    or v.get("name", "").lower() == model.lower()
                ):
                    model_data = v
                    break

        if model_data:
            capabilities = model_data.get("capabilities", {})
            return capabilities.get("web_search_capable", False)

        return False

    def get_tps_estimate(self, provider: str, model: str) -> float:
        """Get TPS from telemetry or return default estimate."""
        if not self.telemetry:
            return 100.0

        try:
            conn = sqlite3.connect(self.telemetry.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT AVG(tps) FROM tps_metrics 
                WHERE provider = ? AND model = ? 
                AND timestamp > datetime('now', '-1 hour')
            """,
                (provider, model),
            )

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

            cursor.execute(
                """
                SELECT 
                    COUNT(CASE WHEN success = 1 THEN 1 END) as success,
                    COUNT(*) as total
                FROM api_calls
                WHERE provider = ? AND model = ?
                AND timestamp > datetime('now', '-30 minutes')
            """,
                (provider, model),
            )

            result = cursor.fetchone()
            conn.close()

            if result and result[1] > 0:
                return result[0] / result[1]

        except Exception as e:
            logger.debug(f"Failed to get availability: {e}")

        return 1.0

    def calculate_coding_score(
        self, provider: str, model: str, virtual_model_type: str = "coding-elite"
    ) -> ModelScore:
        measured = self.get_model_score(provider, model)
        quality_known = measured is not None
        # #1122: unknown != worthless. Unscored models rank at the neutral
        # prior, so they neither lead a chain on faith nor sink to the bottom.
        agentic_score = measured if quality_known else self.neutral_prior
        tps = self.get_tps_estimate(provider, model)
        availability = self.get_availability(provider, model)
        hallucination_rate = self.get_hallucination_rate(provider, model)
        web_search_capable = self.get_web_search_capable(provider, model)

        # ponytail: normalize to 0-100 range to match agentic_score scale.
        # tps/20 caps at 100 (200+ TPS = max). availability already 0-1 -> *100.
        tps_normalized = min(tps / 20.0, 100.0)
        availability_normalized = availability * 100.0
        hallucination_penalty = min(hallucination_rate / 20.0, 100.0)

        if web_search_capable:
            hallucination_penalty *= 0.5

        weights = self.CATEGORY_WEIGHTS.get(virtual_model_type, self.CODING_WEIGHTS)

        total_score = 0.0
        if "agentic" in weights:
            total_score += agentic_score * weights["agentic"]
        if "intelligence" in weights:
            total_score += agentic_score * weights["intelligence"]
        if "tps" in weights:
            total_score += tps_normalized * weights["tps"]
        if "availability" in weights:
            total_score += availability_normalized * weights["availability"]
        if "hallucination_penalty" in weights:
            total_score -= hallucination_penalty * weights["hallucination_penalty"]

        threshold = self.THRESHOLDS.get(virtual_model_type, 0.0)
        meets_threshold = agentic_score >= threshold

        # ponytail: free-baseline exclusion (#341) — coding-* only. Chat
        # chains use intelligence, not agentic_coding, so the coding baseline
        # would wrongly penalize good chat models with low SWE-bench scores.
        if virtual_model_type.startswith("coding") and self._below_free_baseline(
            agentic_score
        ):
            total_score = 0.0
            meets_threshold = False

        return ModelScore(
            provider=provider,
            model=model,
            agentic_score=agentic_score,
            tps=tps,
            availability=availability,
            hallucination_rate=hallucination_rate,
            web_search_capable=web_search_capable,
            total_score=total_score,
            meets_threshold=meets_threshold,
            quality_known=quality_known,
        )

    def rank_models_for_virtual(
        self, virtual_model_type: str, candidates: List[tuple]
    ) -> List[ModelScore]:
        """Rank candidate models by score for a virtual model.

        Args:
            virtual_model_type: e.g., 'coding-elite', 'coding-smart'
            candidates: List of (provider, model) tuples

        Returns:
            List of ModelScore objects sorted by total_score descending.
            Below-threshold models demoted to tail (last-resort) per #341.
        """
        scores = []

        for provider, model in candidates:
            score = self.calculate_coding_score(provider, model, virtual_model_type)
            scores.append(score)

        # ponytail: #341 threshold enforcement. Sort by (meets_threshold,
        # total_score) so below-threshold models sink to tail regardless of
        # their raw score. Free-baseline-excluded (score=0) sort last too.
        scores.sort(
            key=lambda s: (s.meets_threshold, s.total_score),
            reverse=True,
        )

        return scores


def get_scoring_engine(telemetry_manager=None) -> DynamicScoringEngine:
    """Factory function to get scoring engine instance."""
    return DynamicScoringEngine(telemetry_manager)
