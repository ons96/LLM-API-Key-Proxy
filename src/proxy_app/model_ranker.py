import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ModelRanker:
    """
    Ranks provider candidates based on benchmark scores and use-case suitability.
    """

    def __init__(self, config_path: str = "config/model_rankings.yaml"):
        self.rankings = self._load_rankings(config_path)

    def _load_rankings(self, config_path: str) -> Dict[str, Any]:
        """Load model rankings from YAML."""
        root_dir = Path(__file__).resolve().parent.parent.parent
        path = root_dir / config_path

        if not path.exists():
            logger.warning(f"Model rankings file not found: {path}")
            return {}

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
                # Index by model ID for fast lookup
                return {m["id"]: m for m in data.get("models", [])}
        except Exception as e:
            logger.error(f"Failed to load model rankings: {e}")
            return {}

    def rank_candidates(
        self, virtual_model_id: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Re-order candidates based on suitability for the virtual model.
        Returns a sorted list of candidates.
        """
        if not self.rankings:
            return candidates

        # Define scoring metric based on use case
        metric = "speed_tps"  # Default
        if "coding" in virtual_model_id:
            metric = "humaneval"
        elif "smart" in virtual_model_id:
            metric = "humaneval"  # Proxy for reasoning quality
        elif "fast" in virtual_model_id:
            metric = "speed_tps"

        def get_score(candidate):
            # Try to find ranking entry
            # Candidate has 'provider' and 'model'
            # Ranking IDs are usually "provider/model" or just "model"
            # We try to construct ID

            # Direct match?
            provider = candidate.get("provider", "").lower()
            model = candidate.get("model", "").lower()

            # Possible IDs in ranking file
            possible_ids = [
                f"{provider}/{model}",
                f"{model}",
                # Sometimes provider mapping might be loose
            ]

            ranking = None
            for pid in possible_ids:
                if pid in self.rankings:
                    ranking = self.rankings[pid]
                    break

            if not ranking:
                return -1  # No data, push to bottom (or keep original relative order if stable sort)

            # Check if this model is actually tagged for this use case?
            # Prompt says "best_for: [coding-smart]".
            # If specified, we should boost it.
            base_score = ranking.get("scores", {}).get(metric, 0)

            # Boost if explicitly marked best_for this VM
            if virtual_model_id in ranking.get("best_for", []):
                base_score *= 1.5

            return base_score

        # Sort descending (higher score is better)
        # Use existing priority as tie-breaker (negative priority so lower number = higher rank)
        # Python sort is stable.

        # We want to preserve original priority tiers (e.g. don't put a fallback G4F above a paid model just because of score?)
        # Or does intelligent ordering override manual priority?
        # Prompt says "Generates optimal fallback chain". This implies overriding order.
        # But usually we want to respect "Free vs Paid" tiers.
        # Let's assume this re-orders within the same priority tier OR re-ranks completely.
        # Let's allow it to re-rank completely based on score for "smart" routing.

        sorted_candidates = sorted(candidates, key=get_score, reverse=True)

        # Update priority field to match new order
        for i, cand in enumerate(sorted_candidates):
            cand["priority"] = i + 1

        return sorted_candidates
