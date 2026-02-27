"""
Integration module for benchmark data into the proxy routing system.
Connects the benchmark fetcher with the model ranker.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional

from rotator_library.benchmark_fetcher import BenchmarkFetcher, BenchmarkEntry
from proxy_app.model_ranker import ModelRanker

logger = logging.getLogger(__name__)


class BenchmarkIntegration:
    """
    Integrates benchmark data with the proxy routing decisions.
    Updates model rankings based on external benchmark data.
    """
    
    def __init__(
        self,
        model_ranker: Optional[ModelRanker] = None,
        fetcher: Optional[BenchmarkFetcher] = None,
        config_path: Optional[Path] = None
    ):
        self.model_ranker = model_ranker
        self.fetcher = fetcher or BenchmarkFetcher(config_path=config_path)
        self._refresh_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def initialize(self) -> None:
        """Initialize the fetcher and load initial data."""
        await self.fetcher.initialize()
        logger.info("Benchmark integration initialized")
        
    async def refresh_benchmarks(self) -> None:
        """Manually trigger a benchmark refresh."""
        try:
            cache = await self.fetcher.fetch_all(force=True)
            logger.info(f"Refreshed {len(cache.entries)} benchmark entries")
            
            if self.model_ranker:
                self._update_rankings(cache.entries)
                
        except Exception as e:
            logger.error(f"Failed to refresh benchmarks: {e}")
            
    def _update_rankings(self, entries: List[BenchmarkEntry]) -> None:
        """Update model ranker with new benchmark data."""
        # Group by model
        model_scores: Dict[str, Dict[str, float]] = {}
        for entry in entries:
            if entry.model_id not in model_scores:
                model_scores[entry.model_id] = {}
            model_scores[entry.model_id][entry.metric_name] = entry.metric_value
        
        # Update rankings (implementation depends on ModelRanker interface)
        for model_id, scores in model_scores.items():
            logger.debug(f"Updating scores for {model_id}: {scores}")
            # Integration point: notify ranker of new scores
            if hasattr(self.model_ranker, 'update_benchmark_scores'):
                self.model_ranker.update_benchmark_scores(model_id, scores)
    
    async def start_background_refresh(self, interval_minutes: int = 60) -> None:
        """Start background task to periodically refresh benchmarks."""
        self._running = True
        
        async def refresh_loop():
            while self._running:
                try:
                    await self.refresh_benchmarks()
                except Exception as e:
                    logger.error(f"Background refresh error: {e}")
                
                await asyncio.sleep(interval_minutes * 60)
        
        self._refresh_task = asyncio.create_task(refresh_loop())
        logger.info(f"Started background benchmark refresh (interval: {interval_minutes}m)")
        
    async def stop(self) -> None:
        """Stop background refresh."""
        self._running = False
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        await self.fetcher.close()
        logger.info("Benchmark integration stopped")
        
    def get_model_performance_summary(self, model_id: str) -> Dict[str, any]:
        """Get performance summary for a model from latest benchmarks."""
        entries = self.fetcher.cache.get_by_model(model_id)
        
        summary = {
            "model_id": model_id,
            "last_updated": self.fetcher.cache.last_updated,
            "metrics": {},
            "sources": set()
        }
        
        for entry in entries:
            summary["metrics"][entry.metric_name] = {
                "value": entry.metric_value,
                "unit": entry.unit,
                "source": entry.source
            }
            summary["sources"].add(entry.source)
        
        summary["sources"] = list(summary["sources"])
        return summary


# Factory function for easy integration
async def setup_benchmark_integration(
    model_ranker: Optional[ModelRanker] = None,
    auto_refresh: bool = True
) -> BenchmarkIntegration:
    """
    Setup function to integrate benchmarks into the proxy.
    
    Usage in main.py:
        benchmark_integration = await setup_benchmark_integration(model_ranker)
        await benchmark_integration.start_background_refresh()
    """
    integration = BenchmarkIntegration(model_ranker=model_ranker)
    await integration.initialize()
    
    # Do initial fetch
    await integration.refresh_benchmarks()
    
    if auto_refresh:
        await integration.start_background_refresh()
        
    return integration
