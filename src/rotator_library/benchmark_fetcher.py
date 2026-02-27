"""
Benchmark Fetcher Module for Phase 2.1 Data Collection Pipeline.

Fetches model performance data from various benchmark sources including:
- Artificial Analysis API
- LMSYS Chatbot Arena
- Other aggregated leaderboards

Provides normalized data structures for the model ranking system.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Union

import aiohttp
import yaml
from pydantic import BaseModel, Field, validator

# Configure logging
logger = logging.getLogger(__name__)


class BenchmarkEntry(BaseModel):
    """Normalized benchmark entry across all sources."""
    model_id: str
    source: str  # e.g., "artificial_analysis", "lmsys_arena"
    metric_name: str  # e.g., "throughput", "win_rate", "quality_score"
    metric_value: float
    unit: Optional[str] = None  # e.g., "tokens/sec", "percentage"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = Field(default_factory=dict)  # Extra metadata
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BenchmarkSourceConfig(BaseModel):
    """Configuration for a benchmark source."""
    name: str
    enabled: bool = True
    url: str
    api_key: Optional[str] = None
    refresh_interval_minutes: int = 60
    timeout_seconds: int = 30
    retry_attempts: int = 3
    priority: int = 100  # Lower = higher priority
    headers: Dict[str, str] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkCache(BaseModel):
    """Cache structure for benchmark data."""
    entries: List[BenchmarkEntry] = Field(default_factory=list)
    last_updated: Optional[datetime] = None
    source_status: Dict[str, str] = Field(default_factory=dict)  # source -> status
    
    def get_by_model(self, model_id: str) -> List[BenchmarkEntry]:
        """Get all benchmark entries for a specific model."""
        return [e for e in self.entries if e.model_id == model_id]
    
    def get_by_source(self, source: str) -> List[BenchmarkEntry]:
        """Get all entries from a specific source."""
        return [e for e in self.entries if e.source == source]
    
    def get_metric(self, model_id: str, metric_name: str) -> Optional[BenchmarkEntry]:
        """Get specific metric for a model."""
        for entry in self.entries:
            if entry.model_id == model_id and entry.metric_name == metric_name:
                return entry
        return None


class BenchmarkFetcherError(Exception):
    """Base exception for benchmark fetcher."""
    pass


class SourceUnavailableError(BenchmarkFetcherError):
    """Raised when a benchmark source is unavailable."""
    pass


class Fetcher(ABC):
    """Abstract base class for benchmark fetchers."""
    
    def __init__(self, config: BenchmarkSourceConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
    
    @abstractmethod
    async def fetch(self, session: aiohttp.ClientSession) -> List[BenchmarkEntry]:
        """Fetch benchmark data from the source."""
        pass
    
    @abstractmethod
    def normalize_model_id(self, raw_id: str) -> str:
        """Normalize raw model ID to standard format."""
        pass
    
    async def _fetch_with_retry(
        self, 
        session: aiohttp.ClientSession, 
        url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch data with retry logic."""
        url = url or self.config.url
        last_error = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                async with session.get(
                    url,
                    headers=self.config.headers,
                    params=self.config.params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        wait_time = 2 ** attempt
                        self.logger.warning(f"Rate limited, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        response.raise_for_status()
                        
            except aiohttp.ClientError as e:
                last_error = e
                wait_time = 2 ** attempt
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
        
        raise SourceUnavailableError(
            f"Failed to fetch from {self.config.name} after {self.config.retry_attempts} attempts: {last_error}"
        )


class ArtificialAnalysisFetcher(Fetcher):
    """Fetcher for Artificial Analysis API."""
    
    async def fetch(self, session: aiohttp.ClientSession) -> List[BenchmarkEntry]:
        """Fetch throughput and latency data from Artificial Analysis."""
        entries = []
        
        try:
            data = await self._fetch_with_retry(session)
            
            # Parse Artificial Analysis specific format
            models = data.get("models", [])
            for model_data in models:
                model_id = self.normalize_model_id(model_data.get("id", ""))
                
                # Extract throughput metrics
                if "throughput" in model_data:
                    entries.append(BenchmarkEntry(
                        model_id=model_id,
                        source="artificial_analysis",
                        metric_name="throughput",
                        metric_value=float(model_data["throughput"]),
                        unit="tokens/sec",
                        context={
                            "context_length": model_data.get("context_length"),
                            "provider": model_data.get("provider")
                        }
                    ))
                
                # Extract latency metrics
                if "latency" in model_data:
                    entries.append(BenchmarkEntry(
                        model_id=model_id,
                        source="artificial_analysis",
                        metric_name="latency",
                        metric_value=float(model_data["latency"]),
                        unit="ms",
                        context={
                            "percentile": model_data.get("latency_percentile", "p50")
                        }
                    ))
                
                # Extract quality scores if available
                if "quality_score" in model_data:
                    entries.append(BenchmarkEntry(
                        model_id=model_id,
                        source="artificial_analysis",
                        metric_name="quality_score",
                        metric_value=float(model_data["quality_score"]),
                        unit="normalized",
                        context={
                            "benchmark_suite": model_data.get("benchmark_suite", "unknown")
                        }
                    ))
                    
        except Exception as e:
            self.logger.error(f"Error parsing Artificial Analysis data: {e}")
            raise
            
        return entries
    
    def normalize_model_id(self, raw_id: str) -> str:
        """Normalize model ID to standard format."""
        # Remove provider prefixes, standardize casing
        normalized = raw_id.lower().strip()
        replacements = {
            "anthropic/": "",
            "openai/": "",
            "google/": "",
            "meta/": "",
            "-": "_",
            " ": "_"
        }
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        return normalized


class LMSYSArenaFetcher(Fetcher):
    """Fetcher for LMSYS Chatbot Arena leaderboard."""
    
    ARENA_API_URL = "https://chat.lmsys.org/api/leaderboard"
    
    async def fetch(self, session: aiohttp.ClientSession) -> List[BenchmarkEntry]:
        """Fetch ELO ratings and win rates from LMSYS Arena."""
        entries = []
        
        try:
            data = await self._fetch_with_retry(session)
            
            # Parse LMSYS specific format
            leaderboard = data.get("leaderboard", [])
            for entry in leaderboard:
                model_id = self.normalize_model_id(entry.get("model", ""))
                
                # ELO rating
                if "elo" in entry:
                    entries.append(BenchmarkEntry(
                        model_id=model_id,
                        source="lmsys_arena",
                        metric_name="elo_rating",
                        metric_value=float(entry["elo"]),
                        unit="elo",
                        context={
                            "confidence_interval": entry.get("ci"),
                            "votes": entry.get("votes", 0)
                        }
                    ))
                
                # Win rate if available
                if "win_rate" in entry:
                    entries.append(BenchmarkEntry(
                        model_id=model_id,
                        source="lmsys_arena",
                        metric_name="win_rate",
                        metric_value=float(entry["win_rate"]),
                        unit="percentage",
                        context={}
                    ))
                
                # Style control ELO (if available)
                if "style_control_elo" in entry:
                    entries.append(BenchmarkEntry(
                        model_id=model_id,
                        source="lmsys_arena",
                        metric_name="style_control_elo",
                        metric_value=float(entry["style_control_elo"]),
                        unit="elo",
                        context={}
                    ))
                    
        except Exception as e:
            self.logger.error(f"Error parsing LMSYS Arena data: {e}")
            raise
            
        return entries
    
    def normalize_model_id(self, raw_id: str) -> str:
        """Normalize LMSYS model names to standard IDs."""
        normalized = raw_id.lower().strip()
        # Handle specific LMSYS naming conventions
        mapping = {
            "gpt-4": "gpt-4",
            "gpt-4-turbo": "gpt_4_turbo",
            "claude-3-opus": "claude_3_opus",
            "claude-3-sonnet": "claude_3_sonnet",
            "claude-3-haiku": "claude_3_haiku",
            "gemini-pro": "gemini_pro",
            "llama-3": "llama_3",
            "mixtral": "mixtral_8x7b"
        }
        
        for lmsys_name, standard_name in mapping.items():
            if lmsys_name in normalized:
                return standard_name
                
        # General normalization
        return normalized.replace("-", "_").replace(" ", "_")


class StaticFileFetcher(Fetcher):
    """Fetcher for static/local benchmark files."""
    
    async def fetch(self, session: aiohttp.ClientSession) -> List[BenchmarkEntry]:
        """Fetch from local JSON/YAML files."""
        entries = []
        
        try:
            # Check if URL is a local file path
            file_path = Path(self.config.url)
            if not file_path.exists():
                self.logger.warning(f"Static file not found: {file_path}")
                return entries
            
            content = file_path.read_text()
            
            if file_path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(content)
            else:
                data = json.loads(content)
            
            # Expect format: {"benchmarks": [{"model_id": "...", "metric": "...", "value": ...}]}
            benchmarks = data.get("benchmarks", [])
            for item in benchmarks:
                entries.append(BenchmarkEntry(
                    model_id=self.normalize_model_id(item["model_id"]),
                    source=f"static_{self.config.name}",
                    metric_name=item["metric"],
                    metric_value=float(item["value"]),
                    unit=item.get("unit"),
                    context=item.get("context", {})
                ))
                
        except Exception as e:
            self.logger.error(f"Error reading static file {self.config.url}: {e}")
            raise
            
        return entries
    
    def normalize_model_id(self, raw_id: str) -> str:
        """Normalize model ID."""
        return raw_id.lower().strip().replace("-", "_").replace(" ", "_")


class BenchmarkFetcher:
    """
    Main orchestrator for fetching benchmark data from multiple sources.
    
    Usage:
        fetcher = BenchmarkFetcher(config_path="config/benchmark_sources.yaml")
        await fetcher.initialize()
        cache = await fetcher.fetch_all()
    """
    
    FETCHER_REGISTRY = {
        "artificial_analysis": ArtificialAnalysisFetcher,
        "lmsys_arena": LMSYSArenaFetcher,
        "static": StaticFileFetcher,
    }
    
    def __init__(
        self, 
        config_path: Optional[Path] = None,
        cache_path: Optional[Path] = None,
        sources: Optional[List[BenchmarkSourceConfig]] = None
    ):
        self.config_path = config_path or Path("config/benchmark_sources.yaml")
        self.cache_path = cache_path or Path("cache/benchmark_cache.json")
        self.sources = sources or []
        self.fetchers: List[Fetcher] = []
        self.cache = BenchmarkCache()
        self._session: Optional[aiohttp.ClientSession] = None
        
    def register_fetcher(self, name: str, fetcher_class: type[Fetcher]) -> None:
        """Register a custom fetcher type."""
        self.FETCHER_REGISTRY[name] = fetcher_class
        
    async def initialize(self) -> None:
        """Initialize fetchers from config."""
        # Load config if exists
        if self.config_path.exists():
            with open(self.config_path) as f:
                config_data = yaml.safe_load(f)
                for source_config in config_data.get("sources", []):
                    self.sources.append(BenchmarkSourceConfig(**source_config))
        
        # Initialize fetcher instances
        for source in self.sources:
            if not source.enabled:
                logger.info(f"Skipping disabled source: {source.name}")
                continue
                
            fetcher_class = self.FETCHER_REGISTRY.get(source.name)
            if fetcher_class:
                self.fetchers.append(fetcher_class(source))
                logger.info(f"Initialized fetcher for {source.name}")
            else:
                logger.warning(f"Unknown source type: {source.name}")
        
        # Create aiohttp session
        if not self._session:
            self._session = aiohttp.ClientSession()
            
        # Load existing cache if available
        self._load_cache()
    
    async def fetch_all(self, force: bool = False) -> BenchmarkCache:
        """
        Fetch benchmarks from all enabled sources concurrently.
        
        Args:
            force: Force refresh even if cache is fresh
            
        Returns:
            BenchmarkCache with all entries
        """
        if not self.fetchers:
            logger.warning("No fetchers initialized")
            return self.cache
        
        # Check if cache is still fresh (unless forced)
        if not force and self.cache.last_updated:
            age = datetime.utcnow() - self.cache.last_updated
            if age < timedelta(minutes=60):  # Default 1 hour
                logger.info("Using cached benchmark data")
                return self.cache
        
        # Fetch from all sources concurrently
        tasks = []
        for fetcher in self.fetchers:
            task = self._fetch_with_error_handling(fetcher)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        all_entries = []
        for i, result in enumerate(results):
            source_name = self.fetchers[i].config.name
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch from {source_name}: {result}")
                self.cache.source_status[source_name] = f"error: {str(result)}"
            else:
                all_entries.extend(result)
                self.cache.source_status[source_name] = "ok"
                logger.info(f"Fetched {len(result)} entries from {source_name}")
        
        # Update cache
        self.cache.entries = all_entries
        self.cache.last_updated = datetime.utcnow()
        
        # Persist cache
        self._save_cache()
        
        return self.cache
    
    async def _fetch_with_error_handling(self, fetcher: Fetcher) -> List[BenchmarkEntry]:
        """Wrap fetcher with individual error handling."""
        try:
            if not self._session:
                raise BenchmarkFetcherError("Session not initialized")
            return await fetcher.fetch(self._session)
        except Exception as e:
            logger.error(f"Fetcher {fetcher.config.name} failed: {e}")
            raise
    
    def get_model_scores(
        self, 
        model_id: str, 
        metric_filter: Optional[Set[str]] = None
    ) -> Dict[str, float]:
        """
        Get aggregated scores for a specific model.
        
        Args:
            model_id: The model identifier
            metric_filter: Optional set of metric names to include
            
        Returns:
            Dictionary of metric_name -> value
        """
        scores = {}
        for entry in self.cache.entries:
            if entry.model_id == model_id:
                if metric_filter is None or entry.metric_name in metric_filter:
                    # Use source as namespace to avoid collisions
                    key = f"{entry.source}/{entry.metric_name}"
                    scores[key] = entry.metric_value
        return scores
    
    def get_leaderboard(
        self, 
        metric_name: str = "elo_rating", 
        source: Optional[str] = None
    ) -> List[BenchmarkEntry]:
        """Get sorted leaderboard for a specific metric."""
        entries = [
            e for e in self.cache.entries 
            if e.metric_name == metric_name and (source is None or e.source == source)
        ]
        return sorted(entries, key=lambda x: x.metric_value, reverse=True)
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    data = json.load(f)
                    self.cache = BenchmarkCache(**data)
                    logger.info(f"Loaded {len(self.cache.entries)} cached entries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = BenchmarkCache()
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'w') as f:
                # Convert to dict for JSON serialization
                cache_dict = self.cache.dict()
                json.dump(cache_dict, f, indent=2, default=str)
            logger.info(f"Saved cache to {self.cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    async def close(self) -> None:
        """Cleanup resources."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Convenience functions for integration with existing codebase

async def fetch_latest_benchmarks(
    config_path: Optional[Path] = None,
    cache_path: Optional[Path] = None
) -> BenchmarkCache:
    """
    One-shot function to fetch latest benchmarks.
    
    Usage:
        cache = await fetch_latest_benchmarks()
        scores = cache.get_by_model("gpt_4")
    """
    async with BenchmarkFetcher(config_path=config_path, cache_path=cache_path) as fetcher:
        return await fetcher.fetch_all()


def get_fetcher_instance(
    config_path: Optional[Path] = None
) -> BenchmarkFetcher:
    """Get a configured fetcher instance (requires manual initialization)."""
    return BenchmarkFetcher(config_path=config_path)


# Example configuration file content (for reference)
EXAMPLE_CONFIG = """
sources:
  - name: artificial_analysis
    enabled: true
    url: https://api.artificialanalysis.ai/v1/models
    refresh_interval_minutes: 60
    priority: 100
  
  - name: lmsys_arena
    enabled: true
    url: https://chat.lmsys.org/api/leaderboard
    refresh_interval_minutes: 120
    priority: 90
  
  - name: static
    enabled: true
    url: config/static_benchmarks.yaml
    priority: 50
"""
