#!/usr/bin/env python3
"""
Pricing Data Module

Handles collection, storage, and retrieval of LLM provider pricing data.
Part of Phase 2.1 Data Collection Pipeline.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import threading
import yaml
import os

logger = logging.getLogger(__name__)


@dataclass
class ModelPricing:
    """Pricing information for a specific model."""
    provider: str
    model_name: str
    input_price_per_million: float  # Price per 1M input tokens
    output_price_per_million: float  # Price per 1M output tokens
    currency: str = "USD"
    context_window: int = 0  # Context window size in tokens
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['updated_at'] = self.updated_at.isoformat()
        return data


class PricingDatabase:
    """
    Thread-safe pricing data storage and retrieval.
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self._pricing: Dict[str, Dict[str, ModelPricing]] = {}  # provider -> model -> pricing
        self._last_updated: Optional[datetime] = None
        self._source: str = "memory"
        
    def add_pricing(self, pricing: ModelPricing) -> None:
        """Add or update pricing for a model."""
        with self._lock:
            provider = pricing.provider.lower()
            model = pricing.model_name.lower()
            
            if provider not in self._pricing:
                self._pricing[provider] = {}
                
            self._pricing[provider][model] = pricing
            self._last_updated = datetime.utcnow()
            logger.debug(f"Added/updated pricing for {provider}/{model}")
    
    def add_batch(self, pricing_list: List[ModelPricing]) -> None:
        """Add multiple pricing entries at once."""
        with self._lock:
            for pricing in pricing_list:
                self.add_pricing(pricing)
    
    def get_pricing(self, provider: str, model: str) -> Optional[ModelPricing]:
        """Get pricing for a specific provider and model."""
        with self._lock:
            provider = provider.lower()
            model = model.lower()
            
            if provider in self._pricing and model in self._pricing[provider]:
                return self._pricing[provider][model]
            return None
    
    def get_provider_pricing(self, provider: str) -> List[ModelPricing]:
        """Get all pricing data for a specific provider."""
        with self._lock:
            provider = provider.lower()
            if provider in self._pricing:
                return list(self._pricing[provider].values())
            return []
    
    def get_all_pricing(self) -> Dict[str, List[ModelPricing]]:
        """Get all pricing data grouped by provider."""
        with self._lock:
            return {provider: list(models.values()) 
                    for provider, models in self._pricing.items()}
    
    def get_all_as_list(self) -> List[ModelPricing]:
        """Get all pricing data as a flat list."""
        with self._lock:
            all_pricing = []
            for models in self._pricing.values():
                all_pricing.extend(models.values())
            return all_pricing
    
    def search_by_model(self, model_pattern: str) -> List[ModelPricing]:
        """Search for models matching a pattern (case-insensitive)."""
        with self._lock:
            pattern = model_pattern.lower()
            results = []
            for models in self._pricing.values():
                for pricing in models.values():
                    if pattern in pricing.model_name.lower():
                        results.append(pricing)
            return results
    
    def get_cheapest_models(self, limit: int = 10, input_only: bool = False) -> List[ModelPricing]:
        """Get the cheapest models by output price (or input if input_only=True)."""
        with self._lock:
            all_pricing = self.get_all_as_list()
            
            if not all_pricing:
                return []
            
            if input_only:
                sorted_pricing = sorted(all_pricing, key=lambda x: x.input_price_per_million)
            else:
                sorted_pricing = sorted(all_pricing, key=lambda x: x.output_price_per_million)
            
            return sorted_pricing[:limit]
    
    def get_providers(self) -> List[str]:
        """Get list of all providers with pricing data."""
        with self._lock:
            return list(self._pricing.keys())
    
    def get_model_count(self) -> int:
        """Get total number of models with pricing data."""
        with self._lock:
            return sum(len(models) for models in self._pricing.values())
    
    def clear(self) -> None:
        """Clear all pricing data."""
        with self._lock:
            self._pricing.clear()
            self._last_updated = None
    
    def get_last_updated(self) -> Optional[datetime]:
        """Get timestamp of last update."""
        with self._lock:
            return self._last_updated
    
    def set_source(self, source: str) -> None:
        """Set the source of pricing data."""
        with self._lock:
            self._source = source
    
    def get_source(self) -> str:
        """Get the source of pricing data."""
        with self._lock:
            return self._source


class PricingCollector:
    """
    Collects pricing data from various sources.
    """
    
    def __init__(self, pricing_db: Optional[PricingDatabase] = None):
        self.pricing_db = pricing_db or PricingDatabase()
        self._load_default_pricing()
    
    def _load_default_pricing(self) -> None:
        """Load default pricing data from embedded knowledge."""
        # Common provider pricing (as of 2024)
        default_pricing = [
            # OpenAI
            ModelPricing(provider="openai", model_name="gpt-4o", 
                        input_price_per_million=5.0, output_price_per_million=15.0,
                        context_window=128000),
            ModelPricing(provider="openai", model_name="gpt-4o-mini",
                        input_price_per_million=0.15, output_price_per_million=0.60,
                        context_window=128000),
            ModelPricing(provider="openai", model_name="gpt-4-turbo",
                        input_price_per_million=10.0, output_price_per_million=30.0,
                        context_window=128000),
            ModelPricing(provider="openai", model_name="gpt-3.5-turbo",
                        input_price_per_million=0.50, output_price_per_million=1.50,
                        context_window=16385),
            
            # Anthropic
            ModelPricing(provider="anthropic", model_name="claude-3-5-sonnet",
                        input_price_per_million=3.0, output_price_per_million=15.0,
                        context_window=200000),
            ModelPricing(provider="anthropic", model_name="claude-3-opus",
                        input_price_per_million=15.0, output_price_per_million=75.0,
                        context_window=200000),
            ModelPricing(provider="anthropic", model_name="claude-3-haiku",
                        input_price_per_million=0.25, output_price_per_million=1.25,
                        context_window=200000),
            
            # Google
            ModelPricing(provider="google", model_name="gemini-1.5-pro",
                        input_price_per_million=1.25, output_price_per_million=5.0,
                        context_window=2000000),
            ModelPricing(provider="google", model_name="gemini-1.5-flash",
                        input_price_per_million=0.075, output_price_per_million=0.30,
                        context_window=1000000),
            ModelPricing(provider="google", model_name="gemini-1.5-flash-8b",
                        input_price_per_million=0.0375, output_price_per_million=0.15,
                        context_window=1000000),
            
            # Meta
            ModelPricing(provider="meta", model_name="llama-3.1-405b",
                        input_price_per_million=3.5, output_price_per_million=3.5,
                        context_window=128000),
            ModelPricing(provider="meta", model_name="llama-3.1-70b",
                        input_price_per_million=0.88, output_price_per_million=0.88,
                        context_window=128000),
            ModelPricing(provider="meta", model_name="llama-3.1-8b",
                        input_price_per_million=0.22, output_price_per_million=0.22,
                        context_window=128000),
            
            # Mistral
            ModelPricing(provider="mistral", model_name="mistral-large",
                        input_price_per_million=2.0, output_price_per_million=6.0,
                        context_window=128000),
            ModelPricing(provider="mistral", model_name="mistral-small",
                        input_price_per_million=0.20, output_price_per_million=0.60,
                        context_window=128000),
            
            # Cohere
            ModelPricing(provider="cohere", model_name="command-r-plus",
                        input_price_per_million=3.0, output_price_per_million=15.0,
                        context_window=128000),
            ModelPricing(provider="cohere", model_name="command-r",
                        input_price_per_million=0.50, output_price_per_million=1.50,
                        context_window=128000),
            
            # Groq
            ModelPricing(provider="groq", model_name="llama-3.1-70b-versatile",
                        input_price_per_million=0.59, output_price_per_million=0.79,
                        context_window=8192),
            ModelPricing(provider="groq", model_name="mixtral-8x7b-32768",
                        input_price_per_million=0.24, output_price_per_million=0.24,
                        context_window=32768),
            
            # Cerebras
            ModelPricing(provider="cerebras", model_name="llama-3.1-70b",
                        input_price_per_million=0.10, output_price_per_million=0.10,
                        context_window=128000),
            
            # xAI
            ModelPricing(provider="xai", model_name="grok-beta",
                        input_price_per_million=5.0, output_price_per_million=15.0,
                        context_window=131072),
            
            # Perplexity
            ModelPricing(provider="perplexity", model_name="llama-3.1-sonar-large",
                        input_price_per_million=1.0, output_price_per_million=1.0,
                        context_window=127072),
            ModelPricing(provider="perplexity", model_name="llama-3.1-sonar-small",
                        input_price_per_million=0.20, output_price_per_million=0.20,
                        context_window=127072),
            
            # DeepSeek
            ModelPricing(provider="deepseek", model_name="deepseek-chat",
                        input_price_per_million=0.14, output_price_per_million=0.28,
                        context_window=64000),
            ModelPricing(provider="deepseek", model_name="deepseek-coder",
                        input_price_per_million=0.14, output_price_per_million=0.28,
                        context_window=64000),
        ]
        
        self.pricing_db.add_batch(default_pricing)
        self.pricing_db.set_source("default")
        logger.info(f"Loaded {len(default_pricing)} default pricing entries")
    
    def load_from_config(self, config_path: str) -> int:
        """Load pricing data from a YAML config file."""
        if not os.path.exists(config_path):
            logger.warning(f"Pricing config not found: {config_path}")
            return 0
        
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            
            if not data or 'pricing' not in data:
                logger.warning(f"No pricing data in config: {config_path}")
                return 0
            
            pricing_list = []
            for entry in data['pricing']:
                pricing = ModelPricing(
                    provider=entry.get('provider', ''),
                    model_name=entry.get('model', entry.get('model_name', '')),
                    input_price_per_million=entry.get('input_price_per_million', 0),
                    output_price_per_million=entry.get('output_price_per_million', 0),
                    currency=entry.get('currency', 'USD'),
                    context_window=entry.get('context_window', 0),
                    metadata=entry.get('metadata', {})
                )
                pricing_list.append(pricing)
            
            self.pricing_db.add_batch(pricing_list)
            self.pricing_db.set_source(config_path)
            logger.info(f"Loaded {len(pricing_list)} pricing entries from {config_path}")
            return len(pricing_list)
            
        except Exception as e:
            logger.error(f"Failed to load pricing config: {e}")
            return 0
    
    def add_custom_pricing(self, provider: str, model: str, 
                          input_price: float, output_price: float,
                          context_window: int = 0, currency: str = "USD") -> None:
        """Add custom pricing for a model."""
        pricing = ModelPricing(
            provider=provider,
            model_name=model,
            input_price_per_million=input_price,
            output_price_per_million=output_price,
            context_window=context_window,
            currency=currency
        )
        self.pricing_db.add_pricing(pricing)
        logger.info(f"Added custom pricing for {provider}/{model}")
    
    def refresh(self) -> None:
        """Refresh pricing data from configured sources."""
        # This can be extended to fetch from external APIs
        logger.info("Pricing data refresh triggered")


# Global instance for easy access
_global_pricing_collector: Optional[PricingCollector] = None
_global_pricing_db: Optional[PricingDatabase] = None


def get_pricing_collector() -> PricingCollector:
    """Get or create the global pricing collector instance."""
    global _global_pricing_collector
    if _global_pricing_collector is None:
        _global_pricing_collector = PricingCollector()
    return _global_pricing_collector


def get_pricing_database() -> PricingDatabase:
    """Get or create the global pricing database instance."""
    global _global_pricing_db
    if _global_pricing_db is None:
        collector = get_pricing_collector()
        _global_pricing_db = collector.pricing_db
    return _global_pricing_db
