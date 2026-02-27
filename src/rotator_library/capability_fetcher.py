"""
Capability Fetcher for Phase 2.1 Data Collection Pipeline.

Fetches and aggregates model capabilities from multiple sources:
- LiteLLM model registry
- Provider-specific APIs
- Static configuration files
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
import aiohttp
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelCapabilities:
    """Standardized model capabilities data structure."""
    tool_choice: bool = False
    function_calling: bool = False
    reasoning: bool = False
    vision: bool = False
    system_messages: bool = True
    prompt_caching: bool = False
    assistant_prefill: bool = False
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    context_window: Optional[int] = None
    supported_modalities: List[str] = field(default_factory=lambda: ["text"])
    supported_output_modalities: List[str] = field(default_factory=lambda: ["text"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCapabilities":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CapabilityFetcher:
    """
    Fetches model capabilities from multiple sources and aggregates them.
    Supports caching and incremental updates.
    """
    
    def __init__(self, config_path: Optional[Path] = None, cache_ttl: int = 3600):
        self.config_path = config_path or Path("config")
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, ModelCapabilities] = {}
        self._last_fetch: Optional[float] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        
    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "Mirro-Proxy-Capability-Fetcher/2.1"}
        )
        return self
        
    async function __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
            self._session = None
            
    async def fetch_all_capabilities(self, force_refresh: bool = False) -> Dict[str, ModelCapabilities]:
        """
        Fetch capabilities from all available sources.
        
        Args:
            force_refresh: Bypass cache and fetch fresh data
            
        Returns:
            Dictionary mapping model IDs to their capabilities
        """
        async with self._lock:
            if not force_refresh and self._cache and self._last_fetch:
                import time
                if time.time() - self._last_fetch < self.cache_ttl:
                    logger.debug("Returning cached capabilities")
                    return self._cache.copy()
            
            results: Dict[str, ModelCapabilities] = {}
            
            # Source 1: LiteLLM registry (most comprehensive)
            try:
                litellm_caps = await self._fetch_litellm_capabilities()
                results.update(litellm_caps)
                logger.debug(f"Fetched {len(litellm_caps)} capabilities from LiteLLM")
            except Exception as e:
                logger.warning(f"Failed to fetch LiteLLM capabilities: {e}")
            
            # Source 2: Provider-specific implementations
            try:
                provider_caps = await self._fetch_provider_capabilities()
                # Merge provider capabilities (provider-specific overrides general)
                for model_id, caps in provider_caps.items():
                    if model_id in results:
                        results[model_id] = self._merge_capabilities(results[model_id], caps)
                    else:
                        results[model_id] = caps
                logger.debug(f"Fetched {len(provider_caps)} capabilities from providers")
            except Exception as e:
                logger.warning(f"Failed to fetch provider capabilities: {e}")
            
            # Source 3: Static configuration (highest priority)
            try:
                static_caps = self._load_static_capabilities()
                # Merge static capabilities (static config overrides everything)
                for model_id, caps in static_caps.items():
                    if model_id in results:
                        results[model_id] = self._merge_capabilities(results[model_id], caps)
                    else:
                        results[model_id] = caps
                logger.debug(f"Loaded {len(static_caps)} capabilities from static config")
            except Exception as e:
                logger.warning(f"Failed to load static capabilities: {e}")
            
            self._cache = results
            import time
            self._last_fetch = time.time()
            
            logger.info(f"Capability fetcher updated cache with {len(results)} models")
            return results.copy()
    
    async def _fetch_litellm_capabilities(self) -> Dict[str, ModelCapabilities]:
        """Fetch capabilities from LiteLLM's model cost and info registry."""
        capabilities = {}
        
        try:
            import litellm
            from litellm import get_model_info
            
            # Get model list from LiteLLM
            model_list = getattr(litellm, 'model_list', []) or []
            
            # Also try to get from model_cost dictionary
            model_cost = getattr(litellm, 'model_cost', {}) or {}
            all_models = set(model_list) | set(model_cost.keys())
            
            for model_name in all_models:
                try:
                    info = get_model_info(model_name, throw_error=False)
                    if not info:
                        continue
                        
                    caps = ModelCapabilities(
                        tool_choice=info.get("supports_function_calling", False) or info.get("tool_choice", False),
                        function_calling=info.get("supports_function_calling", False),
                        reasoning=info.get("supports_reasoning", False),
                        vision=info.get("supports_vision", False),
                        system_messages=info.get("supports_system_messages", True),
                        prompt_caching=info.get("supports_prompt_caching", False),
                        assistant_prefill=info.get("supports_assistant_prefill", False),
                        max_input_tokens=info.get("max_input_tokens") or info.get("max_tokens"),
                        max_output_tokens=info.get("max_output_tokens") or info.get("max_tokens"),
                        context_window=info.get("context_window") or info.get("max_tokens"),
                        supported_modalities=info.get("modalities", ["text"]) if isinstance(info.get("modalities"), list) else ["text"],
                        supported_output_modalities=info.get("output_modalities", ["text"]) if isinstance(info.get("output_modalities"), list) else ["text"],
                    )
                    capabilities[model_name] = caps
                    
                except Exception as e:
                    logger.debug(f"Could not parse LiteLLM info for {model_name}: {e}")
                    continue
                    
        except ImportError:
            logger.warning("LiteLLM not available for capability fetching")
        except Exception as e:
            logger.error(f"Error fetching LiteLLM capabilities: {e}")
            
        return capabilities
    
    async def _fetch_provider_capabilities(self) -> Dict[str, ModelCapabilities]:
        """Fetch capabilities from registered provider plugins."""
        capabilities = {}
        
        try:
            from rotator_library.providers import PROVIDER_PLUGINS
            
            for provider_name, provider_class in PROVIDER_PLUGINS.items():
                try:
                    # Check if provider implements capability discovery
                    if hasattr(provider_class, 'fetch_capabilities'):
                        if asyncio.iscoroutinefunction(provider_class.fetch_capabilities):
                            provider_caps = await provider_class.fetch_capabilities()
                        else:
                            provider_caps = provider_class.fetch_capabilities()
                            
                        for model_id, caps_data in provider_caps.items():
                            if isinstance(caps_data, dict):
                                capabilities[model_id] = ModelCapabilities.from_dict(caps_data)
                            elif isinstance(caps_data, ModelCapabilities):
                                capabilities[model_id] = caps_data
                                
                    elif hasattr(provider_class, 'get_model_capabilities'):
                        # Alternative method name
                        provider_caps = provider_class.get_model_capabilities()
                        capabilities.update(provider_caps)
                        
                except Exception as e:
                    logger.debug(f"Provider {provider_name} capability fetch failed: {e}")
                    continue
                    
        except ImportError:
            logger.warning("Provider plugins not available")
        except Exception as e:
            logger.error(f"Error fetching provider capabilities: {e}")
            
        return capabilities
    
    def _load_static_capabilities(self) -> Dict[str, ModelCapabilities]:
        """Load capabilities from static YAML configuration."""
        capabilities = {}
        
        cap_file = self.config_path / "model_capabilities.yaml"
        if not cap_file.exists():
            logger.debug(f"Static capabilities file not found: {cap_file}")
            return capabilities
            
        try:
            with open(cap_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                
            models_data = data.get("models", {})
            for model_id, caps_data in models_data.items():
                if not isinstance(caps_data, dict):
                    continue
                    
                # Filter only valid fields
                valid_fields = {k: v for k, v in caps_data.items() 
                              if k in ModelCapabilities.__dataclass_fields__}
                capabilities[model_id] = ModelCapabilities(**valid_fields)
                
        except yaml.YAMLError as e:
            logger.error(f"Error parsing capabilities YAML: {e}")
        except Exception as e:
            logger.error(f"Error loading static capabilities: {e}")
            
        return capabilities
    
    def _merge_capabilities(self, base: ModelCapabilities, override: ModelCapabilities) -> ModelCapabilities:
        """
        Merge two capability objects, with override taking precedence for non-default values.
        """
        base_dict = base.to_dict()
        override_dict = override.to_dict()
        
        merged = {}
        for key in base_dict:
            override_val = override_dict.get(key)
            base_val = base_dict[key]
            
            # Use override if it's different from default and not None
            if override_val is not None and override_val != ModelCapabilities().__dataclass_fields__[key].default:
                merged[key] = override_val
            else:
                merged[key] = base_val
                
        return ModelCapabilities(**merged)
    
    def get_capabilities(self, model_id: str) -> Optional[ModelCapabilities]:
        """Get cached capabilities for a specific model."""
        return self._cache.get(model_id)
    
    def get_models_with_capability(self, capability_name: str) -> List[str]:
        """Get all model IDs that have a specific capability enabled."""
        models = []
        for model_id, caps in self._cache.items():
            if getattr(caps, capability_name, False):
                models.append(model_id)
        return models
    
    async def refresh_model(self, model_id: str) -> Optional[ModelCapabilities]:
        """Refresh capabilities for a single model."""
        # This would fetch from specific endpoints if available
        # For now, just refresh all
        all_caps = await self.fetch_all_capabilities(force_refresh=True)
        return all_caps.get(model_id)
    
    def invalidate_cache(self):
        """Clear the capability cache."""
        self._cache.clear()
        self._last_fetch = None
        logger.info("Capability cache invalidated")
