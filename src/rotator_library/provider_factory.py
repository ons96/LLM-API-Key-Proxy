"""
Provider Factory - Updated for Phase 5.1 Replicate support
"""

from typing import Dict, Type, Optional, Any

from .providers.groq_provider import GroqProvider
from .providers.gemini_provider import GeminiProvider
from .providers.cerebras_provider import CerebrasProvider
from .providers.together_provider import TogetherProvider
from .providers.g4f_provider import G4FProvider
from .providers.cohere_provider import CohereProvider
from .providers.chutes_provider import ChutesProvider
from .providers.replicate_provider import ReplicateProvider


class ProviderFactory:
    """Factory for creating provider instances."""
    
    _providers: Dict[str, Type] = {
        "groq": GroqProvider,
        "gemini": GeminiProvider,
        "cerebras": CerebrasProvider,
        "together": TogetherProvider,
        "g4f": G4FProvider,
        "cohere": CohereProvider,
        "chutes": ChutesProvider,
        "replicate": ReplicateProvider,  # Phase 5.1 addition
    }
    
    @classmethod
    def create_provider(
        cls, 
        provider_name: str, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Create a provider instance by name.
        
        Args:
            provider_name: The provider identifier (e.g., 'replicate', 'groq')
            api_key: API key for authentication
            base_url: Optional custom base URL
            **kwargs: Additional provider-specific configuration
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If provider_name is not recognized
        """
        provider_name = provider_name.lower()
        if provider_name not in cls._providers:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available: {list(cls._providers.keys())}"
            )
        
        provider_class = cls._providers[provider_name]
        return provider_class(api_key=api_key, base_url=base_url, **kwargs)
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type):
        """Register a new provider type dynamically."""
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def list_providers(cls) -> list:
        """List all available provider names."""
        return list(cls._providers.keys())
    
    @classmethod
    def is_supported(cls, provider_name: str) -> bool:
        """Check if a provider is supported."""
        return provider_name.lower() in cls._providers
