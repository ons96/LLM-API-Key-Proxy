"""
Provider Priority Manager - Phase 2.2 Implementation

This module provides the core priority tier system for provider fallback routing.
It enables automatic fallback from higher-priority providers to lower-priority ones
when primary providers fail, with G4F serving as the primary fallback mechanism.
"""

import os
import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import IntEnum

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class ProviderTier(IntEnum):
    """Provider priority tiers (1=highest priority, N=lowest priority)."""
    PREMIUM = 1          # OpenAI, Anthropic - Premium paid providers
    FAST_AFFORDABLE = 2  # Groq, OpenRouter - Fast/affordable providers  
    STANDARD = 3         # Gemini, Mistral - Standard providers
    FALLBACK = 4         # G4F providers - Primary fallback providers
    LOWEST = 5           # Default lowest priority


@dataclass
class ProviderPriorityInfo:
    """Information about a provider's priority tier."""
    provider_name: str
    tier: ProviderTier
    tier_name: str
    is_fallback: bool = False
    custom_priority: bool = False


class ProviderPriorityManager:
    """
    Manages provider priority tiers for fallback routing.
    
    This is the core component that enables automatic provider fallback routing.
    G4F providers are assigned to the fallback tier (Tier 4) by default, making
    them automatically used when higher-tier providers fail.
    """
    
    # Default provider tier mappings
    DEFAULT_PRIORITY_TIERS = {
        # Tier 1: Premium paid providers
        "openai": ProviderTier.PREMIUM,
        "anthropic": ProviderTier.PREMIUM,
        
        # Tier 2: Fast/affordable providers
        "groq": ProviderTier.FAST_AFFORDABLE,
        "openrouter": ProviderTier.FAST_AFFORDABLE,
        "cerebras": ProviderTier.FAST_AFFORDABLE,  # Very fast inference, good free tier
        
        # Tier 3: Standard providers
        "gemini": ProviderTier.STANDARD,
        "gemini_cli": ProviderTier.STANDARD,
        "mistral": ProviderTier.STANDARD,
        "cohere": ProviderTier.STANDARD,
        "nvidia_nim": ProviderTier.STANDARD,
        "agentrouter": ProviderTier.STANDARD,  # Third-party with credits
        
        # Tier 4: G4F fallback providers (primary fallback mechanism)
        "g4f": ProviderTier.FALLBACK,
        
        # Tier 5: Lowest priority (default)
        "qwen_code": ProviderTier.LOWEST,
        "iflow": ProviderTier.LOWEST,
        "antigravity": ProviderTier.LOWEST,
        "chutes": ProviderTier.LOWEST,
    }

    
    TIER_NAMES = {
        ProviderTier.PREMIUM: "Premium",
        ProviderTier.FAST_AFFORDABLE: "Fast/Affordable",
        ProviderTier.STANDARD: "Standard",
        ProviderTier.FALLBACK: "Fallback",
        ProviderTier.LOWEST: "Lowest Priority",
    }
    
    def __init__(self, env_vars: Optional[Dict[str, str]] = None):
        """
        Initialize the provider priority manager.
        
        Args:
            env_vars: Environment variables for custom priority configuration
        """
        self.env_vars = env_vars or os.environ
        self._custom_priorities = self._load_custom_priorities()
        self._provider_tiers = self._build_provider_tiers()
        
        lib_logger.info(f"Provider priority manager initialized with {len(self._provider_tiers)} providers")
    
    def _load_custom_priorities(self) -> Dict[str, int]:
        """
        Load custom provider priorities from environment variables.
        
        Environment variables format: PROVIDER_PRIORITY_<PROVIDER_NAME>=<tier_number>
        Examples:
            PROVIDER_PRIORITY_G4F=4
            PROVIDER_PRIORITY_GROQ=2
            PROVIDER_PRIORITY_OPENAI=1
        
        Returns:
            Dictionary mapping provider names to custom priority tiers
        """
        custom_priorities = {}
        
        for key, value in self.env_vars.items():
            if key.startswith("PROVIDER_PRIORITY_"):
                provider_name = key.replace("PROVIDER_PRIORITY_", "").lower()
                try:
                    priority = int(value)
                    if 1 <= priority <= 10:  # Allow 1-10 for flexibility
                        custom_priorities[provider_name] = priority
                        lib_logger.info(f"Custom priority set for {provider_name}: Tier {priority}")
                    else:
                        lib_logger.warning(f"Invalid priority {priority} for {provider_name}. Must be 1-10.")
                except ValueError:
                    lib_logger.warning(f"Invalid priority value '{value}' for {provider_name}")
        
        return custom_priorities
    
    def _build_provider_tiers(self) -> Dict[str, ProviderPriorityInfo]:
        """
        Build the complete provider tier mapping including custom priorities.
        
        Returns:
            Dictionary mapping provider names to priority information
        """
        provider_tiers = {}
        
        # First, add default tiers
        for provider, default_tier in self.DEFAULT_PRIORITY_TIERS.items():
            provider_tiers[provider] = ProviderPriorityInfo(
                provider_name=provider,
                tier=default_tier,
                tier_name=self.TIER_NAMES[default_tier],
                is_fallback=(default_tier == ProviderTier.FALLBACK)
            )
        
        # Override with custom priorities
        for provider, custom_priority in self._custom_priorities.items():
            tier = ProviderTier(custom_priority)
            provider_tiers[provider] = ProviderPriorityInfo(
                provider_name=provider,
                tier=tier,
                tier_name=self.TIER_NAMES.get(tier, f"Tier {custom_priority}"),
                is_fallback=(tier == ProviderTier.FALLBACK),
                custom_priority=True
            )
            lib_logger.info(f"Applied custom priority for {provider}: {tier.name} (Tier {custom_priority})")
        
        return provider_tiers
    
    def get_provider_tier(self, provider_name: str) -> Optional[ProviderPriorityInfo]:
        """
        Get priority information for a specific provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            ProviderPriorityInfo if found, None otherwise
        """
        return self._provider_tiers.get(provider_name.lower())
    
    def get_all_providers_by_tier(self) -> Dict[ProviderTier, List[str]]:
        """
        Get all providers grouped by their priority tier.
        
        Returns:
            Dictionary mapping tiers to lists of provider names
        """
        providers_by_tier = {}
        for provider, info in self._provider_tiers.items():
            if info.tier not in providers_by_tier:
                providers_by_tier[info.tier] = []
            providers_by_tier[info.tier].append(provider)
        
        return providers_by_tier
    
    def get_fallback_providers(self) -> List[str]:
        """
        Get list of providers designated as fallback providers.
        
        Returns:
            List of fallback provider names
        """
        return [
            provider for provider, info in self._provider_tiers.items()
            if info.is_fallback
        ]
    
    def get_priority_ordered_providers(self) -> List[Tuple[str, ProviderPriorityInfo]]:
        """
        Get all providers ordered by priority (highest to lowest).
        
        Returns:
            List of (provider_name, ProviderPriorityInfo) tuples ordered by tier
        """
        return sorted(
            self._provider_tiers.items(),
            key=lambda x: (x[1].tier, x[0])  # Sort by tier first, then name
        )
    
    def is_provider_available(self, provider_name: str, available_providers: List[str]) -> bool:
        """
        Check if a provider is available for routing.
        
        Args:
            provider_name: Name of the provider to check
            available_providers: List of currently available provider names
            
        Returns:
            True if provider is available, False otherwise
        """
        return provider_name.lower() in [p.lower() for p in available_providers]
    
    def get_fallback_chain(
        self, 
        requested_provider: str, 
        available_providers: List[str]
    ) -> List[str]:
        """
        Get the fallback chain for a requested provider.
        
        This is the core fallback mechanism. When a provider fails,
        the system will try providers in order of decreasing priority.
        G4F providers are typically at the end of the chain as the ultimate fallback.
        
        Args:
            requested_provider: The originally requested provider
            available_providers: List of currently available providers
            
        Returns:
            Ordered list of providers to try (fallback chain)
        """
        # Normalize provider names
        requested_provider = requested_provider.lower()
        available_providers = [p.lower() for p in available_providers]
        
        # If requested provider is not available, build full fallback chain
        if requested_provider not in available_providers:
            # Get all providers ordered by priority
            priority_ordered = self.get_priority_ordered_providers()
            
            # Filter to only available providers and respect priority order
            fallback_chain = []
            for provider_name, priority_info in priority_ordered:
                if provider_name in available_providers:
                    fallback_chain.append(provider_name)
            
            lib_logger.debug(f"Fallback chain for {requested_provider}: {fallback_chain}")
            return fallback_chain
        
        # If requested provider is available, return it and its fallbacks
        requested_priority = self.get_provider_tier(requested_provider)
        if not requested_priority:
            return [requested_provider]
        
        # Get providers of equal or lower priority (fallbacks)
        fallback_chain = [requested_provider]
        for provider_name, priority_info in self.get_priority_ordered_providers():
            if (provider_name != requested_provider and 
                provider_name in available_providers and
                priority_info.tier >= requested_priority.tier):
                fallback_chain.append(provider_name)
        
        return fallback_chain
    
    def get_tier_statistics(self) -> Dict[str, int]:
        """
        Get statistics about provider distribution across tiers.
        
        Returns:
            Dictionary mapping tier names to provider counts
        """
        stats = {}
        for info in self._provider_tiers.values():
            tier_name = info.tier_name
            stats[tier_name] = stats.get(tier_name, 0) + 1
        
        return stats
    
    def __repr__(self) -> str:
        """String representation of the priority manager."""
        tier_stats = self.get_tier_statistics()
        stats_str = ", ".join([f"{tier}: {count}" for tier, count in tier_stats.items()])
        return f"ProviderPriorityManager(tiers={{{stats_str}}})"