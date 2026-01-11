# src/rotator_library/provider_factory.py

import os
import logging
from typing import Dict, Type, Optional, Any

from .providers.gemini_auth_base import GeminiAuthBase
from .providers.qwen_auth_base import QwenAuthBase
from .providers.iflow_auth_base import IFlowAuthBase
from .providers.antigravity_auth_base import AntigravityAuthBase
from .providers.g4f_provider import G4FProvider
from .providers.agentrouter_provider import AgentRouterProvider
from .providers.cerebras_provider import CerebrasProvider
from .providers.puter_provider import PuterProvider

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

# Authentication-based providers (require OAuth credential management)
OAUTH_PROVIDER_MAP = {
    "gemini_cli": GeminiAuthBase,
    "qwen_code": QwenAuthBase,
    "iflow": IFlowAuthBase,
    "antigravity": AntigravityAuthBase,
    "puter": PuterProvider,
}

# Direct provider implementations (use API keys or environment config)
DIRECT_PROVIDER_MAP = {
    "g4f": G4FProvider,
    "agentrouter": AgentRouterProvider,
    "cerebras": CerebrasProvider,
    "puter": PuterProvider,
}

# Combined provider map for compatibility
PROVIDER_MAP = {**OAUTH_PROVIDER_MAP, **DIRECT_PROVIDER_MAP}


def get_provider_auth_class(provider_name: str):
    """
    Returns the authentication class for a given provider.

    Note: Only returns classes for OAuth-based providers that require
    credential management. Direct providers like G4F don't use auth classes.
    """
    provider_class = OAUTH_PROVIDER_MAP.get(provider_name.lower())
    if not provider_class:
        raise ValueError(f"Unknown OAuth provider: {provider_name}")
    return provider_class


def get_provider_class(provider_name: str) -> Type:
    """
    Returns the provider class for a given provider name.

    This includes both OAuth-based and direct provider implementations.
    """
    provider_class = PROVIDER_MAP.get(provider_name.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider_name}")
    return provider_class


def is_oauth_provider(provider_name: str) -> bool:
    """
    Check if a provider uses OAuth authentication.
    """
    return provider_name.lower() in OAUTH_PROVIDER_MAP


def is_direct_provider(provider_name: str) -> bool:
    """
    Check if a provider is a direct implementation (like G4F).
    """
    return provider_name.lower() in DIRECT_PROVIDER_MAP


def get_provider_config(provider_name: str) -> Dict[str, Any]:
    """
    Get provider configuration from environment variables.

    Args:
        provider_name: Name of the provider (e.g., 'g4f', 'gemini_cli')

    Returns:
        Dictionary containing provider configuration
    """
    provider_name = provider_name.lower()
    config = {}

    if provider_name == "g4f":
        # G4F doesn't require specific configuration - it can work with defaults
        config.update(
            {
                "api_base": os.getenv("G4F_MAIN_API_BASE"),
                "default_tier_priority": 5,  # G4F is fallback tier
            }
        )
        lib_logger.debug(f"Loaded G4F configuration: {list(config.keys())}")

    elif provider_name == "puter":
        # Puter.js provider via puter-free-chatbot wrapper
        config.update(
            {
                "api_base": os.getenv(
                    "PUTER_API_BASE", "https://puter-free-chatbot.vercel.app/api"
                ),
                # Default to your deployed instance, can be overridden
                "default_tier_priority": 2,  # High priority for free models
            }
        )
        lib_logger.debug(f"Loaded Puter configuration: {list(config.keys())}")

    return config


def get_available_providers():
    """
    Returns a list of available provider names.
    """
    return list(PROVIDER_MAP.keys())


def get_oauth_providers():
    """
    Returns a list of OAuth-based provider names.
    """
    return list(OAUTH_PROVIDER_MAP.keys())


def get_direct_providers():
    """
    Returns a list of direct provider implementation names.
    """
    return list(DIRECT_PROVIDER_MAP.keys())


def validate_provider_config(provider_name: str) -> bool:
    """
    Validate that required configuration exists for a provider.

    Args:
        provider_name: Name of the provider to validate

    Returns:
        True if configuration is valid, False otherwise
    """
    provider_name = provider_name.lower()

    if provider_name == "g4f":
        # G4F doesn't require specific configuration - it can work with defaults
        # or with any of the optional API base URLs
        config = get_provider_config(provider_name)
        return True  # G4F is always configurable

    # For OAuth providers, validation is handled by their auth classes
    return provider_name in OAUTH_PROVIDER_MAP
