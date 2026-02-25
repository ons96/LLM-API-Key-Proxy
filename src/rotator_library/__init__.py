"""
Rotator Library - LLM Provider Abstraction Layer

Provides a unified interface for interacting with multiple LLM providers
with automatic credential rotation, usage tracking, and rate limit handling.

Key Components:
    RotatingClient: Main client for making LLM requests with automatic failover
    PROVIDER_PLUGINS: Registry of available provider adapters
    ModelInfoService: Service for retrieving model metadata

Providers are auto-discovered from the providers/ package and registered
in PROVIDER_PLUGINS dict on import.

Example:
    from rotator_library import RotatingClient
    client = RotatingClient()
    response = await client.chat_completions(...)

"""

from typing import TYPE_CHECKING, Dict, Type

from .client import RotatingClient

# For type checkers (Pylint, mypy), import PROVIDER_PLUGINS statically
# At runtime, it's lazy-loaded via __getattr__
if TYPE_CHECKING:
    from .providers import PROVIDER_PLUGINS
    from .providers.provider_interface import ProviderInterface
    from .model_info_service import ModelInfoService, ModelInfo, ModelMetadata

__all__ = [
    "RotatingClient",
    "PROVIDER_PLUGINS",
    "ModelInfoService",
    "ModelInfo",
    "ModelMetadata",
]


def __getattr__(name):
    """Lazy-load PROVIDER_PLUGINS and ModelInfoService to speed up module import."""
    if name == "PROVIDER_PLUGINS":
        from .providers import PROVIDER_PLUGINS

        return PROVIDER_PLUGINS
    if name == "ModelInfoService":
        from .model_info_service import ModelInfoService

        return ModelInfoService
    if name == "ModelInfo":
        from .model_info_service import ModelInfo

        return ModelInfo
    if name == "ModelMetadata":
        from .model_info_service import ModelMetadata

        return ModelMetadata
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
