"""G4F Pollinations proxy provider - Free access to Pollinations.ai models."""

import os
import logging
from typing import List, Dict, Any, Optional

from .g4f_provider import G4FProvider

lib_logger = logging.getLogger("rotator_library")


class G4FPollinationsProvider(G4FProvider):
    """G4F Pollinations proxy - free access to Pollinations.ai models.

    Base URL: https://g4f.space/api/pollinations
    Models: openai, openai-fast, qwen-coder, mistral, deepseek, gemini-search, etc.
    """

    provider_name = "g4f_pollinations"
    provider_env_name = "g4f_pollinations"
    default_tier_priority = 2  # Higher priority than main G4F

    _DEFAULT_PUBLIC_BASE = "https://g4f.space/api/pollinations"

    def __init__(self):
        # No API key required for this endpoint
        self.api_key: Optional[str] = os.getenv("G4F_POLLINATIONS_API_KEY") or ""
        self._api_base: Optional[str] = os.getenv(
            "G4F_POLLINATIONS_API_BASE", self._DEFAULT_PUBLIC_BASE
        )

    async def get_models(self) -> List[Dict[str, Any]]:
        """Return available models from G4F Pollinations proxy."""
        return [
            {"id": "openai", "object": "model", "owned_by": "openai"},
            {"id": "openai-fast", "object": "model", "owned_by": "openai"},
            {"id": "qwen-coder", "object": "model", "owned_by": "alibaba"},
            {"id": "mistral", "object": "model", "owned_by": "mistral"},
            {"id": "openai-audio", "object": "model", "owned_by": "openai"},
            {"id": "gemini-fast", "object": "model", "owned_by": "google"},
            {"id": "deepseek", "object": "model", "owned_by": "deepseek"},
            {"id": "gemini-search", "object": "model", "owned_by": "google"},
            {"id": "midijourney", "object": "model", "owned_by": "midijourney"},
            {"id": "claude-fast", "object": "model", "owned_by": "anthropic"},
        ]
