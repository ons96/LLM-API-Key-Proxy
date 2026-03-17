"""G4F Pollinations proxy provider - Free access to Pollinations.ai models."""

import os
import logging
from typing import List, Dict, Any, Optional
import httpx

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

    async def get_models(
        self, api_key: str = None, client: httpx.AsyncClient = None
    ) -> List[str]:
        """Return available models from G4F Pollinations proxy.

        Models are prefixed with provider name to avoid duplication with other G4F variants.
        """
        return [
            "g4f_pollinations/openai",
            "g4f_pollinations/openai-fast",
            "g4f_pollinations/qwen-coder",
            "g4f_pollinations/mistral",
            "g4f_pollinations/openai-audio",
            "g4f_pollinations/gemini-fast",
            "g4f_pollinations/deepseek",
            "g4f_pollinations/gemini-search",
            "g4f_pollinations/midijourney",
            "g4f_pollinations/claude-fast",
        ]
