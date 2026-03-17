"""G4F Gemini proxy provider - Free access to Gemini models via g4f.space."""

import os
import logging
from typing import List, Dict, Any, Optional
import httpx

from .g4f_provider import G4FProvider

lib_logger = logging.getLogger("rotator_library")


class G4FGeminiProvider(G4FProvider):
    """G4F Gemini proxy - free access to Google Gemini models.

    Base URL: https://g4f.space/api/gemini
    Models: gemini-3-pro, gemini-2.5-flash, gemini-2.5-pro
    """

    provider_name = "g4f_gemini"
    provider_env_name = "g4f_gemini"
    default_tier_priority = 2  # Higher priority than main G4F

    _DEFAULT_PUBLIC_BASE = "https://g4f.space/api/gemini"

    def __init__(self):
        # No API key required for this endpoint
        self.api_key: Optional[str] = os.getenv("G4F_GEMINI_API_KEY") or ""
        self._api_base: Optional[str] = os.getenv(
            "G4F_GEMINI_API_BASE", self._DEFAULT_PUBLIC_BASE
        )

    async def get_models(
        self, api_key: str = None, client: httpx.AsyncClient = None
    ) -> List[str]:
        """Return available models from G4F Gemini proxy.

        Models are prefixed with provider name to avoid duplication with other G4F variants.
        """
        return [
            "g4f_gemini/gemini-3-pro",
            "g4f_gemini/gemini-2.5-flash",
            "g4f_gemini/gemini-2.5-pro",
        ]
