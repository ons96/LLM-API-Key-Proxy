"""G4F Groq proxy provider - Free access to Groq models via g4f.space."""

import os
import logging
from typing import List, Dict, Any, Optional

from .g4f_provider import G4FProvider

lib_logger = logging.getLogger("rotator_library")


class G4FGroqProvider(G4FProvider):
    """G4F Groq proxy - free access to Groq-hosted models.

    Base URL: https://g4f.space/api/groq
    Models: llama-3.3-70b, llama-3.1-8b, mixtral, gemma, etc.
    """

    provider_name = "g4f_groq"
    provider_env_name = "g4f_groq"
    default_tier_priority = 2  # Higher priority than main G4F

    _DEFAULT_PUBLIC_BASE = "https://g4f.space/api/groq"

    def __init__(self):
        # No API key required for this endpoint
        self.api_key: Optional[str] = os.getenv("G4F_GROQ_API_KEY") or ""
        self._api_base: Optional[str] = os.getenv(
            "G4F_GROQ_API_BASE", self._DEFAULT_PUBLIC_BASE
        )

    async def get_models(self) -> List[Dict[str, Any]]:
        """Return available models from G4F Groq proxy."""
        return [
            {"id": "llama-3.3-70b-versatile", "object": "model", "owned_by": "meta"},
            {"id": "llama-3.1-8b-instant", "object": "model", "owned_by": "meta"},
            {"id": "llama-3.1-70b-versatile", "object": "model", "owned_by": "meta"},
            {"id": "mixtral-8x7b-32768", "object": "model", "owned_by": "mistral"},
            {"id": "gemma2-9b-it", "object": "model", "owned_by": "google"},
        ]
