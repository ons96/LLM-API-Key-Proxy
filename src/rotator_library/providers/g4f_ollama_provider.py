"""G4F Ollama proxy provider - Free access to Ollama models via g4f.space."""

import os
import logging
from typing import List, Dict, Any, Optional
import httpx

from .g4f_provider import G4FProvider

lib_logger = logging.getLogger("rotator_library")


class G4FOllamaProvider(G4FProvider):
    """G4F Ollama proxy - free access to Ollama-hosted models.

    Base URL: https://g4f.space/api/ollama
    Models: deepseek-v3.2, gemma3:27b, qwen3.5:397b, glm-4.7, etc.
    """

    provider_name = "g4f_ollama"
    provider_env_name = "g4f_ollama"
    default_tier_priority = 2  # Higher priority than main G4F

    _DEFAULT_PUBLIC_BASE = "https://g4f.space/api/ollama"

    def __init__(self):
        # No API key required for this endpoint
        self.api_key: Optional[str] = os.getenv("G4F_OLLAMA_API_KEY") or ""
        self._api_base: Optional[str] = os.getenv(
            "G4F_OLLAMA_API_BASE", self._DEFAULT_PUBLIC_BASE
        )

    async def get_models(
        self, api_key: str = None, client: httpx.AsyncClient = None
    ) -> List[str]:
        """Return available models from G4F Ollama proxy.

        Models are prefixed with provider name to avoid duplication with other G4F variants.
        """
        return [
            "g4f_ollama/deepseek-v3.2",
            "g4f_ollama/devstral-small-2:24b",
            "g4f_ollama/gemini-3-flash-preview",
            "g4f_ollama/gemma3:27b",
            "g4f_ollama/qwen3.5:397b",
            "g4f_ollama/glm-4.7",
            "g4f_ollama/gpt-oss:120b",
            "g4f_ollama/minimax-m2",
            "g4f_ollama/minimax-m2.5",
            "g4f_ollama/rnj-1:8b",
        ]
