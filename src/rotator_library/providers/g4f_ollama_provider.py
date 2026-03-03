"""G4F Ollama proxy provider - Free access to Ollama models via g4f.space."""

import os
import logging
from typing import List, Dict, Any, Optional

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

    async def get_models(self) -> List[Dict[str, Any]]:
        """Return available models from G4F Ollama proxy."""
        return [
            {"id": "deepseek-v3.2", "object": "model", "owned_by": "deepseek"},
            {"id": "devstral-small-2:24b", "object": "model", "owned_by": "mistral"},
            {"id": "gemini-3-flash-preview", "object": "model", "owned_by": "google"},
            {"id": "gemma3:27b", "object": "model", "owned_by": "google"},
            {"id": "qwen3.5:397b", "object": "model", "owned_by": "alibaba"},
            {"id": "glm-4.7", "object": "model", "owned_by": "zhipu"},
            {"id": "gpt-oss:120b", "object": "model", "owned_by": "openai"},
            {"id": "minimax-m2", "object": "model", "owned_by": "minimax"},
            {"id": "minimax-m2.5", "object": "model", "owned_by": "minimax"},
            {"id": "rnj-1:8b", "object": "model", "owned_by": "unknown"},
        ]
