"""G4F Nvidia proxy provider - Free access to Nvidia NIM models via g4f.space."""

import os
import logging
from typing import List, Dict, Any, Optional
import httpx

from .g4f_provider import G4FProvider

lib_logger = logging.getLogger("rotator_library")


class G4FNvidiaProvider(G4FProvider):
    """G4F Nvidia proxy - free access to Nvidia NIM models.

    Base URL: https://g4f.space/api/nvidia
    Models: 170+ models including llama-3.3-70b, deepseek-v3.1, gemma-3-27b, etc.
    """

    provider_name = "g4f_nvidia"
    provider_env_name = "g4f_nvidia"
    default_tier_priority = 1  # Highest priority - most models

    _DEFAULT_PUBLIC_BASE = "https://g4f.space/api/nvidia"

    def __init__(self):
        # No API key required for this endpoint
        self.api_key: Optional[str] = os.getenv("G4F_NVIDIA_API_KEY") or ""
        self._api_base: Optional[str] = os.getenv(
            "G4F_NVIDIA_API_BASE", self._DEFAULT_PUBLIC_BASE
        )

    async def get_models(
        self, api_key: str = None, client: httpx.AsyncClient = None
    ) -> List[str]:
        """Return available models from G4F Nvidia proxy.

        Models are prefixed with provider name to avoid duplication with other G4F variants.
        """
        # Key free models (170+ total available) - prefixed with provider name
        return [
            # Meta Llama models
            "g4f_nvidia/meta/llama-3.1-405b-instruct",
            "g4f_nvidia/meta/llama-3.3-70b-instruct",
            "g4f_nvidia/meta/llama-4-maverick-17b-128e-instruct",
            "g4f_nvidia/meta/llama-4-scout-17b-16e-instruct",
            # DeepSeek models
            "g4f_nvidia/deepseek-ai/deepseek-v3.1",
            "g4f_nvidia/deepseek-ai/deepseek-v3.2",
            "g4f_nvidia/deepseek-ai/deepseek-r1-distill-llama-70b",
            # Google Gemma models
            "g4f_nvidia/google/gemma-3-27b-it",
            "g4f_nvidia/google/gemma-3-12b-it",
            "g4f_nvidia/google/gemma-3-4b-it",
            # Mistral models
            "g4f_nvidia/mistralai/mistral-large-3-675b",
            "g4f_nvidia/mistralai/codestral-2501",
            # Qwen models
            "g4f_nvidia/qwen/qwen3-235b-a22b",
            "g4f_nvidia/qwen/qwen3-coder-480b",
            "g4f_nvidia/qwen/qwen3.5-397b",
            # Nvidia Nemotron
            "g4f_nvidia/nvidia/nemotron-ultra-253b-v1",
            "g4f_nvidia/nvidia/nemotron-super-49b-v1",
            # Microsoft Phi
            "g4f_nvidia/microsoft/phi-4-mini-instruct",
            "g4f_nvidia/microsoft/phi-3.5-mini",
            # Moonshot Kimi
            "g4f_nvidia/moonshotai/kimi-k2-base",
        ]
