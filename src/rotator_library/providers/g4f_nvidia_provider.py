"""G4F Nvidia proxy provider - Free access to Nvidia NIM models via g4f.space."""

import os
import logging
from typing import List, Dict, Any, Optional

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

    async def get_models(self) -> List[Dict[str, Any]]:
        """Return available models from G4F Nvidia proxy."""
        # Key free models (170+ total available)
        return [
            # Meta Llama models
            {
                "id": "meta/llama-3.1-405b-instruct",
                "object": "model",
                "owned_by": "meta",
            },
            {
                "id": "meta/llama-3.3-70b-instruct",
                "object": "model",
                "owned_by": "meta",
            },
            {
                "id": "meta/llama-4-maverick-17b-128e-instruct",
                "object": "model",
                "owned_by": "meta",
            },
            {
                "id": "meta/llama-4-scout-17b-16e-instruct",
                "object": "model",
                "owned_by": "meta",
            },
            # DeepSeek models
            {
                "id": "deepseek-ai/deepseek-v3.1",
                "object": "model",
                "owned_by": "deepseek",
            },
            {
                "id": "deepseek-ai/deepseek-v3.2",
                "object": "model",
                "owned_by": "deepseek",
            },
            {
                "id": "deepseek-ai/deepseek-r1-distill-llama-70b",
                "object": "model",
                "owned_by": "deepseek",
            },
            # Google Gemma models
            {"id": "google/gemma-3-27b-it", "object": "model", "owned_by": "google"},
            {"id": "google/gemma-3-12b-it", "object": "model", "owned_by": "google"},
            {"id": "google/gemma-3-4b-it", "object": "model", "owned_by": "google"},
            # Mistral models
            {
                "id": "mistralai/mistral-large-3-675b",
                "object": "model",
                "owned_by": "mistral",
            },
            {
                "id": "mistralai/codestral-2501",
                "object": "model",
                "owned_by": "mistral",
            },
            # Qwen models
            {"id": "qwen/qwen3-235b-a22b", "object": "model", "owned_by": "alibaba"},
            {"id": "qwen/qwen3-coder-480b", "object": "model", "owned_by": "alibaba"},
            {"id": "qwen/qwen3.5-397b", "object": "model", "owned_by": "alibaba"},
            # Nvidia Nemotron
            {
                "id": "nvidia/nemotron-ultra-253b-v1",
                "object": "model",
                "owned_by": "nvidia",
            },
            {
                "id": "nvidia/nemotron-super-49b-v1",
                "object": "model",
                "owned_by": "nvidia",
            },
            # Microsoft Phi
            {
                "id": "microsoft/phi-4-mini-instruct",
                "object": "model",
                "owned_by": "microsoft",
            },
            {
                "id": "microsoft/phi-3.5-mini",
                "object": "model",
                "owned_by": "microsoft",
            },
            # Moonshot Kimi
            {
                "id": "moonshotai/kimi-k2-base",
                "object": "model",
                "owned_by": "moonshot",
            },
        ]
