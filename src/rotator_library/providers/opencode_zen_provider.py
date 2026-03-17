"""OpenCode Zen provider — free agentic coding models via opencode.ai/zen.

OpenCode Zen provides free unlimited access to select LLM models for agentic coding.
The API is OpenAI-compatible at https://opencode.ai/zen/v1.

No API key required — pass any non-empty string as Bearer token.
Rate limits apply under heavy load but no hard daily limits.
"""

import os
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


# Static model list — all free under OpenCode Zen
ZEN_MODELS = [
    "minimax/m2.1",
    "trinity/large-preview",
    "moonshot/kimi-k2.5",
    "zhipu/glm-4.7",
    "bigcode/pickle",
]

# Parameters supported by the OpenAI-compatible endpoint
SUPPORTED_PARAMS = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "stream",
    "tools",
    "tool_choice",
    "presence_penalty",
    "frequency_penalty",
    "n",
    "stop",
    "seed",
    "response_format",
}


class OpenCodeZenProvider(ProviderInterface):
    """OpenCode Zen — free unlimited LLM access for agentic coding.

    API: https://opencode.ai/zen/v1  (OpenAI-compatible)
    Auth: No key required — any non-empty Bearer token works.
    Models: minimax/m2.1, trinity/large-preview, moonshot/kimi-k2.5, zhipu/glm-4.7, bigcode/pickle
    """

    provider_name = "opencode_zen"
    provider_env_name = "opencode_zen"
    skip_cost_calculation = True
    default_tier_priority = 2  # High priority — free and reliable

    _DEFAULT_API_BASE = "https://opencode.ai/zen/v1"

    def __init__(self):
        self.api_key: str = os.getenv("OPENCODE_ZEN_API_KEY") or "opencode-zen-free"
        self._api_base: str = os.getenv(
            "OPENCODE_ZEN_API_BASE", self._DEFAULT_API_BASE
        ).rstrip("/")

    def has_custom_logic(self) -> bool:
        return True

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Return available OpenCode Zen models prefixed with provider name."""
        return [f"opencode_zen/{m}" for m in ZEN_MODELS]

    def _build_payload(self, **kwargs) -> Dict[str, Any]:
        """Build clean request payload — only supported params."""
        payload = {k: v for k, v in kwargs.items() if k in SUPPORTED_PARAMS}
        payload["stream"] = True  # always stream internally
        return payload

    async def acompletion(
        self,
        client: httpx.AsyncClient,
        **kwargs,
    ) -> Any:
        """Execute a chat completion request against OpenCode Zen via litellm."""
        import litellm

        # Strip provider prefix from model name if present
        model = kwargs.get("model", "")
        if model.startswith("opencode_zen/"):
            kwargs["model"] = model[len("opencode_zen/") :]

        # Only pass supported params
        clean_kwargs = {k: v for k, v in kwargs.items() if k in SUPPORTED_PARAMS}

        lib_logger.debug(
            f"OpenCode Zen request: model={clean_kwargs.get('model')} base={self._api_base}"
        )

        return await litellm.acompletion(
            api_base=self._api_base,
            api_key=self.api_key,
            **clean_kwargs,
        )

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """Standard Bearer token header."""
        return {"Authorization": f"Bearer {credential_identifier or self.api_key}"}
