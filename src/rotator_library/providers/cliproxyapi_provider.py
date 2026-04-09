"""
CLIProxyAPI Provider - Routes through CLIProxyAPI sidecar service.

CLIProxyAPI is a Go-based OAuth router that handles:
- OAuth token refresh (Gemini CLI, Antigravity, Qwen Code)
- Cookie-based authentication (iFlow - IMPORTANT: use cookie method, not OAuth!)
- Multi-account load balancing
- Format translation (OpenAI ↔ Claude ↔ Gemini)

Configuration:
    CLIPROXYAPI_ENABLED=true
    CLIPROXYAPI_BASE_URL=http://127.0.0.1:8317
    CLIPROXYAPI_TIMEOUT=120

Model Routing:
    gemini/* → Gemini CLI OAuth (auto-refresh)
    iflow/* → iFlow cookie-based auth (auto-refresh)
    antigravity/* → Antigravity OAuth (auto-refresh)
    qwen/* → Qwen Code OAuth (auto-refresh)

IMPORTANT for iFlow:
    Use cookie-based authentication, NOT OAuth! OAuth tokens expire every 7 days
    with NO auto-refresh capability. Cookie sessions last months and auto-refresh.

Usage:
    ./cliproxyapi -iflow-cookie  # Use cookie method
    # NOT: ./cliproxyapi -iflow-login  # OAuth expires in 7 days
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from dataclasses import dataclass
from enum import Enum

import httpx
import litellm

from .provider_interface import (
    ProviderInterface,
    TierPriorityMap,
    UsageConfigMap,
    UsageResetConfigDef,
)

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class CLIProxyAPIAuthMethod(Enum):
    """Authentication methods supported by CLIProxyAPI."""

    OAUTH = "oauth"
    COOKIE = "cookie"
    API_KEY = "api_key"


@dataclass
class CLIProxyAPIProviderConfig:
    """Configuration for CLIProxyAPI provider."""

    base_url: str = "http://127.0.0.1:8317"
    timeout: int = 120
    enabled: bool = True
    health_check_interval: int = 30


class CLIProxyAPIProvider(ProviderInterface):
    """
    Provider that routes requests through CLIProxyAPI sidecar.

    CLIProxyAPI handles OAuth token management and cookie-based authentication
    for multiple LLM providers, providing:

    1. Automatic token refresh (Gemini, Antigravity, Qwen)
    2. Cookie-based auth for iFlow (auto-refresh, unlike OAuth)
    3. Multi-account load balancing
    4. Format translation between API standards

    Key Design Decisions:
    - Uses sidecar pattern (separate Go process)
    - HTTP communication via OpenAI-compatible endpoints
    - Provider routing via model name prefixes

    Example:
        provider = CLIProxyAPIProvider()
        async for chunk in provider.acompletion(
            client=httpx.AsyncClient(),
            model="gemini/gemini-2.5-pro",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        ):
            print(chunk)
    """

    skip_cost_calculation: bool = True  # CLIProxyAPI handles cost tracking

    # Provider name for env var lookups
    provider_env_name: str = "cliproxyapi"

    # Tier priorities (CLIProxyAPI accounts are typically free tier)
    tier_priorities: TierPriorityMap = {
        "free": 5,
        "standard": 3,
        "premium": 2,
    }

    default_tier_priority: int = 5

    # Usage reset configuration (daily reset for free tier)
    usage_reset_configs: UsageConfigMap = {
        frozenset({5}): UsageResetConfigDef(
            window_seconds=86400,  # 24 hours
            mode="credential",
            description="Daily reset for CLIProxyAPI accounts",
            field_name="daily",
        ),
        "default": UsageResetConfigDef(
            window_seconds=86400,
            mode="credential",
            description="Default daily reset",
            field_name="daily",
        ),
    }

    # Supported providers via CLIProxyAPI
    SUPPORTED_BACKENDS = {
        "gemini": {
            "auth_method": CLIProxyAPIAuthMethod.OAUTH,
            "models": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash-exp"],
            "auto_refresh": True,
        },
        "iflow": {
            "auth_method": CLIProxyAPIAuthMethod.COOKIE,  # IMPORTANT: use cookie!
            "models": ["glm-4-plus", "glm-4-flash", "glm-4-air"],
            "auto_refresh": True,  # Works with cookie method
        },
        "antigravity": {
            "auth_method": CLIProxyAPIAuthMethod.OAUTH,
            "models": ["gemini-2.0-flash-exp", "gemini-3-flash"],
            "auto_refresh": True,
        },
        "qwen": {
            "auth_method": CLIProxyAPIAuthMethod.OAUTH,
            "models": ["qwen3-coder-plus", "qwen3-coder-lite"],
            "auto_refresh": True,
        },
    }

    def __init__(self, config: Optional[CLIProxyAPIProviderConfig] = None):
        """
        Initialize CLIProxyAPI provider.

        Args:
            config: Optional configuration object. If not provided, uses env vars.
        """
        if config is None:
            config = CLIProxyAPIProviderConfig(
                base_url=os.getenv("CLIPROXYAPI_BASE_URL", "http://127.0.0.1:8317"),
                timeout=int(os.getenv("CLIPROXYAPI_TIMEOUT", "120")),
                enabled=os.getenv("CLIPROXYAPI_ENABLED", "true").lower() == "true",
                health_check_interval=int(
                    os.getenv("CLIPROXYAPI_HEALTH_CHECK_INTERVAL", "30")
                ),
            )

        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._health_status: Dict[str, Any] = {}

        lib_logger.info(f"CLIProxyAPI provider initialized: {self.config.base_url}")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for CLIProxyAPI."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(self.config.timeout),
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetch available models from CLIProxyAPI.

        Returns models in format: "{backend}/{model_name}"
        Examples: "gemini/gemini-2.5-pro", "iflow/glm-4-plus"

        Args:
            api_key: Not used (CLIProxyAPI handles auth internally)
            client: HTTP client (not used, we create our own)

        Returns:
            List of model names with backend prefixes
        """
        models = []

        try:
            client = await self._get_client()
            response = await client.get("/v1/models")
            response.raise_for_status()

            data = response.json()
            for model in data.get("data", []):
                model_id = model.get("id", "")
                if model_id:
                    models.append(model_id)

            lib_logger.info(f"Discovered {len(models)} models from CLIProxyAPI")

        except httpx.RequestError as e:
            lib_logger.warning(f"Failed to fetch models from CLIProxyAPI: {e}")
            # Fall back to static model list
            for backend, info in self.SUPPORTED_BACKENDS.items():
                for model in info["models"]:
                    models.append(f"{backend}/{model}")
            lib_logger.info(f"Using fallback static model list: {len(models)} models")

        return models

    def has_custom_logic(self) -> bool:
        """
        Returns True - we implement custom completion logic.

        We need custom logic because we route through CLIProxyAPI
        instead of using litellm directly.
        """
        return True

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handle chat completion through CLIProxyAPI.

        Routes to CLIProxyAPI's OpenAI-compatible endpoint. CLIProxyAPI
        handles OAuth/cookie auth, token refresh, and format translation.

        Args:
            client: HTTP client (used for base URL detection)
            **kwargs: Standard completion parameters (model, messages, stream, etc.)

        Returns:
            ModelResponse or async generator for streaming

        Example:
            response = await provider.acompletion(
                client=httpx.AsyncClient(),
                model="gemini/gemini-2.5-pro",
                messages=[{"role": "user", "content": "Hello"}]
            )
        """
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)

        # Extract backend from model name (e.g., "gemini/gemini-2.5-pro")
        backend = None
        clean_model = model
        if "/" in model:
            parts = model.split("/", 1)
            backend = parts[0]
            clean_model = parts[1]

        # Log which backend is being used
        if backend:
            backend_info = self.SUPPORTED_BACKENDS.get(backend, {})
            auth_method = backend_info.get("auth_method", CLIProxyAPIAuthMethod.OAUTH)
            if backend == "iflow" and auth_method == CLIProxyAPIAuthMethod.COOKIE:
                lib_logger.debug(
                    f"Routing to iFlow via cookie-based auth (auto-refresh)"
                )

        # Build request payload
        payload = {
            "model": model,  # CLIProxyAPI handles prefix routing
            "messages": messages,
            "stream": stream,
        }

        # Add optional parameters
        optional_params = [
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "tools",
            "tool_choice",
            "response_format",
            "seed",
            "user",
        ]
        for param in optional_params:
            if param in kwargs:
                payload[param] = kwargs[param]

        # Make request to CLIProxyAPI
        cliproxy_client = await self._get_client()

        try:
            if stream:
                return self._stream_completion(cliproxy_client, payload)
            else:
                return await self._non_stream_completion(cliproxy_client, payload)

        except httpx.HTTPStatusError as e:
            lib_logger.error(
                f"CLIProxyAPI HTTP error: {e.response.status_code} - {e.response.text}"
            )
            raise

        except httpx.RequestError as e:
            lib_logger.error(f"CLIProxyAPI connection error: {e}")
            raise

    async def _non_stream_completion(
        self, client: httpx.AsyncClient, payload: Dict[str, Any]
    ) -> litellm.ModelResponse:
        """Handle non-streaming completion."""
        response = await client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()

        data = response.json()

        # Convert to litellm ModelResponse format
        return litellm.ModelResponse(
            id=data.get("id", ""),
            choices=[
                litellm.Choices(
                    index=choice.get("index", 0),
                    message=litellm.Message(
                        content=choice.get("message", {}).get("content", ""),
                        role=choice.get("message", {}).get("role", "assistant"),
                        tool_calls=choice.get("message", {}).get("tool_calls"),
                    ),
                    finish_reason=choice.get("finish_reason"),
                )
                for choice in data.get("choices", [])
            ],
            created=data.get("created", 0),
            model=data.get("model", ""),
            usage=litellm.Usage(
                prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
                total_tokens=data.get("usage", {}).get("total_tokens", 0),
            )
            if data.get("usage")
            else None,
        )

    async def _stream_completion(
        self, client: httpx.AsyncClient, payload: Dict[str, Any]
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """Handle streaming completion."""
        async with client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue

                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix

                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)

                        # Convert to litellm streaming format
                        for choice in data.get("choices", []):
                            delta = choice.get("delta", {})

                            yield litellm.ModelResponse(
                                id=data.get("id", ""),
                                choices=[
                                    litellm.Choices(
                                        index=choice.get("index", 0),
                                        delta=litellm.Delta(
                                            content=delta.get("content"),
                                            role=delta.get("role"),
                                            tool_calls=delta.get("tool_calls"),
                                        ),
                                        finish_reason=choice.get("finish_reason"),
                                    )
                                ],
                                created=data.get("created", 0),
                                model=data.get("model", ""),
                            )

                    except json.JSONDecodeError:
                        lib_logger.warning(f"Failed to parse SSE data: {data_str}")
                        continue

    async def health_check(self) -> Dict[str, Any]:
        """
        Check CLIProxyAPI health status.

        Returns:
            Dict with status and provider health info
        """
        try:
            client = await self._get_client()
            response = await client.get("/health", timeout=10.0)

            if response.status_code == 200:
                data = response.json()
                self._health_status = {
                    "status": "healthy",
                    "cliproxyapi": data,
                    "last_check": data.get("uptime", "unknown"),
                }
                return self._health_status
            else:
                self._health_status = {
                    "status": "unhealthy",
                    "code": response.status_code,
                }
                return self._health_status

        except httpx.RequestError as e:
            self._health_status = {
                "status": "error",
                "error": str(e),
            }
            return self._health_status

        except Exception as e:
            self._health_status = {
                "status": "error",
                "error": f"Unexpected error: {e}",
            }
            return self._health_status

    async def get_provider_status(self) -> Dict[str, Any]:
        """
        Get detailed status of all providers in CLIProxyAPI.

        Returns which providers are authenticated and their health.
        """
        try:
            client = await self._get_client()
            response = await client.get("/v1/models")
            response.raise_for_status()

            data = response.json()
            models = data.get("data", [])

            # Group models by backend
            backends: Dict[str, List[str]] = {}
            for model in models:
                model_id = model.get("id", "")
                if "/" in model_id:
                    backend, model_name = model_id.split("/", 1)
                    if backend not in backends:
                        backends[backend] = []
                    backends[backend].append(model_name)

            return {
                "status": "healthy",
                "providers": backends,
                "total_models": len(models),
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }

    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def get_credential_tier_name(self, credential: str) -> Optional[str]:
        """
        Get tier name for a credential.

        CLIProxyAPI accounts are typically free tier.
        """
        # CLIProxyAPI doesn't expose tier info, default to free
        return "free"

    @staticmethod
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse quota errors from CLIProxyAPI.

        CLIProxyAPI passes through provider-specific errors.
        """
        if error_body:
            try:
                data = json.loads(error_body)
                error_info = data.get("error", {})

                # Check for quota-related errors
                if "quota" in error_info.get("message", "").lower():
                    return {
                        "reason": "QUOTA_EXHAUSTED",
                        "retry_after": error_info.get("retry_after"),
                    }

                if "rate" in error_info.get("message", "").lower():
                    return {
                        "reason": "RATE_LIMITED",
                        "retry_after": error_info.get("retry_after", 60),
                    }

            except json.JSONDecodeError:
                pass

        return None


# Provider registration metadata
PROVIDER_CLASS = CLIProxyAPIProvider
PROVIDER_NAME = "cliproxyapi"

# Default supported models (static fallback)
SUPPORTED_MODELS = []
for backend, info in CLIProxyAPIProvider.SUPPORTED_BACKENDS.items():
    for model in info["models"]:
        SUPPORTED_MODELS.append(f"{backend}/{model}")
