"""
Provider Adapter Module

Defines clean adapter interfaces for provider models and handles capability mapping.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
import logging
import time
import asyncio
import os
from dataclasses import dataclass

import litellm
from fastapi import HTTPException
import httpx

logger = logging.getLogger(__name__)


@dataclass
class ModelCapabilities:
    """Model capability flags and metadata."""

    provider: str
    model: str

    # Core capabilities
    supports_tools: bool = False
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_structured_output: bool = False
    supports_system_messages: bool = True
    supports_streaming: bool = True

    # Context limits
    max_context_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None

    # Performance characteristics
    average_latency_ms: Optional[float] = None
    throughput_tokens_per_second: Optional[float] = None

    # Cost information (None for free tier)
    input_cost_per_token: Optional[float] = None
    output_cost_per_token: Optional[float] = None

    # Free tier status
    free_tier_available: bool = True
    rate_limit_requests_per_minute: Optional[int] = None

    # Tags for routing
    tags: Optional[List[str]] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "provider": self.provider,
            "model": self.model,
            "supports_tools": self.supports_tools,
            "supports_function_calling": self.supports_function_calling,
            "supports_vision": self.supports_vision,
            "supports_structured_output": self.supports_structured_output,
            "supports_system_messages": self.supports_system_messages,
            "supports_streaming": self.supports_streaming,
            "max_context_tokens": self.max_context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "free_tier_available": self.free_tier_available,
            "rate_limit_requests_per_minute": self.rate_limit_requests_per_minute,
            "tags": self.tags,
        }


class BaseProviderAdapter(ABC):
    """Base adapter for all providers."""

    def __init__(self, provider_name: str, api_key: Optional[str] = None):
        self.provider_name = provider_name
        self.api_key = api_key
        self.models: Dict[str, ModelCapabilities] = {}
        self._initialize_models()

    @abstractmethod
    def _initialize_models(self):
        """Initialize model capabilities."""
        pass

    @abstractmethod
    async def list_models(self) -> List[str]:
        """List available models."""
        pass

    @abstractmethod
    async def chat_completions(
        self,
        request: Dict[str, Any],
        stream: bool = False,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Execute chat completion request."""
        pass

    def get_model_capabilities(self, model: str) -> Optional[ModelCapabilities]:
        """Get capabilities for a specific model."""
        return self.models.get(model)

    def is_model_available(self, model: str, free_only: bool = True) -> bool:
        """Check if model is available given constraints."""
        capabilities = self.get_model_capabilities(model)
        if not capabilities:
            return False

        if free_only and not capabilities.free_tier_available:
            return False

        return True


class OpenAICompatibleAdapter(BaseProviderAdapter):
    def __init__(
        self,
        provider_name: str,
        api_key: Optional[str],
        api_base: Optional[str],
        models: Optional[List[str]] = None,
    ):
        self.api_base = api_base
        self.model_list = models or []
        super().__init__(provider_name, api_key)

    def _initialize_models(self):
        models: Dict[str, ModelCapabilities] = {}
        for model in self.model_list:
            models[model] = ModelCapabilities(
                provider=self.provider_name,
                model=model,
                supports_tools=True,
                supports_function_calling=True,
                supports_streaming=True,
                free_tier_available=True,
                tags=["openai_compatible", "custom"],
            )
        self.models.update(models)

    async def list_models(self) -> List[str]:
        return list(self.models.keys())

    async def chat_completions(
        self,
        request: Dict[str, Any],
        stream: bool = False,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        if not self.api_key:
            raise ValueError(f"{self.provider_name} API key not configured")
        if not self.api_base:
            raise ValueError(f"{self.provider_name} API base not configured")

        request_with_key = request.copy()

        model = request_with_key.get("model")
        if isinstance(model, str):
            prefix = f"{self.provider_name}/"
            if model.startswith(prefix):
                request_with_key["model"] = model[len(prefix) :]

        request_with_key["api_key"] = self.api_key
        request_with_key["api_base"] = self.api_base
        request_with_key["custom_llm_provider"] = "openai"

        if stream:
            return self._stream_completion(request_with_key)
        return await self._non_stream_completion(request_with_key)

    async def _non_stream_completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = await litellm.acompletion(**request)
            return self._convert_response(response)
        except Exception as e:
            logger.error(f"{self.provider_name} completion failed: {e}")
            raise self._convert_error(e)

    async def _stream_completion(
        self, request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            stream_resp: Any = await litellm.acompletion(**request, stream=True)
            async for chunk in stream_resp:
                yield self._convert_chunk(chunk)
        except Exception as e:
            logger.error(f"{self.provider_name} streaming failed: {e}")
            raise self._convert_error(e)

    def _convert_response(self, response: Any) -> Dict[str, Any]:
        if hasattr(response, "dict"):
            return response.dict()
        if hasattr(response, "__dict__"):
            return response.__dict__
        return response

    def _convert_chunk(self, chunk: Any) -> Dict[str, Any]:
        if hasattr(chunk, "dict"):
            return chunk.dict()
        if hasattr(chunk, "__dict__"):
            return chunk.__dict__
        return chunk

    def _convert_error(self, error: Exception) -> HTTPException:
        """Convert exception to HTTPException with proper status code for fallback routing.

        Enhanced to detect usage/quota limits from various providers including:
        - Standard OpenAI: "rate_limit", "insufficient_quota"
        - AIHubMix/GLM: error code 1302, "exceeded", "concurrency"
        - Generic: "quota", "usage", "exhausted", "daily", "limit"
        """
        error_str = str(error).lower()
        if "authentication" in error_str or "api_key" in error_str:
            status_code = 401
        elif any(
            kw in error_str
            for kw in [
                "rate_limit",
                "too many",
                "429",
                "quota",
                "usage",
                "exceeded",
                "exhausted",
                "limit",
                "daily",
                "insufficient",
                "1302",  # GLM-specific concurrency/usage error code
                "concurrency",
                "requests per",
                "resource_exhausted",
                "capacity",
            ]
        ):
            status_code = 429
        elif "invalid" in error_str or "bad request" in error_str:
            status_code = 400
        else:
            status_code = 500
        return HTTPException(status_code=status_code, detail=str(error))


class SupacoderAdapter(OpenAICompatibleAdapter):
    """Adapter for Supacoder with aggressive stream stall protection."""

    def __init__(
        self,
        provider_name: str,
        api_key: Optional[str],
        api_base: Optional[str],
        models: Optional[List[str]] = None,
        stream_timeout_seconds: int = 45,
    ):
        env_timeout = os.getenv("SUPACODER_STREAM_TIMEOUT_SECONDS")
        self.stream_timeout_seconds = (
            int(env_timeout)
            if env_timeout and env_timeout.isdigit()
            else stream_timeout_seconds
        )
        self.stream_timeout_seconds = max(
            self.stream_timeout_seconds, 90
        )  # Minimum 90s for reliable streaming
        super().__init__(provider_name, api_key, api_base, models)

        # Supacoder endpoints behave like text-only Chat Completions; function/tool calls are not reliable.
        for caps in self.models.values():
            caps.supports_tools = False
            caps.supports_function_calling = False

    async def chat_completions(
        self,
        request: Dict[str, Any],
        stream: bool = False,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        # Explicitly block tool/function calls so the router can fall back to a tool-capable provider.
        if request.get("tools") or request.get("functions"):
            raise HTTPException(
                status_code=400,
                detail="supacoder does not support tool/function calls; fallback required",
            )

        return await super().chat_completions(request, stream=stream, client=client)

    async def _stream_completion(
        self, request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream with timeout; on stall, emit retryable HTTP error and log telemetry."""
        model = request.get("model")
        start = time.time()
        telemetry = None
        try:
            try:
                from ..rotator_library.telemetry import get_telemetry_manager

                telemetry = get_telemetry_manager()
            except Exception:
                telemetry = None

            stream_resp: Any = await litellm.acompletion(**request, stream=True)

            while True:
                try:
                    chunk = await asyncio.wait_for(
                        stream_resp.__anext__(), timeout=self.stream_timeout_seconds
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    elapsed_ms = int((time.time() - start) * 1000)
                    if telemetry:
                        telemetry.record_call(
                            provider=self.provider_name,
                            model=model or "",
                            success=False,
                            response_time_ms=elapsed_ms,
                            error_reason="stream_timeout",
                        )
                        telemetry.update_provider_health(
                            provider=self.provider_name,
                            model=model,
                            is_healthy=False,
                            failure_rate=1.0,
                        )
                    raise HTTPException(
                        status_code=504,
                        detail="stream_timeout: supacoder stream stalled",
                    )

                yield self._convert_chunk(chunk)

        except HTTPException:
            raise
        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            if telemetry:
                telemetry.record_call(
                    provider=self.provider_name,
                    model=model or "",
                    success=False,
                    response_time_ms=elapsed_ms,
                    error_reason=str(e),
                )
                telemetry.update_provider_health(
                    provider=self.provider_name,
                    model=model,
                    is_healthy=False,
                    failure_rate=1.0,
                )
            logger.error(f"{self.provider_name} streaming failed: {e}")
            raise self._convert_error(e)


class GroqAdapter(BaseProviderAdapter):
    """Adapter for Groq provider."""

    def _initialize_models(self):
        """Initialize Groq model capabilities."""
        groq_models = {
            "llama-3.3-70b-versatile": ModelCapabilities(
                provider="groq",
                model="llama-3.3-70b-versatile",
                supports_tools=True,
                supports_function_calling=True,
                supports_streaming=True,
                max_context_tokens=32768,
                max_output_tokens=32768,
                free_tier_available=True,
                rate_limit_requests_per_minute=30,
                tags=["coding", "reasoning", "general"],
            ),
            "llama-3.1-8b-instant": ModelCapabilities(
                provider="groq",
                model="llama-3.1-8b-instant",
                supports_tools=True,
                supports_function_calling=True,
                supports_streaming=True,
                max_context_tokens=8192,
                max_output_tokens=8192,
                free_tier_available=True,
                rate_limit_requests_per_minute=30,
                tags=["fast", "chat", "general"],
            ),
            "mixtral-8x7b-32768": ModelCapabilities(
                provider="groq",
                model="mixtral-8x7b-32768",
                supports_tools=True,
                supports_streaming=True,
                max_context_tokens=32768,
                max_output_tokens=32768,
                free_tier_available=True,
                rate_limit_requests_per_minute=30,
                tags=["coding", "specialized", "general"],
            ),
            "gemma2-9b-it": ModelCapabilities(
                provider="groq",
                model="gemma2-9b-it",
                supports_tools=True,
                supports_streaming=True,
                max_context_tokens=8192,
                max_output_tokens=8192,
                free_tier_available=True,
                rate_limit_requests_per_minute=30,
                tags=["fast", "general"],
            ),
        }

        self.models.update(groq_models)

    async def list_models(self) -> List[str]:
        """List available Groq models."""
        return list(self.models.keys())

    async def chat_completions(
        self,
        request: Dict[str, Any],
        stream: bool = False,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Execute chat completion via Groq."""
        if not self.api_key:
            raise ValueError("Groq API key not configured")

        # Set API key in request
        request_with_key = request.copy()
        request_with_key["api_key"] = self.api_key

        if stream:
            return self._stream_completion(request_with_key)
        else:
            return await self._non_stream_completion(request_with_key)

    async def _non_stream_completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Non-streaming completion."""
        try:
            response = await litellm.acompletion(**request)
            return self._convert_response(response)
        except Exception as e:
            logger.error(f"Groq completion failed: {e}")
            raise self._convert_error(e)

    async def _stream_completion(
        self, request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            stream_resp: Any = await litellm.acompletion(**request, stream=True)
            async for chunk in stream_resp:
                yield self._convert_chunk(chunk)
        except Exception as e:
            logger.error(f"Groq streaming failed: {e}")
            raise self._convert_error(e)

    def _convert_response(self, response: Any) -> Dict[str, Any]:
        """Convert LiteLLM response to standard format."""
        if hasattr(response, "dict"):
            return response.dict()
        elif hasattr(response, "__dict__"):
            return response.__dict__
        return response

    def _convert_chunk(self, chunk: Any) -> Dict[str, Any]:
        """Convert LiteLLM streaming chunk to standard format."""
        if hasattr(chunk, "dict"):
            return chunk.dict()
        elif hasattr(chunk, "__dict__"):
            return chunk.__dict__
        return chunk

    def _convert_error(self, error: Exception) -> HTTPException:
        """Convert provider error to HTTPException."""
        error_str = str(error).lower()

        if "authentication" in error_str or "api_key" in error_str:
            status_code = 401
        elif "rate_limit" in error_str or "too many" in error_str:
            status_code = 429
        elif "invalid" in error_str or "bad request" in error_str:
            status_code = 400
        else:
            status_code = 500

        return HTTPException(status_code=status_code, detail=str(error))


class GeminiAdapter(BaseProviderAdapter):
    """Adapter for Gemini provider."""

    def _initialize_models(self):
        """Initialize Gemini model capabilities (Updated for 2026)."""
        gemini_models = {
            "gemini-2.5-flash": ModelCapabilities(
                provider="gemini",
                model="gemini-2.5-flash",
                supports_tools=True,
                supports_function_calling=True,
                supports_vision=True,
                supports_structured_output=True,
                supports_streaming=True,
                max_context_tokens=1048576,  # 1M tokens
                max_output_tokens=65536,
                free_tier_available=True,
                rate_limit_requests_per_minute=15,
                tags=["fast", "vision", "long_context", "general"],
            ),
            "gemini-2.5-pro": ModelCapabilities(
                provider="gemini",
                model="gemini-2.5-pro",
                supports_tools=True,
                supports_function_calling=True,
                supports_vision=True,
                supports_structured_output=True,
                supports_streaming=True,
                max_context_tokens=1048576,  # 1M tokens
                max_output_tokens=65536,
                free_tier_available=True,
                rate_limit_requests_per_minute=2,
                tags=["reasoning", "vision", "long_context", "coding", "research"],
            ),
            "gemini-3-pro-preview": ModelCapabilities(
                provider="gemini",
                model="gemini-3-pro-preview",
                supports_tools=True,
                supports_function_calling=True,
                supports_vision=True,
                supports_streaming=True,
                max_context_tokens=1048576,
                max_output_tokens=65536,
                free_tier_available=True,
                rate_limit_requests_per_minute=2,
                tags=["ultra_brain", "vision", "coding"],
            ),
            "gemma-3-27b-it": ModelCapabilities(
                provider="gemini",
                model="gemma-3-27b-it",
                supports_tools=True,
                supports_streaming=True,
                max_context_tokens=131072,
                max_output_tokens=8192,
                free_tier_available=True,
                rate_limit_requests_per_minute=15,
                tags=["open_model", "fast", "coding"],
            ),
        }

        self.models.update(gemini_models)

    async def list_models(self) -> List[str]:
        """List available Gemini models."""
        return list(self.models.keys())

    async def chat_completions(
        self,
        request: Dict[str, Any],
        stream: bool = False,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Execute chat completion via Gemini."""
        if not self.api_key:
            raise ValueError("Gemini API key not configured")

        # Set API key in request
        request_with_key = request.copy()
        request_with_key["api_key"] = self.api_key

        if stream:
            return self._stream_completion(request_with_key)
        else:
            return await self._non_stream_completion(request_with_key)

    async def _non_stream_completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Non-streaming completion."""
        try:
            response = await litellm.acompletion(**request)
            return self._convert_response(response)
        except Exception as e:
            logger.error(f"Gemini completion failed: {e}")
            raise self._convert_error(e)

    async def _stream_completion(
        self, request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            stream_resp: Any = await litellm.acompletion(**request, stream=True)
            async for chunk in stream_resp:
                yield self._convert_chunk(chunk)
        except Exception as e:
            logger.error(f"Gemini streaming failed: {e}")
            raise self._convert_error(e)

    def _convert_response(self, response: Any) -> Dict[str, Any]:
        """Convert LiteLLM response to standard format."""
        if hasattr(response, "dict"):
            return response.dict()
        elif hasattr(response, "__dict__"):
            return response.__dict__
        return response

    def _convert_chunk(self, chunk: Any) -> Dict[str, Any]:
        """Convert LiteLLM streaming chunk to standard format."""
        if hasattr(chunk, "dict"):
            return chunk.dict()
        elif hasattr(chunk, "__dict__"):
            return chunk.__dict__
        return chunk

    def _convert_error(self, error: Exception) -> HTTPException:
        """Convert provider error to HTTPException."""
        error_str = str(error).lower()

        if "api_key" in error_str or "permission" in error_str:
            status_code = 401
        elif "quota" in error_str or "rate" in error_str:
            status_code = 429
        elif "invalid" in error_str or "bad request" in error_str:
            status_code = 400
        else:
            status_code = 500

        return HTTPException(status_code=status_code, detail=str(error))


class G4FAdapter(BaseProviderAdapter):
    """Adapter for G4F (Free GPT) provider."""

    def __init__(self, provider_name: str = "g4f", api_key: Optional[str] = None):
        super().__init__(provider_name, api_key)

    def _initialize_models(self):
        """Initialize G4F model capabilities."""
        if self.provider_name == "g4f":
            g4f_models = {
                "gpt-4": ModelCapabilities(
                    provider="g4f",
                    model="gpt-4",
                    supports_tools=False,
                    supports_streaming=True,
                    max_context_tokens=8192,
                    max_output_tokens=4096,
                    free_tier_available=True,
                    rate_limit_requests_per_minute=10,
                    tags=["general", "fallback"],
                ),
                "gpt-3.5-turbo": ModelCapabilities(
                    provider="g4f",
                    model="gpt-3.5-turbo",
                    supports_tools=False,
                    supports_streaming=True,
                    max_context_tokens=4096,
                    max_output_tokens=2048,
                    free_tier_available=True,
                    rate_limit_requests_per_minute=15,
                    tags=["fast", "general", "fallback"],
                ),
            }
        elif self.provider_name == "g4f_nvidia":
            g4f_models = {
                "meta/llama-3.3-70b-instruct": ModelCapabilities(
                    provider="g4f_nvidia",
                    model="meta/llama-3.3-70b-instruct",
                    supports_tools=True,
                    supports_streaming=True,
                    max_context_tokens=131072,
                    max_output_tokens=8192,
                    free_tier_available=True,
                    rate_limit_requests_per_minute=30,
                    tags=["coding", "general"],
                ),
                "deepseek-ai/deepseek-v3.1": ModelCapabilities(
                    provider="g4f_nvidia",
                    model="deepseek-ai/deepseek-v3.1",
                    supports_tools=True,
                    supports_streaming=True,
                    max_context_tokens=65536,
                    max_output_tokens=8192,
                    free_tier_available=True,
                    rate_limit_requests_per_minute=30,
                    tags=["coding", "general"],
                ),
            }
        elif self.provider_name == "g4f_pollinations":
            g4f_models = {
                "openai": ModelCapabilities(
                    provider="g4f_pollinations",
                    model="openai",
                    supports_tools=False,
                    supports_streaming=True,
                    max_context_tokens=8192,
                    max_output_tokens=4096,
                    free_tier_available=True,
                    rate_limit_requests_per_minute=60,
                    tags=["general", "fast"],
                ),
                "openai-fast": ModelCapabilities(
                    provider="g4f_pollinations",
                    model="openai-fast",
                    supports_tools=False,
                    supports_streaming=True,
                    max_context_tokens=4096,
                    max_output_tokens=2048,
                    free_tier_available=True,
                    rate_limit_requests_per_minute=60,
                    tags=["fast"],
                ),
            }
        elif self.provider_name == "g4f_ollama":
            g4f_models = {
                "deepseek-v3.2": ModelCapabilities(
                    provider="g4f_ollama",
                    model="deepseek-v3.2",
                    supports_tools=True,
                    supports_streaming=True,
                    max_context_tokens=65536,
                    max_output_tokens=8192,
                    free_tier_available=True,
                    rate_limit_requests_per_minute=30,
                    tags=["coding", "general"],
                ),
            }
        elif self.provider_name == "g4f_gemini":
            g4f_models = {
                "gemini-2.5-flash": ModelCapabilities(
                    provider="g4f_gemini",
                    model="gemini-2.5-flash",
                    supports_tools=True,
                    supports_streaming=True,
                    max_context_tokens=32768,
                    max_output_tokens=4096,
                    free_tier_available=True,
                    rate_limit_requests_per_minute=30,
                    tags=["fast", "general"],
                ),
            }
        elif self.provider_name == "g4f_groq":
            g4f_models = {
                "llama-3.3-70b-versatile": ModelCapabilities(
                    provider="g4f_groq",
                    model="llama-3.3-70b-versatile",
                    supports_tools=True,
                    supports_streaming=True,
                    max_context_tokens=131072,
                    max_output_tokens=8192,
                    free_tier_available=True,
                    rate_limit_requests_per_minute=30,
                    tags=["coding", "general", "fast"],
                ),
            }
        else:
            g4f_models = {
                "gpt-4": ModelCapabilities(
                    provider=self.provider_name,
                    model="gpt-4",
                    supports_tools=False,
                    supports_streaming=True,
                    max_context_tokens=8192,
                    max_output_tokens=4096,
                    free_tier_available=True,
                    rate_limit_requests_per_minute=10,
                    tags=["general"],
                ),
            }

        self.models.update(g4f_models)

    async def list_models(self) -> List[str]:
        """List available G4F models."""
        return list(self.models.keys())

    async def chat_completions(
        self,
        request: Dict[str, Any],
        stream: bool = False,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Execute chat completion via G4F."""
        try:
            # Import g4f dynamically to avoid issues if not installed
            import g4f

            # Extract messages
            messages = request.get("messages", [])

            # Get the last user message for G4F
            user_content = ""
            for message in reversed(messages):
                if message.get("role") == "user":
                    content = message.get("content", "")
                    if isinstance(content, str):
                        user_content = content
                    elif isinstance(content, list):
                        # Extract text from content items
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                user_content = item.get("text", "")
                                break
                    break

            if not user_content:
                raise ValueError("No user message found for G4F")

            # Create G4F request
            model_str = request.get("model", "")
            for prefix in [
                "g4f/",
                "g4f_ollama/",
                "g4f_pollinations/",
                "g4f_nvidia/",
                "g4f_gemini/",
                "g4f_groq/",
            ]:
                model_str = model_str.replace(prefix, "")

            if stream:
                return self._stream_g4f_completion(user_content, model_str)
            else:
                return await self._non_stream_g4f_completion(user_content, model_str)

        except ImportError:
            raise HTTPException(status_code=500, detail="G4F library not installed")
        except Exception as e:
            logger.error(f"G4F completion failed: {e}")
            raise self._convert_error(e)

    async def _non_stream_g4f_completion(
        self, content: str, model: str
    ) -> Dict[str, Any]:
        """Non-streaming G4F completion."""
        import g4f

        try:
            coro_resp: Any = g4f.ChatCompletion.create_async(
                model=model, messages=[{"role": "user", "content": content}]
            )
            response = await coro_resp

            # Convert to OpenAI format
            return {
                "id": f"g4f-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": f"g4f/{model}",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(content.split()),
                    "completion_tokens": len(response.split()),
                    "total_tokens": len(content.split()) + len(response.split()),
                },
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"G4F error: {str(e)}")

    async def _stream_g4f_completion(
        self, content: str, model: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming G4F completion."""
        import g4f

        try:
            stream_coro: Any = g4f.ChatCompletion.create_async(
                model=model,
                messages=[{"role": "user", "content": content}],
                stream=True,
            )
            response = await stream_coro

            chunk_id = f"g4f-{int(time.time())}"
            created = int(time.time())

            async for chunk in response:
                if chunk:
                    yield {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": f"g4f/{model}",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None,
                            }
                        ],
                    }

            # Send final chunk
            yield {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": f"g4f/{model}",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

        except Exception as e:
            yield {"error": f"G4F streaming error: {str(e)}"}

    def _convert_error(self, error: Exception) -> HTTPException:
        """Convert provider error to HTTPException."""
        error_str = str(error).lower()

        if "rate" in error_str:
            status_code = 429
        elif "invalid" in error_str:
            status_code = 400
        else:
            status_code = 500

        return HTTPException(status_code=status_code, detail=f"G4F error: {str(error)}")


class TogetherAdapter(BaseProviderAdapter):
    """Adapter for Together AI provider."""

    def _initialize_models(self):
        """Initialize Together AI model capabilities."""
        together_models = {
            "togethercomputer/MoA-1": ModelCapabilities(
                provider="together",
                model="togethercomputer/MoA-1",
                supports_tools=True,
                supports_streaming=True,
                max_context_tokens=32768,
                free_tier_available=True,
                tags=["moe", "general"],
            ),
            "togethercomputer/MoA-1-Turbo": ModelCapabilities(
                provider="together",
                model="togethercomputer/MoA-1-Turbo",
                supports_tools=True,
                supports_streaming=True,
                max_context_tokens=32768,
                free_tier_available=True,
                tags=["moe", "fast", "general"],
            ),
            "meta-llama/Llama-3.3-70B-Instruct-Turbo": ModelCapabilities(
                provider="together",
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                supports_tools=True,
                supports_function_calling=True,
                supports_streaming=True,
                max_context_tokens=131072,
                free_tier_available=False,  # Paid model
                tags=["large", "reasoning"],
            ),
            "deepseek-ai/DeepSeek-V3": ModelCapabilities(
                provider="together",
                model="deepseek-ai/DeepSeek-V3",
                supports_tools=True,
                supports_streaming=True,
                max_context_tokens=163840,
                free_tier_available=False,
                tags=["flagship", "coding"],
            ),
        }

        self.models.update(together_models)

    async def list_models(self) -> List[str]:
        """List available Together AI models."""
        return list(self.models.keys())

    async def chat_completions(
        self,
        request: Dict[str, Any],
        stream: bool = False,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Execute chat completion via Together AI."""
        if not self.api_key:
            raise ValueError("Together AI API key not configured")

        request_with_key = request.copy()
        request_with_key["api_key"] = self.api_key

        model = request_with_key.get("model", "")
        if model.startswith("together/"):
            request_with_key["model"] = model.replace("together/", "together_ai/", 1)
        elif not model.startswith("together_ai/"):
            request_with_key["model"] = f"together_ai/{model}"

        if stream:
            return self._stream_completion(request_with_key)
        else:
            return await self._non_stream_completion(request_with_key)

    async def _non_stream_completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Non-streaming completion."""
        try:
            response = await litellm.acompletion(**request)
            return self._convert_response(response)
        except Exception as e:
            logger.error(f"Together completion failed: {e}")
            raise self._convert_error(e)

    async def _stream_completion(
        self, request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming completion."""
        try:
            # Use typing.cast to avoid "AsyncResult" is not awaitable error
            from typing import cast

            coro = litellm.acompletion(**request)
            async for chunk in await cast(Any, coro):
                yield self._convert_chunk(chunk)
        except Exception as e:
            logger.error(f"Together streaming failed: {e}")
            raise self._convert_error(e)

    def _convert_response(self, response: Any) -> Dict[str, Any]:
        """Convert LiteLLM response to standard format."""
        if hasattr(response, "dict"):
            return response.dict()
        elif hasattr(response, "__dict__"):
            return response.__dict__
        return response

    def _convert_chunk(self, chunk: Any) -> Dict[str, Any]:
        """Convert LiteLLM streaming chunk to standard format."""
        if hasattr(chunk, "dict"):
            return chunk.dict()
        elif hasattr(chunk, "__dict__"):
            return chunk.__dict__
        return chunk

    def _convert_error(self, error: Exception) -> HTTPException:
        """Convert provider error to HTTPException."""
        return HTTPException(status_code=500, detail=str(error))


class KiloAdapter(BaseProviderAdapter):
    def _initialize_models(self):
        kilo_models = {
            "x-ai/grok-code-fast-1:optimized:free": ModelCapabilities(
                provider="kilo",
                model="x-ai/grok-code-fast-1:optimized:free",
                supports_tools=True,
                supports_function_calling=True,
                supports_streaming=True,
                free_tier_available=True,
                tags=["coding", "fast", "free"],
            ),
            "arcee-ai/trinity-large-preview:free": ModelCapabilities(
                provider="kilo",
                model="arcee-ai/trinity-large-preview:free",
                supports_tools=True,
                supports_function_calling=True,
                supports_streaming=True,
                free_tier_available=True,
                tags=["coding", "free"],
            ),
            "minimax/minimax-m2.5:free": ModelCapabilities(
                provider="kilo",
                model="minimax/minimax-m2.5:free",
                supports_tools=True,
                supports_function_calling=True,
                supports_streaming=True,
                free_tier_available=True,
                tags=["general", "free"],
            ),
            "nvidia/nemotron-3-super-120b-a12b:free": ModelCapabilities(
                provider="kilo",
                model="nvidia/nemotron-3-super-120b-a12b:free",
                supports_tools=True,
                supports_function_calling=True,
                supports_streaming=True,
                free_tier_available=True,
                tags=["coding", "reasoning", "free"],
            ),
            "stepfun/step-3.5-flash:free": ModelCapabilities(
                provider="kilo",
                model="stepfun/step-3.5-flash:free",
                supports_tools=True,
                supports_function_calling=True,
                supports_streaming=True,
                free_tier_available=True,
                tags=["fast", "free"],
            ),
            "corethink:free": ModelCapabilities(
                provider="kilo",
                model="corethink:free",
                supports_tools=False,
                supports_streaming=True,
                free_tier_available=True,
                tags=["reasoning", "free"],
            ),
            "kilo/auto-free": ModelCapabilities(
                provider="kilo",
                model="kilo/auto-free",
                supports_tools=True,
                supports_function_calling=True,
                supports_streaming=True,
                free_tier_available=True,
                tags=["auto", "free"],
            ),
        }
        self.models.update(kilo_models)

    async def list_models(self) -> List[str]:
        return list(self.models.keys())

    async def chat_completions(
        self,
        request: Dict[str, Any],
        stream: bool = False,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        if not self.api_key:
            raise ValueError("Kilo API key not configured")

        request_with_key = request.copy()
        request_with_key["api_key"] = self.api_key
        request_with_key["api_base"] = (
            self.api_base or "https://api.kilo.ai/api/gateway"
        )
        model = request_with_key.get("model", "")
        if model.startswith("kilo/"):
            request_with_key["model"] = "openai/" + model[len("kilo/") :]

        if stream:
            return self._stream_completion(request_with_key)
        else:
            return await self._non_stream_completion(request_with_key)

    async def _non_stream_completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = await litellm.acompletion(**request)
            return self._convert_response(response)
        except Exception as e:
            logger.error(f"Kilo completion failed: {e}")
            raise self._convert_error(e)

    async def _stream_completion(
        self, request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            stream_resp: Any = await litellm.acompletion(**request, stream=True)
            async for chunk in stream_resp:
                yield self._convert_chunk(chunk)
        except Exception as e:
            logger.error(f"Kilo streaming failed: {e}")
            raise self._convert_error(e)

    def _convert_response(self, response: Any) -> Dict[str, Any]:
        if hasattr(response, "dict"):
            return response.dict()
        elif hasattr(response, "__dict__"):
            return response.__dict__
        return response

    def _convert_chunk(self, chunk: Any) -> Dict[str, Any]:
        if hasattr(chunk, "dict"):
            return chunk.dict()
        elif hasattr(chunk, "__dict__"):
            return chunk.__dict__
        return chunk

    def _convert_error(self, error: Exception) -> HTTPException:
        error_str = str(error).lower()
        if "authentication" in error_str or "api_key" in error_str:
            status_code = 401
        elif "rate_limit" in error_str or "too many" in error_str:
            status_code = 429
        elif "invalid" in error_str or "bad request" in error_str:
            status_code = 400
        else:
            status_code = 500
        return HTTPException(status_code=status_code, detail=str(error))


class ProviderAdapterFactory:
    """Factory for creating provider adapters."""

    @staticmethod
    def create_adapter(
        provider_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_list: Optional[List[str]] = None,
    ) -> BaseProviderAdapter:
        """Create provider adapter instance."""
        provider_key = provider_name.lower()

        if provider_key == "supacoder":
            return SupacoderAdapter(provider_name, api_key, api_base, model_list)

        openai_compatible = {
            "noobrouter",
            "wiwi",
            "aihubmix",
            "opencode_zen",
            "iflow",
            "bluesminds",
        }

        if provider_key in openai_compatible:
            if provider_key == "bluesminds" and not api_base:
                api_base = "https://api.bluesminds.com/v1"
            return OpenAICompatibleAdapter(provider_name, api_key, api_base, model_list)

        adapters = {
            "groq": GroqAdapter,
            "gemini": GeminiAdapter,
            "g4f": G4FAdapter,
            "g4f_ollama": G4FAdapter,
            "g4f_pollinations": G4FAdapter,
            "g4f_nvidia": G4FAdapter,
            "g4f_gemini": G4FAdapter,
            "g4f_groq": G4FAdapter,
            "together": TogetherAdapter,
            "kilo": KiloAdapter,
        }

        adapter_class = adapters.get(provider_key)
        if adapter_class:
            return adapter_class(provider_name, api_key)

        # Dynamic config-driven: any provider with a base_url gets OpenAICompatibleAdapter
        # Adding a new OpenAI-compatible API provider only requires:
        # - Adding it to router_config.yaml under providers: with base_url and env_var
        # - Setting the API key in .env
        # No code changes needed.
        if api_base:
            return OpenAICompatibleAdapter(provider_name, api_key, api_base, model_list)

        raise ValueError(f"Unknown provider: {provider_name}. Add it to router_config.yaml with a base_url to enable automatic OpenAI-compatible routing.")

    @staticmethod
    def list_supported_providers() -> List[str]:
        """List all supported providers."""
        return [
            "groq",
            "gemini",
            "g4f",
            "g4f_ollama",
            "g4f_pollinations",
            "g4f_nvidia",
            "g4f_gemini",
            "g4f_groq",
            "together",
            "kilo",
            "noobrouter",
            "supacoder",
            "wiwi",
            "aihubmix",
            "opencode_zen",
            "iflow",
            "bluesminds",
        ]
