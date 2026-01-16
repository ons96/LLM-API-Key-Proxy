"""
Provider Adapter Module

Defines clean adapter interfaces for provider models and handles capability mapping.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
import logging
import time
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
    tags: List[str] = None

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
        """Streaming completion."""
        try:
            async for chunk in await litellm.acompletion(**request):
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
        """Initialize Gemini model capabilities."""
        gemini_models = {
            "gemini-1.5-flash": ModelCapabilities(
                provider="gemini",
                model="gemini-1.5-flash",
                supports_tools=True,
                supports_function_calling=True,
                supports_vision=True,
                supports_structured_output=True,
                supports_streaming=True,
                max_context_tokens=1048576,  # 1M tokens
                max_output_tokens=8192,
                free_tier_available=True,
                rate_limit_requests_per_minute=15,
                tags=["fast", "vision", "long_context", "general"],
            ),
            "gemini-1.5-flash-8b": ModelCapabilities(
                provider="gemini",
                model="gemini-1.5-flash-8b",
                supports_tools=True,
                supports_vision=True,
                supports_structured_output=True,
                supports_streaming=True,
                max_context_tokens=1048576,
                max_output_tokens=8192,
                free_tier_available=True,
                rate_limit_requests_per_minute=15,
                tags=["fast", "vision", "long_context", "general"],
            ),
            "gemini-1.5-pro": ModelCapabilities(
                provider="gemini",
                model="gemini-1.5-pro",
                supports_tools=True,
                supports_function_calling=True,
                supports_vision=True,
                supports_structured_output=True,
                supports_streaming=True,
                max_context_tokens=2097152,  # 2M tokens for specific use cases
                max_output_tokens=8192,
                free_tier_available=True,  # Has free tier with rate limits
                rate_limit_requests_per_minute=2,  # Lower rate limit for Pro
                tags=["reasoning", "vision", "long_context", "coding", "research"],
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
        """Streaming completion."""
        try:
            async for chunk in await litellm.acompletion(**request):
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

    def _initialize_models(self):
        """Initialize G4F model capabilities."""
        g4f_models = {
            "gpt-4": ModelCapabilities(
                provider="g4f",
                model="gpt-4",
                supports_tools=False,  # Limited tool support
                supports_streaming=True,
                max_context_tokens=8192,
                max_output_tokens=4096,
                free_tier_available=True,  # Completely free
                rate_limit_requests_per_minute=10,  # Conservative estimate
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
            model_str = request.get("model", "").replace("g4f/", "")

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
            response = await g4f.ChatCompletion.create_async(
                model=model, messages=[{"role": "user", "content": content}]
            )

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
            response = await g4f.ChatCompletion.create_async(
                model=model,
                messages=[{"role": "user", "content": content}],
                stream=True,
            )

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


class ProviderAdapterFactory:
    """Factory for creating provider adapters."""

    @staticmethod
    def create_adapter(
        provider_name: str, api_key: Optional[str] = None
    ) -> BaseProviderAdapter:
        """Create provider adapter instance."""
        adapters = {"groq": GroqAdapter, "gemini": GeminiAdapter, "g4f": G4FAdapter}

        adapter_class = adapters.get(provider_name.lower())
        if not adapter_class:
            raise ValueError(f"Unknown provider: {provider_name}")

        return adapter_class(provider_name, api_key)

    @staticmethod
    def list_supported_providers() -> List[str]:
        """List all supported providers."""
        return ["groq", "gemini", "g4f"]
