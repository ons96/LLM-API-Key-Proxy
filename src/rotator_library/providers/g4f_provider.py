"""
G4F (GPT4Free) Provider Integration

Provides access to free LLM APIs through the g4f library.
"""

import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, Optional, Union

from . import antigravity_provider as base

logger = logging.getLogger(__name__)


class G4FProvider(base.BaseProvider):
    """G4F Provider - wrapper for GPT4Free API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize G4F Provider.

        Args:
            api_key: Optional API key (not required for G4F)
            base_url: Custom base URL for API endpoint
            **kwargs: Additional provider configuration
        """
        super().__init__(
            provider_name="g4f",
            api_key=api_key,
            base_url=base_url or "https://api.g4f.pro",
            **kwargs
        )
        self._client = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the G4F client."""
        if self._initialized:
            return

        try:
            import g4f
            self.g4f = g4f
            self._initialized = True
            logger.info("G4F provider initialized successfully")
        except ImportError:
            logger.error("g4f package not installed")
            raise RuntimeError("g4f package is required for G4F provider")

    async def close(self) -> None:
        """Close provider connections."""
        self._initialized = False
        logger.info("G4F provider closed")

    async def chat_completions(
        self,
        messages: list,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion.

        Args:
            messages: List of message dictionaries
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Response dictionary in OpenAI-compatible format
        """
        if not self._initialized:
            await self.initialize()

        try:
            response = self.g4f.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            # Convert to OpenAI format
            return self._format_response(response, model)
        except Exception as e:
            logger.error(f"G4F chat completion error: {e}")
            raise

    async def chat_completions_stream(
        self,
        messages: list,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Create a streaming chat completion.

        Args:
            messages: List of message dictionaries
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Yields:
            Response chunks in OpenAI-compatible format
        """
        if not self._initialized:
            await self.initialize()

        try:
            response = self.g4f.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )

            for chunk in response:
                yield self._format_stream_chunk(chunk, model)
        except Exception as e:
            logger.error(f"G4F stream error: {e}")
            raise

    def _format_response(self, response: Any, model: str) -> Dict[str, Any]:
        """Format G4F response to OpenAI format."""
        if isinstance(response, str):
            content = response
        elif hasattr(response, 'choices'):
            content = response.choices[0].message.content
        else:
            content = str(response)

        return {
            "id": f"g4f-chat-{asyncio.get_event_loop().time()}",
            "object": "chat.completion",
            "created": int(asyncio.get_event_loop().time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

    def _format_stream_chunk(self, chunk: Any, model: str) -> Dict[str, Any]:
        """Format streaming chunk to OpenAI format."""
        content = ""
        if isinstance(chunk, str):
            content = chunk
        elif hasattr(chunk, 'choices') and chunk.choices:
            content = chunk.choices[0].delta.content or ""

        return {
            "id": f"g4f-chat-{asyncio.get_event_loop().time()}",
            "object": "chat.completion.chunk",
            "created": int(asyncio.get_event_loop().time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "content": content
                },
                "finish_reason": None
            }]
        }

    @property
    def supported_models(self) -> list:
        """List of supported models."""
        return [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "claude-3-opus",
            "claude-3-sonnet",
            "llama-3-70b",
            "mistral-large"
        ]

    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            "name": "G4F",
            "version": "1.0",
            "models": self.supported_models,
            "features": ["streaming", "function_calls"]
        }
