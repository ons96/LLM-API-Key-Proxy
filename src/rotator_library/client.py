# src/rotator_library/client.py
"""
HTTP client factory with centralized timeout configuration.
"""

import httpx
from typing import Optional, Dict, Any, AsyncGenerator
import json
import logging

from .timeout_config import TimeoutConfig

logger = logging.getLogger(__name__)


class StreamingClient:
    """
    HTTP client wrapper that applies appropriate timeouts based on streaming mode.
    """
    
    def __init__(self, streaming: bool = False, headers: Optional[Dict[str, str]] = None):
        self.streaming = streaming
        self.headers = headers or {}
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        timeout = TimeoutConfig.streaming() if self.streaming else TimeoutConfig.non_streaming()
        self._client = httpx.AsyncClient(timeout=timeout, headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
    
    async def post(self, url: str, **kwargs) -> httpx.Response:
        """Make POST request with configured timeout."""
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        return await self._client.post(url, **kwargs)
    
    async def get(self, url: str, **kwargs) -> httpx.Response:
        """Make GET request with configured timeout."""
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        return await self._client.get(url, **kwargs)
    
    async def stream(self, method: str, url: str, **kwargs) -> AsyncGenerator[bytes, None]:
        """
        Stream response with streaming-appropriate timeouts.
        Forces streaming timeout configuration regardless of initial mode.
        """
        if not self._client:
            # Create client with streaming timeout if streaming content
            timeout = TimeoutConfig.streaming()
            self._client = httpx.AsyncClient(timeout=timeout, headers=self.headers)
        
        async with self._client.stream(method, url, **kwargs) as response:
            async for chunk in response.aiter_bytes():
                yield chunk


def create_client(streaming: bool = False, headers: Optional[Dict[str, str]] = None) -> httpx.AsyncClient:
    """
    Factory function to create an httpx.AsyncClient with appropriate timeout configuration.
    
    Args:
        streaming: Whether this client will be used for streaming requests
        headers: Default headers to include
        
    Returns:
        Configured httpx.AsyncClient instance
    """
    timeout = TimeoutConfig.streaming() if streaming else TimeoutConfig.non_streaming()
    return httpx.AsyncClient(timeout=timeout, headers=headers or {})


def is_streaming_request(body: bytes) -> bool:
    """
    Detect if a request body indicates streaming mode.
    
    Checks for OpenAI-compatible stream parameter in JSON body.
    
    Args:
        body: Raw request body bytes
        
    Returns:
        True if streaming is requested
    """
    try:
        if not body:
            return False
        data = json.loads(body)
        return data.get("stream", False) is True
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return False


def get_timeout_for_request(body: Optional[bytes] = None, explicit_streaming: Optional[bool] = None) -> httpx.Timeout:
    """
    Determine appropriate timeout configuration based on request characteristics.
    
    Args:
        body: Request body to inspect for streaming flag
        explicit_streaming: Explicit streaming flag if already known
        
    Returns:
        httpx.Timeout configuration
    """
    if explicit_streaming is not None:
        return TimeoutConfig.streaming() if explicit_streaming else TimeoutConfig.non_streaming()
    
    if body and is_streaming_request(body):
        return TimeoutConfig.streaming()
    
    return TimeoutConfig.non_streaming()
