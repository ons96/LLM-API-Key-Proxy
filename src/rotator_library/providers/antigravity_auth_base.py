# src/rotator_library/providers/antigravity_auth_base.py
"""
Base class for Antigravity providers with streaming timeout support.
"""

import httpx
from typing import Optional, Dict, Any, AsyncGenerator
import logging

from ..timeout_config import TimeoutConfig
from ..client import create_client, is_streaming_request

logger = logging.getLogger(__name__)


class AntigravityAuthBase:
    """
    Base class for Antigravity providers implementing timeout-aware HTTP handling.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self._client: Optional[httpx.AsyncClient] = None
        self._streaming_client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self, streaming: bool = False) -> httpx.AsyncClient:
        """
        Get or create HTTP client with appropriate timeout configuration.
        
        Maintains separate clients for streaming and non-streaming to avoid
        timeout configuration conflicts.
        """
        if streaming:
            if self._streaming_client is None or self._streaming_client.is_closed:
                self._streaming_client = create_client(streaming=True, headers=self._get_headers())
            return self._streaming_client
        else:
            if self._client is None or self._client.is_closed:
                self._client = create_client(streaming=False, headers=self._get_headers())
            return self._client
    
    def _get_headers(self) -> Dict[str, str]:
        """Get default headers for requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def make_request(
        self, 
        endpoint: str, 
        payload: Dict[str, Any],
        stream: bool = False
    ) -> httpx.Response:
        """
        Make a request with appropriate timeout handling.
        
        Args:
            endpoint: API endpoint path
            payload: Request payload
            stream: Whether this is a streaming request
            
        Returns:
            httpx.Response object
        """
        client = await self._get_client(streaming=stream)
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = await client.post(url, json=payload)
            return response
        except httpx.TimeoutException as e:
            logger.error(f"Timeout during {'streaming' if stream else 'standard'} request to {url}: {e}")
            raise
    
    async def stream_request(
        self,
        endpoint: str,
        payload: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Make a streaming request with streaming-appropriate timeouts.
        
        Args:
            endpoint: API endpoint path
            payload: Request payload requiring streaming response
            
        Yields:
            Response chunks as strings
        """
        # Ensure streaming is enabled in payload
        payload["stream"] = True
        
        # Use streaming client
        client = await self._get_client(streaming=True)
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for chunk in response.aiter_text():
                    yield chunk
        except httpx.TimeoutException:
            logger.error(f"Streaming timeout on {url} - no chunks received within {TimeoutConfig.read_streaming()}s")
            raise
        except Exception as e:
            logger.error(f"Streaming error on {url}: {e}")
            raise
    
    async def close(self):
        """Close all HTTP clients."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        if self._streaming_client and not self._streaming_client.is_closed:
            await self._streaming_client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
