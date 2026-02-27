# src/proxy_app/enhanced_proxy.py
"""
Enhanced proxy with streaming-aware timeout handling.
"""

import json
import logging
from typing import Optional, Dict, Any, AsyncGenerator
import httpx
from fastapi import Request, Response, HTTPException
from fastapi.responses import StreamingResponse

from rotator_library.timeout_config import TimeoutConfig
from rotator_library.client import is_streaming_request, get_timeout_for_request

logger = logging.getLogger(__name__)


class EnhancedProxy:
    """
    Proxy handler that applies appropriate timeouts based on streaming mode.
    """
    
    def __init__(self):
        self.session_stats = {
            "streaming_requests": 0,
            "non_streaming_requests": 0
        }
    
    async def proxy_request(
        self, 
        request: Request, 
        target_url: str, 
        provider_headers: Optional[Dict[str, str]] = None
    ) -> Response:
        """
        Proxy a request to a provider with appropriate timeout handling.
        
        Automatically detects streaming requests and applies suitable timeouts.
        """
        body = await request.body()
        is_streaming = is_streaming_request(body)
        
        # Update stats
        if is_streaming:
            self.session_stats["streaming_requests"] += 1
            logger.debug(f"Detected streaming request to {target_url}")
        else:
            self.session_stats["non_streaming_requests"] += 1
        
        # Select appropriate timeout
        timeout = get_timeout_for_request(explicit_streaming=is_streaming)
        
        # Prepare headers
        headers = self._prepare_headers(request, provider_headers)
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if is_streaming:
                    return await self._handle_streaming_request(
                        client, request, target_url, body, headers
                    )
                else:
                    return await self._handle_non_streaming_request(
                        client, request, target_url, body, headers
                    )
        except httpx.TimeoutException as e:
            logger.error(f"Timeout {'streaming' if is_streaming else 'non-streaming'} request to {target_url}: {e}")
            raise HTTPException(
                status_code=504,
                detail=f"Gateway timeout: {'streaming' if is_streaming else 'completion'} request exceeded time limit"
            )
        except Exception as e:
            logger.error(f"Error proxying request to {target_url}: {e}")
            raise HTTPException(status_code=502, detail="Bad gateway")
    
    async def _handle_streaming_request(
        self,
        client: httpx.AsyncClient,
        request: Request,
        target_url: str,
        body: bytes,
        headers: Dict[str, str]
    ) -> StreamingResponse:
        """Handle streaming request with chunked timeout handling."""
        
        async def stream_generator() -> AsyncGenerator[str, None]:
            try:
                async with client.stream(
                    method=request.method,
                    url=target_url,
                    content=body,
                    headers=headers
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_text():
                        yield chunk
            except httpx.TimeoutException:
                logger.error("Streaming timeout - no data received within chunk timeout window")
                yield "data: [ERROR] Stream timeout - connection stalled\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: [ERROR] {str(e)}\n\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    async def _handle_non_streaming_request(
        self,
        client: httpx.AsyncClient,
        request: Request,
        target_url: str,
        body: bytes,
        headers: Dict[str, str]
    ) -> Response:
        """Handle standard non-streaming request."""
        response = await client.request(
            method=request.method,
            url=target_url,
            content=body,
            headers=headers
        )
        response.raise_for_status()
        
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers)
        )
    
    def _prepare_headers(
        self, 
        request: Request, 
        provider_headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Prepare headers for upstream request."""
        headers = {}
        
        # Copy safe headers from original request
        safe_headers = ["content-type", "accept", "user-agent"]
        for header in safe_headers:
            if value := request.headers.get(header):
                headers[header] = value
        
        # Add provider-specific headers (auth, etc.)
        if provider_headers:
            headers.update(provider_headers)
        
        # Remove problematic headers
        headers.pop("host", None)
        headers.pop("content-length", None)  # httpx will recalculate
        
        return headers
    
    def get_stats(self) -> Dict[str, int]:
        """Get session statistics."""
        return self.session_stats.copy()


class TimeoutAwareRouter:
    """
    Router component that ensures providers respect streaming timeouts.
    """
    
    @staticmethod
    def validate_streaming_timeout_config():
        """
        Validate that timeout configuration is suitable for expected workloads.
        Logs warnings if timeouts seem misconfigured.
        """
        streaming_read = TimeoutConfig.read_streaming()
        non_streaming_read = TimeoutConfig.read_non_streaming()
        
        if streaming_read > 300:
            logger.warning(
                f"Streaming read timeout ({streaming_read}s) is quite high. "
                "Consider reducing to detect stalled connections faster."
            )
        
        if non_streaming_read < 60:
            logger.warning(
                f"Non-streaming read timeout ({non_streaming_read}s) is very low. "
                "Long completions may fail."
            )
        
        logger.info(
            f"Timeout configuration - Streaming: {streaming_read}s, "
            f"Non-streaming: {non_streaming_read}s"
        )
