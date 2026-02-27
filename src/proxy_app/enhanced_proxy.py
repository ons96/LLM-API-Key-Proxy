"""Enhanced Proxy with Rate Limiting Integration

This extends the base proxy functionality with per-API-key rate limiting.
"""

import logging
from typing import Optional, Dict, Any
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse

from .rate_limiter import RateLimitMiddleware, get_rate_limiter, RateLimitConfig
from .settings_tool import get_settings

logger = logging.getLogger(__name__)


class RateLimitedProxy:
    """Proxy handler with integrated rate limiting."""
    
    def __init__(self):
        self.rate_limit_middleware: Optional[RateLimitMiddleware] = None
        self._initialized = False
    
    def initialize(self):
        """Initialize rate limiting from configuration."""
        if self._initialized:
            return
            
        try:
            settings = get_settings()
            rate_limit_config = getattr(settings, 'rate_limiting', {})
            
            # Configure default limits
            default_rpm = rate_limit_config.get('requests_per_minute', 60)
            default_burst = rate_limit_config.get('burst_size', default_rpm)
            
            config = RateLimitConfig(
                requests_per_minute=default_rpm,
                burst_size=default_burst,
                requests_per_hour=rate_limit_config.get('requests_per_hour', 1000),
                requests_per_day=rate_limit_config.get('requests_per_day', 10000)
            )
            
            rate_limiter = get_rate_limiter()
            rate_limiter.set_default_config(config)
            
            # Configure specific API keys
            key_limits = rate_limit_config.get('per_key_limits', {})
            for api_key, limits in key_limits.items():
                key_config = RateLimitConfig(
                    requests_per_minute=limits.get('rpm', default_rpm),
                    requests_per_hour=limits.get('rph', 1000),
                    requests_per_day=limits.get('rpd', 10000),
                    burst_size=limits.get('burst', limits.get('rpm', default_rpm))
                )
                rate_limiter.configure_key(api_key, key_config)
                logger.info(f"Configured rate limits for key {api_key[:8]}...: {limits.get('rpm', default_rpm)} RPM")
            
            self.rate_limit_middleware = RateLimitMiddleware(rate_limiter)
            self._initialized = True
            logger.info(f"Rate limiting initialized: {default_rpm} RPM default")
            
        except Exception as e:
            logger.error(f"Failed to initialize rate limiting: {e}")
            # Fail open - continue without rate limiting if config fails
            self.rate_limit_middleware = None
    
    async def check_rate_limit(self, request: Request) -> Optional[JSONResponse]:
        """
        Check rate limit for incoming request.
        
        Returns:
            JSONResponse if rate limited, None if allowed
        """
        if not self._initialized:
            self.initialize()
            
        if not self.rate_limit_middleware:
            return None
        
        try:
            headers = dict(request.headers)
            path = request.url.path
            
            allowed, response_headers, error_msg = await self.rate_limit_middleware.process_request(
                headers=headers,
                path=path
            )
            
            if not allowed:
                logger.warning(f"Rate limit exceeded for {headers.get('X-Forwarded-For', 'unknown')}: {error_msg}")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": {
                            "message": error_msg,
                            "type": "rate_limit_exceeded",
                            "code": "rate_limit_exceeded"
                        }
                    },
                    headers=response_headers
                )
            
            # Store headers to add to response
            request.state.rate_limit_headers = response_headers
            return None
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            # Fail open on errors
            return None
    
    def add_rate_limit_headers(self, response: Response, request: Request) -> Response:
        """Add rate limit headers to response."""
        if hasattr(request.state, 'rate_limit_headers'):
            for header, value in request.state.rate_limit_headers.items():
                response.headers[header] = value
        return response


# Singleton instance
_proxy_instance: Optional[RateLimitedProxy] = None


def get_rate_limited_proxy() -> RateLimitedProxy:
    """Get or create proxy instance with rate limiting."""
    global _proxy_instance
    if _proxy_instance is None:
        _proxy_instance = RateLimitedProxy()
        _proxy_instance.initialize()
    return _proxy_instance


async def rate_limit_dependency(request: Request):
    """
    FastAPI dependency for rate limiting.
    
    Usage:
        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request, _=Depends(rate_limit_dependency)):
            ...
    """
    proxy = get_rate_limited_proxy()
    blocked_response = await proxy.check_rate_limit(request)
    if blocked_response:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=blocked_response.body.decode() if hasattr(blocked_response, 'body') else "Rate limit exceeded",
            headers=dict(blocked_response.headers) if hasattr(blocked_response, 'headers') else {}
        )
