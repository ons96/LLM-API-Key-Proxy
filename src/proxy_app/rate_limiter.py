"""Rate Limiter for API Key-based Request Throttling

Implements token bucket algorithm for distributed rate limiting per API key.
Supports configurable limits via settings and provides standard rate limit headers.
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RateLimitScope(Enum):
    """Scopes for rate limiting."""
    GLOBAL = "global"
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: Optional[int] = None  # If None, uses requests_per_minute
    
    def __post_init__(self):
        if self.burst_size is None:
            self.burst_size = self.requests_per_minute


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    tokens: float
    last_update: float
    config: RateLimitConfig
    
    @property
    def rate_per_second(self) -> float:
        """Calculate token refill rate based on per-minute limit."""
        return self.config.requests_per_minute / 60.0


class RateLimiter:
    """Token bucket rate limiter with per-API-key tracking."""
    
    def __init__(self, cleanup_interval: int = 300):
        self.buckets: Dict[str, TokenBucket] = {}
        self.configs: Dict[str, RateLimitConfig] = {}  # Per-API-key configs
        self.default_config = RateLimitConfig()
        self.cleanup_interval = cleanup_interval
        self._lock = asyncio.Lock()
        self._last_cleanup = time.time()
        
    def configure_key(self, api_key: str, config: RateLimitConfig):
        """Configure custom rate limits for a specific API key."""
        self.configs[api_key] = config
        # Reset bucket to apply new config
        if api_key in self.buckets:
            del self.buckets[api_key]
    
    def set_default_config(self, config: RateLimitConfig):
        """Set default rate limit configuration."""
        self.default_config = config
    
    def _get_bucket_key(self, api_key: str, scope: RateLimitScope = RateLimitScope.GLOBAL) -> str:
        """Generate storage key for bucket."""
        return f"{api_key}:{scope.value}"
    
    async def check_rate_limit(
        self, 
        api_key: str, 
        scope: RateLimitScope = RateLimitScope.GLOBAL,
        cost: float = 1.0
    ) -> Tuple[bool, Dict[str, str], Optional[str]]:
        """
        Check if request is within rate limit.
        
        Args:
            api_key: The API key to check
            scope: Rate limiting scope (global, chat, etc.)
            cost: Token cost for this request (default 1.0)
            
        Returns:
            Tuple of (allowed, headers, error_message)
            Headers include X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset
        """
        async with self._lock:
            now = time.time()
            
            # Cleanup old entries periodically
            if now - self._last_cleanup > self.cleanup_interval:
                self._cleanup_old_buckets(now)
                self._last_cleanup = now
            
            key = self._get_bucket_key(api_key, scope)
            config = self.configs.get(api_key, self.default_config)
            
            # Initialize bucket if doesn't exist
            if key not in self.buckets:
                self.buckets[key] = TokenBucket(
                    tokens=config.burst_size - cost,
                    last_update=now,
                    config=config
                )
                
                reset_time = int(now + 60)  # Reset window is 1 minute for RPM
                headers = self._build_headers(
                    limit=config.requests_per_minute,
                    remaining=int(config.burst_size - cost),
                    reset_time=reset_time
                )
                return True, headers, None
            
            bucket = self.buckets[key]
            
            # Refill tokens based on time passed
            time_passed = now - bucket.last_update
            tokens_to_add = time_passed * bucket.rate_per_second
            bucket.tokens = min(config.burst_size, bucket.tokens + tokens_to_add)
            bucket.last_update = now
            
            # Check if we have enough tokens
            if bucket.tokens >= cost:
                bucket.tokens -= cost
                remaining = max(0, int(bucket.tokens))
                
                # Calculate reset time based on when bucket will be full again
                seconds_to_refill = (config.burst_size - bucket.tokens) / bucket.rate_per_second
                reset_time = int(now + seconds_to_refill)
                
                headers = self._build_headers(
                    limit=config.requests_per_minute,
                    remaining=remaining,
                    reset_time=reset_time
                )
                return True, headers, None
            else:
                # Rate limit exceeded
                seconds_until_ready = (cost - bucket.tokens) / bucket.rate_per_second
                reset_time = int(now + seconds_until_ready)
                
                headers = self._build_headers(
                    limit=config.requests_per_minute,
                    remaining=0,
                    reset_time=reset_time
                )
                error_msg = (
                    f"Rate limit exceeded. Limit: {config.requests_per_minute} requests per minute. "
                    f"Retry after {int(seconds_until_ready)} seconds."
                )
                return False, headers, error_msg
    
    def _build_headers(self, limit: int, remaining: int, reset_time: int) -> Dict[str, str]:
        """Build rate limit headers."""
        return {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_time),
            "Retry-After": str(max(0, reset_time - int(time.time())))
        }
    
    def _cleanup_old_buckets(self, now: float):
        """Remove buckets that haven't been used recently."""
        cutoff = now - self.cleanup_interval * 2
        expired_keys = [
            key for key, bucket in self.buckets.items() 
            if bucket.last_update < cutoff
        ]
        for key in expired_keys:
            del self.buckets[key]
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired rate limit buckets")
    
    def get_stats(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Get current rate limit statistics."""
        if api_key:
            key = self._get_bucket_key(api_key)
            bucket = self.buckets.get(key)
            if bucket:
                return {
                    "api_key": api_key[:8] + "...",  # Mask key
                    "current_tokens": bucket.tokens,
                    "last_update": bucket.last_update,
                    "config": {
                        "rpm": bucket.config.requests_per_minute,
                        "burst": bucket.config.burst_size
                    }
                }
            return {"error": "No data for key"}
        
        return {
            "total_buckets": len(self.buckets),
            "configured_keys": len(self.configs),
            "default_rpm": self.default_config.requests_per_minute
        }


# Global rate limiter instance
_rate_limiter_instance: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter instance."""
    global _rate_limiter_instance
    if _rate_limiter_instance is None:
        _rate_limiter_instance = RateLimiter()
    return _rate_limiter_instance


def configure_rate_limiter(default_config: Optional[RateLimitConfig] = None, cleanup_interval: int = 300):
    """Configure the global rate limiter."""
    global _rate_limiter_instance
    _rate_limiter_instance = RateLimiter(cleanup_interval=cleanup_interval)
    if default_config:
        _rate_limiter_instance.set_default_config(default_config)
    return _rate_limiter_instance


class RateLimitMiddleware:
    """FastAPI-style middleware for rate limiting."""
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        self.rate_limiter = rate_limiter or get_rate_limiter()
    
    async def process_request(self, headers: Dict[str, str], path: str = "") -> Tuple[bool, Dict[str, str], Optional[str]]:
        """
        Process request and check rate limits.
        
        Returns:
            (allowed, headers_to_add, error_message)
        """
        api_key = self._extract_api_key(headers)
        
        # Determine scope based on path
        scope = RateLimitScope.GLOBAL
        if "/chat" in path or "/v1/chat" in path:
            scope = RateLimitScope.CHAT
        elif "/embeddings" in path:
            scope = RateLimitScope.EMBEDDING
        elif "/completions" in path:
            scope = RateLimitScope.COMPLETION
            
        allowed, response_headers, error = await self.rate_limiter.check_rate_limit(
            api_key=api_key,
            scope=scope
        )
        
        return allowed, response_headers, error
    
    def _extract_api_key(self, headers: Dict[str, str]) -> str:
        """Extract API key from headers."""
        # Try Authorization header first (Bearer token)
        auth = headers.get("Authorization", headers.get("authorization", ""))
        if auth.startswith("Bearer "):
            return auth[7:].strip()
        elif auth.startswith("bearer "):
            return auth[7:].strip()
        
        # Try X-API-Key header
        api_key = headers.get("X-API-Key", headers.get("x-api-key", ""))
        if api_key:
            return api_key.strip()
        
        # Fallback to IP-based limiting (X-Forwarded-For or X-Real-IP)
        ip = headers.get("X-Forwarded-For", headers.get("X-Real-IP", "anonymous"))
        if "," in ip:
            ip = ip.split(",")[0].strip()
        return f"ip:{ip}"
    
    def configure_key_limits(self, api_key: str, requests_per_minute: int, 
                           requests_per_hour: int = 1000, 
                           requests_per_day: int = 10000):
        """Configure custom limits for a specific API key."""
        config = RateLimitConfig(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
            requests_per_day=requests_per_day
        )
        self.rate_limiter.configure_key(api_key, config)
