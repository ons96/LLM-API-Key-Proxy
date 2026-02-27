"""
Cache middleware for FastAPI application.
Provides response caching for specific endpoints like GET /v1/models.
Part of Phase 4.1 Response Caching.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from typing import Any, Dict, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("proxy_app")


def _env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


class ModelsCacheMiddleware(BaseHTTPMiddleware):
    """
    Middleware to cache GET /v1/models responses.
    
    Reduces load on backend providers by caching the models list
    with a configurable TTL (default 5 minutes).
    
    Environment Variables:
        MODELS_CACHE_TTL: Cache TTL in seconds (default: 300)
        MODELS_CACHE_MAX_ENTRIES: Maximum cache entries (default: 100)
        MODELS_CACHE_ENABLE: Enable/disable caching (default: true)
    """
    
    def __init__(
        self,
        app,
        ttl_seconds: Optional[int] = None,
        max_entries: Optional[int] = None,
        enabled: Optional[bool] = None
    ):
        super().__init__(app)
        self.ttl = ttl_seconds or _env_int("MODELS_CACHE_TTL", 300)
        self.max_entries = max_entries or _env_int("MODELS_CACHE_MAX_ENTRIES", 100)
        self.enabled = enabled if enabled is not None else os.getenv("MODELS_CACHE_ENABLE", "true").lower() != "false"
        
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        
        if self.enabled:
            logger.info(f"ModelsCacheMiddleware initialized (ttl={self.ttl}s, max_entries={self.max_entries})")
        else:
            logger.info("ModelsCacheMiddleware initialized (disabled)")
        
    def _get_cache_key(self, request: Request) -> str:
        """
        Generate cache key for request.
        Includes method, path, and query string, but excludes auth headers
        to ensure all users share the same models cache.
        """
        key_parts = [request.method.upper(), request.url.path]
        
        if request.url.query:
            key_parts.append(request.url.query)
            
        # Add relevant headers that might affect response (e.g., Accept version)
        accept_header = request.headers.get("accept", "")
        if "application/json" in accept_header:
            key_parts.append("json")
            
        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]
    
    def _is_cacheable(self, request: Request) -> bool:
        """Determine if request should be cached."""
        if not self.enabled:
            return False
            
        return (
            request.method.upper() == "GET" and
            request.url.path == "/v1/models"
        )
    
    async def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached response if valid."""
        async with self._lock:
            if key not in self._cache:
                return None
                
            entry = self._cache[key]
            if time.time() > entry["expires"]:
                # Expired, clean up
                del self._cache[key]
                return None
                
            self._hits += 1
            return entry["data"]
    
    async def _store_cached(self, key: str, data: Dict[str, Any]):
        """Store response in cache."""
        async with self._lock:
            # Eviction if at capacity (remove oldest by timestamp)
            if len(self._cache) >= self.max_entries:
                oldest_key = min(
                    self._cache.keys(), 
                    key=lambda k: self._cache[k]["timestamp"]
                )
                del self._cache[oldest_key]
                
            self._cache[key] = {
                "data": data,
                "timestamp": time.time(),
                "expires": time.time() + self.ttl
            }
            self._misses += 1
    
    async def dispatch(self, request: Request, call_next):
        """Handle request/response with caching logic."""
        if not self._is_cacheable(request):
            return await call_next(request)
            
        cache_key = self._get_cache_key(request)
        
        # Try cache
        cached = await self._get_cached(cache_key)
        if cached:
            logger.debug(f"Cache hit for {request.url.path}")
            return Response(
                content=cached["body"],
                status_code=cached["status_code"],
                headers=cached["headers"],
                media_type=cached.get("media_type", "application/json")
            )
        
        # Execute request
        response = await call_next(request)
        
        # Only cache successful responses with JSON content
        if (response.status_code == 200 and 
            response.media_type and 
            "json" in response.media_type):
            try:
                # Consume response body
                body_bytes = b""
                async for chunk in response.body_iterator:
                    body_bytes += chunk
                
                # Prepare cache entry (exclude sensitive headers)
                cache_data = {
                    "body": body_bytes,
                    "status_code": response.status_code,
                    "headers": {
                        k: v for k, v in response.headers.items()
                        if k.lower() not in {
                            "set-cookie", "authorization", 
                            "www-authenticate", "x-request-id"
                        }
                    },
                    "media_type": response.media_type
                }
                
                await self._store_cached(cache_key, cache_data)
                
                # Return new response since we consumed the iterator
                return Response(
                    content=body_bytes,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type
                )
                
            except Exception as e:
                logger.error(f"Failed to cache models response: {e}")
                # Return original response if caching fails
                return response
        
        return response
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total = self._hits + self._misses
            return {
                "enabled": self.enabled,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
                "entries": len(self._cache),
                "ttl_seconds": self.ttl,
                "max_entries": self.max_entries
            }
    
    async def clear(self):
        """Clear all cached entries."""
        async with self._lock:
            self._cache.clear()
            logger.info("Models cache cleared")
    
    async def invalidate(self, pattern: Optional[str] = None):
        """
        Invalidate cache entries matching pattern.
        If pattern is None, clears all entries (same as clear()).
        """
        async with self._lock:
            if pattern is None:
                self._cache.clear()
            else:
                # Simple substring matching on keys (keys are hashes, so this is limited)
                # In practice, use clear() for models endpoint
                keys_to_remove = [
                    k for k in self._cache.keys() 
                    if pattern in k
                ]
                for k in keys_to_remove:
                    del self._cache[k]
            
            logger.info(f"Cache invalidated (pattern={pattern})")
