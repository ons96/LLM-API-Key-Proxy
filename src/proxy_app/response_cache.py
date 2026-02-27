"""Response caching module for API requests."""
import hashlib
import json
import time
import asyncio
import logging
from typing import Any, Dict, Optional, Union, List
from collections import OrderedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float
    ttl: int
    hits: int = 0
    model: Optional[str] = None


class ResponseCache:
    """Async-safe LRU cache with TTL support for API responses."""
    
    def __init__(
        self, 
        max_size: int = 1000, 
        default_ttl: int = 3600,
        excluded_models: Optional[List[str]] = None
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.excluded_models = set(excluded_models or [])
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
    def _generate_cache_key(self, request_body: Dict[str, Any]) -> str:
        """
        Generate a deterministic cache key from request body.
        Excludes non-deterministic fields like 'user', 'stream', 'request_id'.
        """
        # Create cacheable copy
        cache_body = {}
        
        for k, v in request_body.items():
            # Skip fields that don't affect output
            if k in ('user', 'stream', 'request_id', 'metadata', 'seed'):
                continue
            cache_body[k] = v
            
        # Handle special case: messages with potential timestamps or IDs
        if 'messages' in cache_body:
            # Normalize messages (remove trailing whitespace, etc.)
            normalized_messages = []
            for msg in cache_body['messages']:
                norm_msg = {
                    'role': msg.get('role'),
                    'content': msg.get('content', '').rstrip() if isinstance(msg.get('content'), str) else msg.get('content')
                }
                # Include tool_calls if present (for function calling)
                if 'tool_calls' in msg:
                    norm_msg['tool_calls'] = msg['tool_calls']
                if 'name' in msg:
                    norm_msg['name'] = msg['name']
                normalized_messages.append(norm_msg)
            cache_body['messages'] = normalized_messages
            
        # Sort keys and create deterministic JSON
        try:
            key_str = json.dumps(cache_body, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to serialize request for cache key: {e}")
            key_str = str(hash(str(cache_body)))
            
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()
    
    def should_cache_request(self, request_body: Dict[str, Any]) -> bool:
        """Determine if this request should be cached."""
        # Don't cache streaming requests
        if request_body.get('stream', False):
            return False
            
        # Don't cache if temperature is random (0 means deterministic)
        temp = request_body.get('temperature')
        if temp is not None and temp != 0:
            return False
            
        # Check excluded models
        model = request_body.get('model', '')
        if model in self.excluded_models:
            return False
            
        # Don't cache if explicitly requested to skip
        if request_body.get('cache', True) is False:
            return False
            
        return True
    
    async def get(self, request_body: Dict[str, Any]) -> Optional[Any]:
        """Get cached response if available and not expired."""
        if not self.should_cache_request(request_body):
            return None
            
        cache_key = self._generate_cache_key(request_body)
        
        async with self._lock:
            if cache_key not in self._cache:
                self._misses += 1
                return None
                
            entry = self._cache[cache_key]
            
            # Check TTL
            if time.time() - entry.timestamp > entry.ttl:
                # Expired
                del self._cache[cache_key]
                self._misses += 1
                return None
            
            # Cache hit
            entry.hits += 1
            self._hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            
            logger.debug(f"Cache hit for key {cache_key[:8]}... (model: {entry.model})")
            return entry.value
    
    async def set(
        self, 
        request_body: Dict[str, Any], 
        value: Any, 
        ttl: Optional[int] = None
    ) -> None:
        """Store response in cache."""
        if not self.should_cache_request(request_body):
            return
            
        if ttl is None:
            ttl = self.default_ttl
            
        cache_key = self._generate_cache_key(request_body)
        model = request_body.get('model', 'unknown')
        
        async with self._lock:
            # Evict oldest entries if at capacity
            while len(self._cache) >= self.max_size:
                try:
                    self._cache.popitem(last=False)
                    self._evictions += 1
                except KeyError:
                    break
            
            self._cache[cache_key] = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl,
                model=model
            )
            
            logger.debug(f"Cached response for key {cache_key[:8]}... (model: {model})")
    
    async def invalidate_model(self, model: str) -> int:
        """Invalidate all cache entries for a specific model."""
        removed = 0
        async with self._lock:
            keys_to_remove = [
                k for k, v in self._cache.items() 
                if v.model == model
            ]
            for k in keys_to_remove:
                del self._cache[k]
                removed += 1
        if removed > 0:
            logger.info(f"Invalidated {removed} cache entries for model {model}")
        return removed
    
    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info(f"Cleared {count} entries from response cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate_percent": round(hit_rate, 2),
            "default_ttl_seconds": self.default_ttl,
            "excluded_models_count": len(self.excluded_models)
        }


# Global instance
_response_cache: Optional[ResponseCache] = None


def init_cache(
    max_size: int = 1000, 
    default_ttl: int = 3600,
    excluded_models: Optional[List[str]] = None
) -> ResponseCache:
    """Initialize the global response cache."""
    global _response_cache
    _response_cache = ResponseCache(
        max_size=max_size,
        default_ttl=default_ttl,
        excluded_models=excluded_models
    )
    logger.info(f"Initialized response cache (max_size={max_size}, ttl={default_ttl})")
    return _response_cache


def get_cache() -> Optional[ResponseCache]:
    """Get the global cache instance."""
    return _response_cache


async def cache_middleware(request: Request, call_next):
    """FastAPI middleware for automatic response caching."""
    from fastapi.responses import JSONResponse
    
    cache = get_cache()
    
    # Only cache specific paths
    if not cache or request.url.path not in ("/v1/chat/completions", "/v1/embeddings"):
        return await call_next(request)
    
    # Only cache POST requests
    if request.method != "POST":
        return await call_next(request)
    
    # Read body
    body = await request.body()
    if not body:
        return await call_next(request)
        
    try:
        body_json = json.loads(body)
    except json.JSONDecodeError:
        return await call_next(request)
    
    # Check cache
    cached = await cache.get(body_json)
    if cached:
        return JSONResponse(content=cached)
    
    # Process request
    response = await call_next(request)
    
    # Cache successful responses (status 200)
    if response.status_code == 200:
        # Read response body
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk
        
        try:
            response_json = json.loads(response_body)
            await cache.set(body_json, response_json)
        except (json.JSONDecodeError, Exception):
            pass
            
        # Return new response with body
        return JSONResponse(
            content=json.loads(response_body),
            status_code=response.status_code,
            headers=dict(response.headers)
        )
    
    return response
