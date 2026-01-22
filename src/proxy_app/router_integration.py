"""
Router Integration Module

Integrates the new router system with the existing proxy infrastructure.
Maintains backward compatibility while adding new features.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, AsyncGenerator, Union

from fastapi import HTTPException, Request

from .router_core import RouterCore, CapabilityRequirements
from .provider_adapter import ProviderAdapterFactory

logger = logging.getLogger(__name__)


class RouterIntegration:
    """Integration layer between the new router and existing proxy."""
    
    def __init__(self, 
                 rotating_client: Any = None,
                 config_path: str = "config/router_config.yaml"):
        self.router = RouterCore(config_path)
        self.rotating_client = rotating_client
        self.adapter_factory = ProviderAdapterFactory()
        
        # Initialize provider adapters from environment
        self.adapters: Dict[str, Any] = {}
        self._initialize_adapters()
        
        # Start background tasks (reordering scheduler, etc.)
        self.router.start_background_tasks()
        
        logger.info(f"Router integration initialized with {len(self.adapters)} providers")
    
    def _initialize_adapters(self):
        """Initialize provider adapters from environment variables."""
        provider_configs = {
            "groq": "GROQ_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "g4f": None  # g4f doesn't need API key
        }
        
        for provider_name, env_var in provider_configs.items():
            if env_var is None:  # No API key needed
                try:
                    adapter = self.adapter_factory.create_adapter(provider_name, None)
                    self.adapters[provider_name] = adapter
                    logger.info(f"Initialized {provider_name} adapter (no API key required)")
                except Exception as e:
                    logger.warning(f"Failed to initialize {provider_name} adapter: {e}")
                continue
            
            # Check if API key is available
            import os
            api_key = os.getenv(env_var)
            
            if api_key:
                try:
                    adapter = self.adapter_factory.create_adapter(provider_name, api_key)
                    self.adapters[provider_name] = adapter
                    logger.info(f"Initialized {provider_name} adapter")
                except Exception as e:
                    logger.warning(f"Failed to initialize {provider_name} adapter: {e}")
            else:
                logger.debug(f"No API key found for {provider_name}, skipping")
    
    async def chat_completions(self, 
                             request_data: Dict[str, Any], 
                             raw_request: Request,
                             enable_logging: bool = True) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Main chat completions endpoint integrated with router."""
        
        # Generate request ID
        request_id = f"req_{uuid.uuid4().hex[:16]}"
        
        # Log request if enabled
        if enable_logging:
            logger.info(f"[{request_id}] Incoming request for model: {request_data.get('model', 'unknown')}")
            logger.debug(f"[{request_id}] Request data: {request_data}")
        
        # Check if the existing rotating client should handle this (backward compatibility)
        if self._should_use_rotating_client(request_data):
            logger.info(f"[{request_id}] Using legacy rotating client")
            return await self._handle_with_rotating_client(request_data, raw_request, request_id)
        
        # Use new router
        try:
            return await self._handle_with_router(request_data, request_id)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Router failed: {e}")
            raise HTTPException(status_code=500, detail=f"Router error: {str(e)}")
    
    def _should_use_rotating_client(self, request_data: Dict[str, Any]) -> bool:
        """Determine if request should use legacy rotating client."""
        # Use rotating client for direct provider paths that aren't in router config
        model_id = request_data.get("model", "")
        
        # If it's a virtual model, use router
        if model_id.startswith("router/"):
            return False
        
        # If provider has a dedicated adapter but is not explicitly in legacy fallback mode
        # use the router for better capability handling
        return False  # Default to using router for everything now
    
    async def _handle_with_rotating_client(self, 
                                         request_data: Dict[str, Any],
                                         raw_request: Request,
                                         request_id: str) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Handle request with legacy rotating client (backward compatibility)."""
        if not self.rotating_client:
            raise HTTPException(status_code=500, detail="Rotating client not available")
        
        # This would call the existing rotating client logic
        # For now, fall back to router
        logger.warning(f"[{request_id}] Rotating client fallback requested but using router")
        return await self._handle_with_router(request_data, request_id)
    
    async def _handle_with_router(self, 
                                request_data: Dict[str, Any],
                                request_id: str) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Handle request with new router system."""
        
        # Check if streaming is requested
        streaming = request_data.get("stream", False)
        
        # Route the request
        start_time = time.time()
        
        try:
            response = await self.router.route_request(request_data, request_id)
            
            # Handle streaming response
            if streaming:
                return self._wrap_streaming_response(response, request_id, start_time)
            else:
                # Log completion
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"[{request_id}] Request completed in {duration_ms:.1f}ms")
                return response
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Router request failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _wrap_streaming_response(self, 
                                     response: AsyncGenerator[Dict[str, Any], None],
                                     request_id: str,
                                     start_time: float) -> AsyncGenerator[Dict[str, Any], None]:
        """Wrap streaming response with logging and final stats."""
        try:
            chunk_count = 0
            async for chunk in response:
                chunk_count += 1                
                # Pass through the chunk
                yield chunk
            
            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"[{request_id}] Stream completed in {duration_ms:.1f}ms ({chunk_count} chunks)")
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"[{request_id}] Stream failed after {duration_ms:.1f}ms: {e}")
            raise
    
    def get_models(self) -> Dict[str, Any]:
        """Get available models (combines router and legacy models)."""
        
        # Get models from router
        router_models = self.router.get_model_list()
        
        # Get models from adapters
        adapter_models = []
        for provider_name, adapter in self.adapters.items():
            models = asyncio.run(adapter.list_models())
            for model in models:
                caps = adapter.get_model_capabilities(model)
                model_entry = {
                    "id": f"{provider_name}/{model}",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": provider_name
                }
                
                if caps:
                    model_entry["capabilities"] = caps.tags
                    model_entry["max_context_tokens"] = caps.max_context_tokens
                    model_entry["free_tier"] = caps.free_tier_available
                
                adapter_models.append(model_entry)
        
        # Combine and deduplicate
        all_models = router_models + adapter_models
        
        # Remove duplicates (prefer router models)
        model_lookup = {model["id"]: model for model in all_models}
        return list(model_lookup.values())
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status."""
        health_status = self.router.get_health_status()
        
        # Add adapter-specific health info
        health_status["adapters"] = {}
        for provider_name, adapter in self.adapters.items():
            health_status["adapters"][provider_name] = {
                "available": True,
                "models": len(adapter.models)
            }
        
        return health_status
    
    def refresh_configuration(self):
        """Refresh router configuration."""
        # Reinitialize router with potential new config
        self.router = RouterCore()
        self._initialize_adapters()
        logger.info("Router configuration refreshed")
    
    @property 
    def free_only_mode(self) -> bool:
        """Get FREE_ONLY_MODE status."""
        return self.router.free_only_mode


def extract_search_requirements(request_data: Dict[str, Any]) -> tuple[bool, str]:
    """Extract whether search is needed and the search query."""
    
    # Check for explicit search indicators in the last message
    messages = request_data.get("messages", [])
    if not messages:
        return False, ""
    
    last_message = messages[-1]
    content = ""
    
    if isinstance(last_message.get("content"), str):
        content = last_message["content"]
    elif isinstance(last_message.get("content"), list):
        # Extract text content
        for item in last_message["content"]:
            if isinstance(item, dict) and item.get("type") == "text":
                content = item.get("text", "")
                break
    
    # Check for search indicators
    search_indicators = [
        "latest", "recent", "current", "news", "shopping",
        "best", "top", "compare", "sources", "citations",
        "what's new", "up to date", "2024", "2025"
    ]
    
    content_lower = content.lower()
    needs_search = any(indicator in content_lower for indicator in search_indicators)
    
    # Only search if content is substantial enough
    if needs_search and len(content.split()) > 3:
        # Extract a search query (simplified - could be improved)
        search_query = content[:200]  # First 200 chars as search query
        return True, search_query
    
    return False, ""