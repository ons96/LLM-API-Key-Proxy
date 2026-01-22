"""
Router Wrapper for FastAPI Integration

This module provides a clean wrapper to integrate the router with the existing FastAPI endpoints.
"""

import logging
import time
import asyncio
from typing import Dict, Any, AsyncGenerator, Union
from fastapi import HTTPException, Request

from .router_integration import RouterIntegration

logger = logging.getLogger(__name__)


class RouterWrapper:
    """Wrapper class to handle router integration with existing endpoints."""
    
    def __init__(self, rotating_client: Any = None):
        self.router_integration = RouterIntegration(rotating_client)
        self._initialized = True
        logger.info("RouterWrapper initialized with router integration")
    
    async def handle_chat_completions(self, 
                                    request_data: Dict[str, Any], 
                                    raw_request: Request) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Handle chat completions with router integration."""
        
        # Check if we should use legacy path (when no virtual models are specified)
        model_id = request_data.get("model", "")
        
        # Always use router for virtual models
        if model_id.startswith("router/"):
            logger.info(f"Virtual model requested: {model_id}")
            return await self.router_integration.chat_completions(request_data, raw_request)
        
        # For specific provider/model combinations, still use router
        if "/" in model_id:
            provider = model_id.split("/", 1)[0]
            if provider in ["groq", "gemini", "g4f"]:
                logger.info(f"Routed model requested: {model_id}")
                return await self.router_integration.chat_completions(request_data, raw_request)
        
        # For generic models, use router as well (it will handle provider selection)
        logger.info(f"Model requested via router: {model_id}")
        return await self.router_integration.chat_completions(request_data, raw_request)
    
    def get_models(self) -> Dict[str, Any]:
        """Get available models."""
        return self.router_integration.get_models()
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status."""
        return self.router_integration.get_health()
    
    @property 
    def free_only_mode(self) -> bool:
        """Get FREE_ONLY_MODE status."""
        return self.router_integration.free_only_mode


# Global instance
_router_wrapper = None


def initialize_router(rotating_client: Any = None):
    """Initialize the global router wrapper."""
    global _router_wrapper
    _router_wrapper = RouterWrapper(rotating_client)
    logger.info("Global router wrapper initialized")


def get_router() -> RouterWrapper:
    """Get the global router wrapper instance."""
    if _router_wrapper is None:
        raise RuntimeError("Router not initialized. Call initialize_router() first.")
    return _router_wrapper


# Compatibility functions for direct integration
async def chat_completions_with_router(
    request_data: Dict[str, Any],
    raw_request: Request,
    rotating_client: Any = None
) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
    """Compatibility function for chat completions with router."""
    
    # Initialize router if not already done
    try:
        router = get_router()
    except RuntimeError:
        initialize_router(rotating_client)
        router = get_router()
    
    return await router.handle_chat_completions(request_data, raw_request)