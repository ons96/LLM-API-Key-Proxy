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
from .semantic_router import resolve_auto

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
        
        # Semantic auto-router: resolve `model="auto"` to a concrete chain.
        # Runs before any other routing so tool-capability guard + intent
        # classification see the full request. Mutates request_data["model"]
        # in place; downstream code is unaware that "auto" was ever requested.
        if model_id == "auto" or model_id == "router/auto":
            try:
                result = resolve_auto(request_data, current_model=model_id)
                logger.debug(
                    "auto-router: intent=%s chain=%s source=%s conf=%.2f",
                    result.intent.name, result.chain, result.source, result.confidence,
                )
                request_data["model"] = result.chain
                model_id = result.chain
            except Exception as exc:
                # Tool-aware fallback: if request carries tools, fall back to a
                # tool-capable chain instead of chat-fast (which is non-tool).
                # Never silently serve a non-tool chain when client asked for tools.
                from .semantic_router import DEFAULT_CHAIN, DEFAULT_TOOL_CHAIN
                needs_tools = bool(request_data.get("tools") or request_data.get("tool_choice"))
                fallback = DEFAULT_TOOL_CHAIN if needs_tools else DEFAULT_CHAIN
                logger.warning(
                    "auto-router failed (%r); falling back to %s", exc, fallback
                )
                request_data["model"] = fallback
                model_id = fallback
        
        # Always use router for virtual models
        if model_id.startswith("router/"):
            logger.info(f"Virtual model requested: {model_id}")
            return await self.router_integration.chat_completions(request_data, raw_request)
        
        # For specific provider/model combinations, still use router
        if "/" in model_id:
            provider = model_id.split("/", 1)[0]
            if True:  # All providers route through integration (config-driven + hardcoded)
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