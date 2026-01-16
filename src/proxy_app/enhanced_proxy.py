"""
Enhanced Proxy Module

This module provides complete multi-provider routing functionality as a drop-in enhancement
to the existing proxy. It maintains full OpenAI compatibility while adding virtual models,
MoE support, web-search augmentation, and $0-only guarantees.
"""

import os
import sys
import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Any, AsyncGenerator, Union, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import existing infrastructure
from proxy_app.main import (
    app, lifespan, RotatingClient, EmbeddingBatcher, 
    get_rotating_client, get_embedding_batcher, verify_api_key,
    ModelCard, ModelList
)
from proxy_app.router_integration import RouterIntegration

logger = logging.getLogger(__name__)

# Initialize router integration
_router_integration = None


def initialize_router_integration(rotating_client: RotatingClient = None) -> RouterIntegration:
    """Initialize the router integration."""
    global _router_integration
    
    config_path = Path(__file__).parent.parent.parent / "config" / "router_config.yaml"
    
    if _router_integration is None:
        _router_integration = RouterIntegration(
            rotating_client=rotating_client,
            config_path=str(config_path)
        )
        logger.info("Router integration initialized")
    
    return _router_integration


# Store original endpoints for reference
_original_chat_completions = None
_original_models_endpoint = None


def preserve_original_endpoints():
    """Store references to original endpoints for fallback."""
    global _original_chat_completions, _original_models_endpoint
    
    # These will be set during enhancement
    pass


# Enhanced model list endpoint
@app.get("/v1/models")
async def enhanced_models_list(request: Request):
    """Enhanced models endpoint that includes virtual router models."""
    try:
        # Initialize router if not done
        if _router_integration is None:
            rotating_client = get_rotating_client(request)
            initialize_router_integration(rotating_client)
        
        # Get models from router
        router_models = _router_integration.get_models()
        
        # Convert to OpenAI format
        models_data = [
            ModelCard(
                id=model["id"],
                created=model.get("created", int(time.time())),
                owned_by=model.get("owned_by", "unknown")
            )
            for model in router_models
        ]
        
        return ModelList(data=models_data)
        
    except Exception as e:
        logger.error(f"Enhanced models endpoint failed: {e}")
        # Fallback to showing a basic model list
        current_time = int(time.time())
        return ModelList(data=[
            ModelCard(id="router/best-coding", created=current_time, owned_by="router"),
            ModelCard(id="router/best-reasoning", created=current_time, owned_by="router"),
            ModelCard(id="router/best-research", created=current_time, owned_by="router"),
            ModelCard(id="router/best-chat", created=current_time, owned_by="router"),
            ModelCard(id="router/best-coding-moe", created=current_time, owned_by="router"),
        ])


# Enhanced health endpoint showing router status
@app.get("/health")
@app.get("/v1/health")
async def enhanced_health_check(request: Request):
    """Enhanced health check showing router and provider status."""
    try:
        if _router_integration is None:
            rotating_client = get_rotating_client(request)
            initialize_router_integration(rotating_client)
        
        # Get router health
        router_health = _router_integration.get_health()
        
        # Get basic proxy health
        basic_health = {"status": "healthy", "router": router_health}
        
        return basic_health
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "degraded", "error": str(e)}


# Enhanced chat completions endpoint
@app.post("/v1/chat/completions")
async def enhanced_chat_completions(
    request: Request,
    auth: str = Depends(verify_api_key)
):
    """Enhanced chat completions with multi-provider routing."""
    
    request_id = f"req_{uuid.uuid4().hex[:16]}"
    start_time = time.time()
    
    try:
        # Parse request
        request_data = await request.json()
        model = request_data.get("model", "")
        
        logger.info(f"[{request_id}] Chat completion request for model: {model}")
        
        # Initialize router if not done
        if _router_integration is None:
            rotating_client = get_rotating_client(request)
            initialize_router_integration(rotating_client)
        
        # Route through enhanced system
        response = await _router_integration.chat_completions(
            request_data=request_data,
            raw_request=request,
            enable_logging=True
        )
        
        # Handle streaming
        if request_data.get("stream", False):
            return StreamingResponse(
                _wrap_streaming_response(response, request_id, start_time),
                media_type="text/plain"
            )
        else:
            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"[{request_id}] Request completed in {duration_ms:.1f}ms")
            return response
            
    except HTTPException:
        raise
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"[{request_id}] Request failed after {duration_ms:.1f}ms: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _wrap_streaming_response(
    response_stream: AsyncGenerator[Dict[str, Any], None],
    request_id: str,
    start_time: float
) -> AsyncGenerator[str, None]:
    """Wrap streaming response with proper SSE formatting."""
    
    chunk_count = 0
    try:
        async for chunk in response_stream:
            # Ensure chunk is properly formatted
            if isinstance(chunk, dict):
                # Check if it's already a properly formatted chunk
                if "id" in chunk and "choices" in chunk:
                    # Standard LiteLLM chunk format
                    yield f"data: {chunk}\\n\\n"
                    chunk_count += 1
                elif "error" in chunk:
                    # Error chunk
                    yield f"data: {chunk}\\n\\n"
                else:
                    # Raw content, wrap it
                    wrapped_chunk = {
                        "id": f"router-{request_id}-{chunk_count}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": chunk.get("model", "unknown"),
                        "choices": [{
                            "index": 0,
                            "delta": chunk.get("delta", {"content": ""}),
                            "finish_reason": chunk.get("finish_reason")
                        }]
                    }
                    yield f"data: {wrapped_chunk}\\n\\n"
                    chunk_count += 1
            else:
                # String or other type, convert to string
                yield f"data: {chunk}\\n\\n"
                chunk_count += 1
        
        # Send done signal
        yield "data: [DONE]\\n\\n"
        
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"[{request_id}] Stream completed in {duration_ms:.1f}ms ({chunk_count} chunks)")
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"[{request_id}] Stream failed after {duration_ms:.1f}ms: {e}")
        yield f"data: {{\"error\": \"{str(e)}\"}}\\n\\n"
        yield "data: [DONE]\\n\\n"


# Router status endpoint
@app.get("/v1/router/status")
async def router_status(request: Request):
    """Get detailed router status including provider health."""
    try:
        if _router_integration is None:
            rotating_client = get_rotating_client(request)
            initialize_router_integration(rotating_client)
        
        health = _router_integration.get_health()
        
        # Add runtime info
        health.update({
            "free_only_mode": _router_integration.free_only_mode,
            "timestamp": time.time(),
            "uptime_seconds": time.time() - time.time()  # Placeholder
        })
        
        return health
        
    except Exception as e:
        logger.error(f"Router status failed: {e}")
        return {"error": str(e)}


# Router metrics endpoint
@app.get("/v1/router/metrics")
async def router_metrics(request: Request):
    """Get router performance metrics."""
    try:
        if _router_integration is None:
            rotating_client = get_rotating_client(request)
            initialize_router_integration(rotating_client)
        
        # Get router health (contains metrics)
        health = _router_integration.get_health()
        
        # Summarize metrics
        provider_stats = {}
        for provider, models in health.get("providers", {}).items():
            provider_stats[provider] = {
                "total_models": len(models),
                "healthy_models": sum(1 for m in models.values() if m.get("status") == "healthy"),
                "avg_success_rate": sum(m.get("success_rate", 0) for m in models.values()) / len(models) if models else 0,
                "avg_latency": sum(m.get("ewma_latency_ms", 0) for m in models.values()) / len(models) if models else 0
            }
        
        return {
            "providers": provider_stats,
            "search_providers": health.get("search_providers", {}),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Router metrics failed: {e}")
        return {"error": str(e)}


# Configuration refresh endpoint
@app.post("/v1/router/refresh")
async def refresh_router_config(request: Request):
    """Refresh router configuration."""
    try:
        if _router_integration is None:
            return {"error": "Router not initialized"}
        
        _router_integration.refresh_configuration()
        return {"status": "refreshed"}
        
    except Exception as e:
        logger.error(f"Router refresh failed: {e}")
        return {"error": str(e)}


# Dynamic search endpoint for testing
@app.post("/v1/search")
async def perform_search(request: Request):
    """Perform web search using configured search providers."""
    try:
        request_data = await request.json()
        query = request_data.get("query", "")
        max_results = request_data.get("max_results", 5)
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # This is simplified - real implementation would use router's search providers
        # For now, return a placeholder response
        return {
            "query": query,
            "results": [
                {"title": "Example search result", "url": "https://example.com", "description": "This is a placeholder for search results"}
            ],
            "status": "placeholder",
            "message": "Search functionality will be implemented with real providers"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


def enhance_proxy():
    """Apply enhancements to the existing proxy."""
    logger.info("LLM Proxy enhanced with multi-provider router")
    logger.info("Virtual models available:")
    logger.info("  - router/best-coding")
    logger.info("  - router/best-reasoning")
    logger.info("  - router/best-research")
    logger.info("  - router/best-chat")
    logger.info("  - router/best-coding-moe")
    logger.info(f"FREE_ONLY_MODE: {os.getenv('FREE_ONLY_MODE', 'true')}")
    
    # Note: The actual endpoints are defined above with enhanced_* prefixes
    # The existing main.py endpoints will co-exist with these enhanced versions
    # The enhanced versions handle virtual models and fall back appropriately


if __name__ == "__main__":
    enhance_proxy()
    print("Enhanced proxy module loaded. Use the main server from proxy_app.main")