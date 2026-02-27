```python
"""
Enhanced Proxy Application - Main FastAPI Application
"""
import time
import logging
from typing import AsyncGenerator, Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader
import json

from .router_wrapper import initialize_router, get_router
from .health_checker import HealthChecker
from .detailed_logger import DetailedLogger
from .batch_manager import EmbeddingBatcher
from .request_logger import log_request_to_console
from .model_ranker import ModelRanker
from ..rotator_library import RotatingClient
from ..rotator_library.credential_manager import CredentialManager
from ..rotator_library.background_refresher import BackgroundRefresher

# Initialize detailed logger
detailed_logger = DetailedLogger()

# Proxy API key security
proxy_api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def verify_proxy_api_key(api_key: Optional[str] = Depends(proxy_api_key_header)):
    """Verify the proxy API key if configured."""
    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    proxy_api_key = os.getenv("PROXY_API_KEY")
    if proxy_api_key and api_key != proxy_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("🚀 Starting Enhanced Proxy...")
    
    # Initialize router
    initialize_router()
    
    # Initialize health checker
    app.state.health_checker = HealthChecker()
    
    # Start background tasks
    background_refresher = BackgroundRefresher()
    asyncio.create_task(background_refresher.run_periodic_refresh())
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Enhanced Proxy...")


# Create FastAPI app
app = FastAPI(
    title="Mirro Proxy API",
    description="Unified API for multiple LLM providers with automatic failover",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}


@app.get("/v1/models")
async def list_models(api_key: Optional[str] = Depends(verify_proxy_api_key)):
    """List available models (virtual + real)."""
    router = get_router()
    
    # Get virtual models from config
    virtual_models = router.get_virtual_models()
    
    # Get real models from providers
    real_models = router.get_available_models()
    
    # Combine and format
    models = []
    
    # Add virtual models
    for vm_id, vm_config in virtual_models.items():
        models.append({
            "id": vm_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "mirro-proxy",
            "virtual": True,
        })
    
    # Add real models
    for model in real_models:
        models.append({
            "id": model.get("id", "unknown"),
            "object": "model",
            "created": model.get("created", int(time.time())),
            "owned_by": model.get("owned_by", "unknown"),
            "virtual": False,
        })
    
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Depends(verify_proxy_api_key)
):
    """
    Handle chat completion requests.
    Supports both virtual models and direct provider models.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="Missing 'model' field")
    
    router = get_router()
    
    # Check if it's a virtual model
    virtual_model_config = router.get_virtual_model_config(model)
    
    if virtual_model_config:
        # Handle virtual model - resolve to actual provider model
        actual_model, provider = router.resolve_virtual_model(model)
        
        if not actual_model or not provider:
            raise HTTPException(
                status_code=503,
                detail=f"Unable to resolve virtual model '{model}' to a provider"
            )
        
        # Update body with actual model
        body["model"] = actual_model
        
        # Log the routing
        detailed_logger.log_routing(model, actual_model, provider)
        
        # Execute with the resolved provider
        try:
            result = await router.execute_with_provider(
                provider=provider,
                model=actual_model,
                messages=body.get("messages", []),
                params=body
            )
        except Exception as e:
            # Try fallback providers
            fallback_providers = virtual_model_config.get("fallback_providers", [])
            result = None
            last_error = None
            
            for fallback_provider in fallback_providers:
                try:
                    fallback_model = virtual_model_config.get("model_mapping", {}).get(fallback_provider)
                    if fallback_model:
                        body["model"] = fallback_model
                        result = await router.execute_with_provider(
                            provider=fallback_provider,
                            model=fallback_model,
                            messages=body.get("messages", []),
                            params=body
                        )
                        detailed_logger.log_routing(model, fallback_model, fallback_provider)
                        break
                except Exception as fallback_error:
                    last_error = fallback_error
                    continue
            
            if result is None:
                raise HTTPException(
                    status_code=502,
                    detail=f"All providers failed for virtual model '{model}'. Last error: {last_error}"
                )
    else:
        # Direct model - use router's automatic selection
        try:
            result = await router.execute_chat_completion(
                model=model,
                messages=body.get("messages", []),
                params=body
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))
    
    # Handle streaming
    if body.get("stream", False):
        async def generate():
            if isinstance(result, str):
                # Already a serialized JSON string
                yield f"data: {result}\n\n"
            else:
                # Dict - serialize
                yield f"data: {json.dumps(result)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    return JSONResponse(content=result)


@app.post("/v1/embeddings")
async def embeddings(
    request: Request,
    api_key: Optional[str] = Depends(verify_proxy_api_key)
):
    """Handle embedding requests."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    model = body.get("model")
    input_text = body.get("input")
    
    if not model:
        raise HTTPException(status_code=400, detail="Missing 'model' field")
    if not input_text:
        raise HTTPException(status_code=400, detail="Missing 'input' field")
    
    router = get_router()
    
    # Batch embeddings if multiple inputs
    if isinstance(input_text, list):
        batcher = EmbeddingBatcher()
        results = await batcher.batch_embed(router, model, input_text)
        return JSONResponse(content={
            "object": "list",
            "data": results,
            "model": model
        })
    
    # Single embedding
    try:
        result = await router.execute_embedding(model, input_text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    
    return JSONResponse(content=result)


# Import asyncio for background tasks
import asyncio
```
