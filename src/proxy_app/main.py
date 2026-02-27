"""
Main entry point for the Proxy Application.
Integrates FastAPI with caching middleware and router components.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .cache_middleware import ModelsCacheMiddleware
from .enhanced_proxy import EnhancedProxy
from .status_api import StatusAPI

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("proxy_app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Proxy Application...")
    
    # Initialize cache reference for later access
    app.state.cache_middleware = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down Proxy Application...")


def create_application() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="LLM Proxy Router",
        description="Intelligent routing and caching layer for LLM providers",
        version="4.1.0",
        lifespan=lifespan
    )
    
    # Add cache middleware early in the stack (before auth, after exception handlers)
    # This ensures /v1/models responses are cached efficiently
    app.add_middleware(
        ModelsCacheMiddleware,
        ttl_seconds=int(os.getenv("MODELS_CACHE_TTL", "300")),
        max_entries=int(os.getenv("MODELS_CACHE_MAX_ENTRIES", "100")),
        enabled=os.getenv("MODELS_CACHE_ENABLE", "true").lower() == "true"
    )
    
    # Initialize core components
    enhanced_proxy = EnhancedProxy()
    status_api = StatusAPI()
    
    # Include routers
    app.include_router(enhanced_proxy.router, prefix="/v1")
    app.include_router(status_api.router, prefix="")
    
    # Health check endpoint (bypasses cache)
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "version": "4.1.0"}
    
    # Cache stats endpoint for monitoring
    @app.get("/admin/cache/stats")
    async def cache_stats(request: Request):
        """Get cache statistics for monitoring."""
        # Find the middleware instance
        for middleware in request.app.user_middleware:
            if isinstance(middleware.cls, type) and middleware.cls.__name__ == "ModelsCacheMiddleware":
                # Access the middleware instance from app state if stored
                if hasattr(request.app.state, "cache_middleware") and request.app.state.cache_middleware:
                    stats = await request.app.state.cache_middleware.get_stats()
                    return JSONResponse(content=stats)
        
        # Alternative: return basic info if middleware not accessible via state
        return JSONResponse(
            content={"note": "Cache middleware active but stats endpoint needs middleware reference"}
        )
    
    @app.post("/admin/cache/clear")
    async def clear_cache(request: Request):
        """Clear the models cache."""
        for middleware in request.app.user_middleware:
            if isinstance(middleware.cls, type) and middleware.cls.__name__ == "ModelsCacheMiddleware":
                if hasattr(request.app.state, "cache_middleware") and request.app.state.cache_middleware:
                    await request.app.state.cache_middleware.clear()
                    return JSONResponse(content={"status": "cache cleared"})
        
        return JSONResponse(
            content={"error": "Cache middleware not found"}, 
            status_code=404
        )
    
    return app


# Create application instance
app = create_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.proxy_app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true"
    )
