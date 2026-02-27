"""Main entry point for the proxy application."""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import router_wrapper, status_api, settings_tool
from .config_watcher import ConfigWatcher
from ..rotator_library import (
    get_scheduler, 
    SchedulerConfig,
    leaderboard_updater,
    provider_priority_manager,
    model_info_service
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global scheduler instance
scheduler = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan handler."""
        global scheduler
        
        # Initialize scheduler with configuration
        scheduler_config = SchedulerConfig(
            inactivity_threshold=300,  # 5 minutes
            min_recalc_interval=300,   # 5 minutes
            provider_refresh_interval=21600,  # 6 hours
            benchmark_update_interval=86400   # 24 hours
        )
        scheduler = get_scheduler(scheduler_config)
        
        # Set up callbacks
        async def ranking_recalc():
            """Callback for ranking recalculation."""
            logger.info("Running scheduled ranking recalculation")
            await router_wrapper.update_rankings()
            
        async def provider_refresh():
            """Callback for provider model refresh."""
            logger.info("Running scheduled provider model refresh")
            await provider_priority_manager.refresh_provider_models()
            await model_info_service.update_all_providers()
            
        async def benchmark_update():
            """Callback for benchmark data update."""
            logger.info("Running scheduled benchmark data update")
            await leaderboard_updater.update_leaderboard()
            
        scheduler.set_callbacks(
            recalc_callback=ranking_recalc,
            provider_refresh_callback=provider_refresh,
            benchmark_update_callback=benchmark_update
        )
        
        # Start the scheduler
        await scheduler.start()
        
        # Initial provider load
        await provider_priority_manager.initialize()
        
        # Initial benchmark data load
        try:
            await leaderboard_updater.update_leaderboard()
        except Exception as e:
            logger.warning(f"Initial leaderboard load failed: {e}")
        
        logger.info("Application started successfully")
        
        yield
        
        # Shutdown
        if scheduler:
            await scheduler.stop()
        logger.info("Application shutdown complete")
    
    app = FastAPI(
        title="Proxy Rotator API",
        description="Dynamic model routing with smart scheduling",
        version="2.1.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(router_wrapper.router, prefix="/v1")
    app.include_router(status_api.router, prefix="/status")
    app.include_router(settings_tool.router, prefix="/settings")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "scheduler": "active" if scheduler and scheduler._running else "inactive"}
    
    return app


app = create_app()


def main():
    """Run the application."""
    import uvicorn
    
    # Handle shutdown signals
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    uvicorn.run(
        "src.proxy_app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
