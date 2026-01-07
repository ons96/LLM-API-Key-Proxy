#!/usr/bin/env python3
"""
Provider Status API Module

FastAPI endpoints for accessing provider health status information.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, Any, List
import csv
import io
import logging

from rotator_library.provider_status_tracker import ProviderStatusTracker

# Configure logging
logger = logging.getLogger(__name__)

# Create router for status API endpoints
router = APIRouter(prefix="/api/providers", tags=["provider-status"])


def get_status_tracker(request: Request) -> ProviderStatusTracker:
    """Dependency to get the provider status tracker from app state."""
    if not hasattr(request.app.state, "provider_status_tracker"):
        raise HTTPException(status_code=500, detail="Provider status tracker not initialized")
    return request.app.state.provider_status_tracker


@router.get("/status")
async def get_provider_status(
    request: Request,
    tracker: ProviderStatusTracker = Depends(get_status_tracker)
) -> Dict[str, Any]:
    """
    Get current status snapshot for all providers.
    
    Returns:
        JSON with timestamp and provider status details
    """
    try:
        status = tracker.get_current_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get provider status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get provider status: {e}")


@router.get("/status/{provider_name}")
async def get_single_provider_status(
    request: Request,
    provider_name: str,
    tracker: ProviderStatusTracker = Depends(get_status_tracker)
) -> Dict[str, Any]:
    """
    Get detailed status for a single provider.
    
    Args:
        provider_name: Name of the provider to query
        
    Returns:
        JSON with provider status details
    """
    try:
        current_status = tracker.get_current_status()
        
        if provider_name not in current_status["providers"]:
            raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")
        
        provider_data = current_status["providers"][provider_name]
        
        # Add provider name to response
        provider_data["provider_name"] = provider_name
        
        return provider_data
        
    except Exception as e:
        logger.error(f"Failed to get status for {provider_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get provider status: {e}")


@router.get("/best")
async def get_best_provider(
    request: Request,
    tracker: ProviderStatusTracker = Depends(get_status_tracker)
) -> Dict[str, Any]:
    """
    Get the healthiest/fastest provider.
    
    Returns:
        JSON with best provider recommendation and alternatives
    """
    try:
        best_provider = tracker.get_best_provider()
        return best_provider
    except Exception as e:
        logger.error(f"Failed to get best provider: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get best provider: {e}")


@router.get("/history")
async def get_provider_history(
    request: Request,
    hours: int = 24,
    tracker: ProviderStatusTracker = Depends(get_status_tracker)
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get historical data for all providers.
    
    Args:
        hours: Time window in hours (default: 24)
        
    Returns:
        JSON with time-series data for each provider
    """
    try:
        if hours <= 0:
            hours = 24
        
        history = tracker.get_all_history(hours)
        return history
    except Exception as e:
        logger.error(f"Failed to get provider history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get provider history: {e}")


@router.get("/history/{provider_name}")
async def get_single_provider_history(
    request: Request,
    provider_name: str,
    hours: int = 24,
    tracker: ProviderStatusTracker = Depends(get_status_tracker)
) -> List[Dict[str, Any]]:
    """
    Get historical data for a single provider.
    
    Args:
        provider_name: Name of the provider to query
        hours: Time window in hours (default: 24)
        
    Returns:
        JSON with time-series data for the provider
    """
    try:
        if hours <= 0:
            hours = 24
        
        history = tracker.get_provider_history(provider_name, hours)
        
        if not history:
            raise HTTPException(status_code=404, detail=f"No history found for provider '{provider_name}'")
        
        return history
        
    except Exception as e:
        logger.error(f"Failed to get history for {provider_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get provider history: {e}")


@router.get("/export/csv")
async def export_provider_status_csv(
    request: Request,
    tracker: ProviderStatusTracker = Depends(get_status_tracker)
) -> StreamingResponse:
    """
    Export current provider status to CSV format.
    
    Returns:
        CSV file with provider status data
    """
    try:
        csv_data = tracker.export_to_csv()
        
        # Create streaming response
        stream = io.StringIO(csv_data)
        
        return StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=provider_status.csv",
                "Content-Type": "text/csv"
            }
        )
    except Exception as e:
        logger.error(f"Failed to export provider status to CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export provider status: {e}")


@router.get("/health")
async def health_check_endpoint(
    request: Request,
    tracker: ProviderStatusTracker = Depends(get_status_tracker)
) -> Dict[str, Any]:
    """
    Health check endpoint for the status tracker itself.
    
    Returns:
        JSON with tracker status and basic info
    """
    try:
        return {
            "status": "healthy",
            "providers_monitored": len(tracker.providers_to_monitor),
            "check_interval_minutes": tracker.check_interval_minutes,
            "running": tracker.running
        }
    except Exception as e:
        logger.error(f"Status tracker health check failed: {e}")
        return {
            "status": "degraded",
            "error": str(e)
        }


# Integration function for router logic
def get_healthiest_provider(tracker: ProviderStatusTracker) -> str:
    """
    Get the healthiest provider for routing decisions.
    
    Args:
        tracker: ProviderStatusTracker instance
        
    Returns:
        Name of the healthiest provider, or None if no healthy providers available
    """
    try:
        best_provider_info = tracker.get_best_provider()
        return best_provider_info.get("best_provider")
    except Exception as e:
        logger.error(f"Failed to get healthiest provider: {e}")
        return None


# Function to initialize the status tracker
def initialize_status_tracker(app) -> ProviderStatusTracker:
    """
    Initialize the provider status tracker and add it to the app state.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        Initialized ProviderStatusTracker instance
    """
    try:
        # Create tracker instance
        tracker = ProviderStatusTracker(
            check_interval_minutes=5,  # Default: 5 minutes
            max_consecutive_failures=3,  # Default: 3 failures before marking as down
            degraded_latency_threshold_ms=1000  # Default: 1000ms for degraded status
        )
        
        # Add to app state
        app.state.provider_status_tracker = tracker
        
        # Start the tracker
        import asyncio
        asyncio.create_task(tracker.start())
        
        logger.info("Provider status tracker initialized and started")
        
        return tracker
        
    except Exception as e:
        logger.error(f"Failed to initialize provider status tracker: {e}")
        raise