#!/usr/bin/env python3
"""
Pricing API Module

FastAPI endpoints for accessing LLM provider pricing information.
Part of Phase 2.1 Data Collection Pipeline.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, Any, List, Optional
import csv
import io
import logging
from datetime import datetime

from rotator_library.pricing_data import (
    PricingDatabase,
    PricingCollector,
    ModelPricing,
    get_pricing_database,
    get_pricing_collector
)

# Configure logging
logger = logging.getLogger(__name__)

# Create router for pricing API endpoints
router = APIRouter(prefix="/api/pricing", tags=["pricing"])


def get_pricing_db(request: Request) -> PricingDatabase:
    """Dependency to get the pricing database from app state."""
    if hasattr(request.app.state, "pricing_database"):
        return request.app.state.pricing_database
    # Fallback to global instance
    return get_pricing_database()


@router.get("/")
async def get_all_pricing(
    request: Request, db: PricingDatabase = Depends(get_pricing_db)
) -> Dict[str, Any]:
    """
    Get all pricing data for all providers.

    Returns:
        JSON with all pricing information
    """
    try:
        all_pricing = db.get_all_as_list()
        
        # Group by provider
        providers = {}
        for pricing in all_pricing:
            provider = pricing.provider
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(pricing.to_dict())
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "source": db.get_source(),
            "total_models": db.get_model_count(),
            "providers": providers,
            "last_updated": db.get_last_updated().isoformat() if db.get_last_updated() else None
        }
    except Exception as e:
        logger.error(f"Failed to get all pricing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get pricing: {e}")


@router.get("/providers")
async def get_pricing_providers(
    request: Request, db: PricingDatabase = Depends(get_pricing_db)
) -> Dict[str, Any]:
    """
    Get list of all providers with pricing data.

    Returns:
        JSON with provider list and model counts
    """
    try:
        providers = db.get_providers()
        provider_info = []
        
        for provider in providers:
            pricing_list = db.get_provider_pricing(provider)
            provider_info.append({
                "name": provider,
                "model_count": len(pricing_list),
                "cheapest_input": min(p.input_price_per_million for p in pricing_list) if pricing_list else 0,
                "cheapest_output": min(p.output_price_per_million for p in pricing_list) if pricing_list else 0
            })
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "providers": provider_info,
            "total_providers": len(providers)
        }
    except Exception as e:
        logger.error(f"Failed to get providers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get providers: {e}")


@router.get("/provider/{provider_name}")
async def get_provider_pricing(
    request: Request,
    provider_name: str,
    db: PricingDatabase = Depends(get_pricing_db)
) -> Dict[str, Any]:
    """
    Get pricing for a specific provider.

    Args:
        provider_name: Name of the provider

    Returns:
        JSON with provider pricing details
    """
    try:
        provider_name = provider_name.lower()
        pricing_list = db.get_provider_pricing(provider_name)
        
        if not pricing_list:
            raise HTTPException(
                status_code=404, 
                detail=f"No pricing data found for provider '{provider_name}'"
            )
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "provider": provider_name,
            "model_count": len(pricing_list),
            "models": [p.to_dict() for p in pricing_list]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pricing for {provider_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get provider pricing: {e}")


@router.get("/model/{model_name}")
async def get_model_pricing(
    request: Request,
    model_name: str,
    db: PricingDatabase = Depends(get_pricing_db)
) -> Dict[str, Any]:
    """
    Search for pricing by model name.

    Args:
        model_name: Model name or search pattern

    Returns:
        JSON with matching model pricing
    """
    try:
        # Try exact match first
        all_pricing = db.get_all_as_list()
        exact_match = None
        for pricing in all_pricing:
            if pricing.model_name.lower() == model_name.lower():
                exact_match = pricing
                break
        
        if exact_match:
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "match_type": "exact",
                "model": exact_match.to_dict()
            }
        
        # Fall back to search
        matches = db.search_by_model(model_name)
        
        if not matches:
            raise HTTPException(
                status_code=404,
                detail=f"No models found matching '{model_name}'"
            )
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "match_type": "search",
            "query": model_name,
            "count": len(matches),
            "models": [p.to_dict() for p in matches]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search for model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search model: {e}")


@router.get("/cheapest")
async def get_cheapest_models(
    request: Request,
    limit: int = 10,
    input_only: bool = False,
    db: PricingDatabase = Depends(get_pricing_db)
) -> Dict[str, Any]:
    """
    Get the cheapest models by price.

    Args:
        limit: Maximum number of models to return (default: 10)
        input_only: Sort by input price instead of output price

    Returns:
        JSON with cheapest models
    """
    try:
        if limit <= 0:
            limit = 10
        if limit > 100:
            limit = 100
            
        cheapest = db.get_cheapest_models(limit=limit, input_only=input_only)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "sort_by": "input" if input_only else "output",
            "count": len(cheapest),
            "models": [p.to_dict() for p in cheapest]
        }
    except Exception as e:
        logger.error(f"Failed to get cheapest models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cheapest models: {e}")


@router.get("/compare")
async def compare_models(
    request: Request,
    models: str = "",  # Comma-separated list of model names
    db: PricingDatabase = Depends(get_pricing_db)
) -> Dict[str, Any]:
    """
    Compare pricing for multiple models.

    Args:
        models: Comma-separated list of model names to compare

    Returns:
        JSON with comparison data
    """
    try:
        if not models:
            raise HTTPException(status_code=400, detail="No models specified")
        
        model_list = [m.strip() for m in models.split(",")]
        all_pricing = db.get_all_as_list()
        
        matches = []
        for pricing in all_pricing:
            if pricing.model_name.lower() in [m.lower() for m in model_list]:
                matches.append(pricing)
        
        if not matches:
            raise HTTPException(
                status_code=404,
                detail="No matching models found"
            )
        
        # Sort by output price
        matches.sort(key=lambda x: x.output_price_per_million)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "count": len(matches),
            "models": [p.to_dict() for p in matches]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compare models: {e}")


@router.get("/export/csv")
async def export_pricing_csv(
    request: Request, db: PricingDatabase = Depends(get_pricing_db)
) -> StreamingResponse:
    """
    Export all pricing data to CSV format.

    Returns:
        CSV file with pricing data
    """
    try:
        all_pricing = db.get_all_as_list()
        
        # Create CSV
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            "provider", "model", "input_price_per_million", 
            "output_price_per_million", "currency", "context_window", "updated_at"
        ])
        
        # Data
        for pricing in all_pricing:
            writer.writerow([
                pricing.provider,
                pricing.model_name,
                pricing.input_price_per_million,
                pricing.output_price_per_million,
                pricing.currency,
                pricing.context_window,
                pricing.updated_at.isoformat()
            ])
        
        output.seek(0)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=pricing_data.csv",
                "Content-Type": "text/csv",
            }
        )
    except Exception as e:
        logger.error(f"Failed to export pricing CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export pricing: {e}")


@router.post("/refresh")
async def refresh_pricing(
    request: Request,
    db: PricingDatabase = Depends(get_pricing_db)
) -> Dict[str, Any]:
    """
    Trigger a refresh of pricing data.

    Returns:
        JSON with refresh status
    """
    try:
        collector = get_pricing_collector()
        collector.refresh()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success",
            "message": "Pricing data refresh triggered",
            "total_models": db.get_model_count()
        }
    except Exception as e:
        logger.error(f"Failed to refresh pricing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh pricing: {e}")
