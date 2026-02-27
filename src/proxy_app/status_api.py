import time
from typing import Dict, Any
from fastapi import APIRouter, Request, HTTPException

router = APIRouter()


@router.get("/health")
async def health_check(request: Request) -> Dict[str, Any]:
    """
    Health check endpoint returning provider status information.
    Part of Phase 3.1 Uptime Monitoring.
    """
    health_checker = getattr(request.app.state, "health_checker", None)
    
    if not health_checker:
        raise HTTPException(
            status_code=503, 
            detail="Health checker not initialized"
        )
    
    stats = health_checker.get_stats()
    
    # Calculate overall system health
    if not stats:
        return {
            "status": "initializing",
            "timestamp": time.time(),
            "providers": {}
        }
    
    healthy_count = sum(
        1 for s in stats.values() 
        if s.get("status") == "healthy"
    )
    total_count = len(stats)
    
    if healthy_count == total_count:
        overall_status = "healthy"
    elif healthy_count > 0:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    return {
        "status": overall_status,
        "healthy_providers": healthy_count,
        "total_providers": total_count,
        "timestamp": time.time(),
        "providers": stats
    }
