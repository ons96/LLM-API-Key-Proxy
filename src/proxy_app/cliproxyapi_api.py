"""
CLIProxyAPI Health and Status Endpoints.

Provides endpoints to monitor the CLIProxyAPI sidecar service,
including health checks, provider status, and model availability.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cliproxyapi", tags=["cliproxyapi"])


def get_cliproxyapi_provider(request: Request):
    """Get CLIProxyAPI provider instance from app state."""
    if not hasattr(request.app.state, "providers"):
        raise HTTPException(status_code=500, detail="Providers not initialized")

    provider = request.app.state.providers.get("cliproxyapi")
    if not provider:
        raise HTTPException(
            status_code=404, detail="CLIProxyAPI provider not configured"
        )

    return provider


@router.get("/health")
async def cliproxyapi_health(
    request: Request, provider=Depends(get_cliproxyapi_provider)
) -> Dict[str, Any]:
    """
    Check CLIProxyAPI sidecar health.

    Returns:
        JSON with health status and connection info
    """
    try:
        health_status = await provider.health_check()
        return health_status
    except Exception as e:
        logger.error(f"CLIProxyAPI health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "provider": "cliproxyapi",
        }


@router.get("/status")
async def cliproxyapi_status(
    request: Request, provider=Depends(get_cliproxyapi_provider)
) -> Dict[str, Any]:
    """
    Get detailed CLIProxyAPI status including all backends.

    Returns:
        JSON with:
        - Overall status
        - Per-backend health (gemini, iflow, antigravity, qwen)
        - Model counts
        - Last check timestamp
    """
    try:
        provider_status = await provider.get_provider_status()
        health_status = await provider.health_check()

        return {
            "status": provider_status.get("status", "unknown"),
            "provider": "cliproxyapi",
            "base_url": provider.config.base_url,
            "timeout": provider.config.timeout,
            "backends": provider_status.get("providers", {}),
            "total_models": provider_status.get("total_models", 0),
            "health": health_status,
        }
    except Exception as e:
        logger.error(f"CLIProxyAPI status check failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get CLIProxyAPI status: {e}"
        )


@router.get("/models")
async def cliproxyapi_models(
    request: Request, provider=Depends(get_cliproxyapi_provider)
) -> Dict[str, Any]:
    """
    Get available models from CLIProxyAPI.

    Returns:
        JSON with list of available models grouped by backend
    """
    try:
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{provider.config.base_url}/v1/models")
            response.raise_for_status()

            data = response.json()
            models = data.get("data", [])

            # Group models by backend
            backends: Dict[str, list] = {}
            for model in models:
                model_id = model.get("id", "")
                if "/" in model_id:
                    backend, model_name = model_id.split("/", 1)
                    if backend not in backends:
                        backends[backend] = []
                    backends[backend].append(
                        {
                            "id": model_id,
                            "name": model_name,
                        }
                    )

            return {
                "total": len(models),
                "backends": backends,
                "models": models,
            }

    except httpx.RequestError as e:
        logger.error(f"Failed to fetch models from CLIProxyAPI: {e}")
        raise HTTPException(status_code=503, detail=f"CLIProxyAPI unavailable: {e}")
    except Exception as e:
        logger.error(f"Failed to get CLIProxyAPI models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {e}")


@router.get("/backend/{backend_name}")
async def cliproxyapi_backend_status(
    request: Request, backend_name: str, provider=Depends(get_cliproxyapi_provider)
) -> Dict[str, Any]:
    """
    Get status for a specific CLIProxyAPI backend.

    Args:
        backend_name: Backend name (gemini, iflow, antigravity, qwen)

    Returns:
        JSON with backend status and available models
    """
    backend_name = backend_name.lower()

    supported_backends = ["gemini", "iflow", "antigravity", "qwen"]
    if backend_name not in supported_backends:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown backend: {backend_name}. Supported: {supported_backends}",
        )

    try:
        provider_status = await provider.get_provider_status()
        all_backends = provider_status.get("providers", {})

        if backend_name not in all_backends:
            return {
                "backend": backend_name,
                "status": "not_configured",
                "models": [],
                "auth_method": provider.SUPPORTED_BACKENDS.get(backend_name, {})
                .get("auth_method", "unknown")
                .value
                if backend_name in provider.SUPPORTED_BACKENDS
                else "unknown",
            }

        models = all_backends[backend_name]
        backend_config = provider.SUPPORTED_BACKENDS.get(backend_name, {})

        return {
            "backend": backend_name,
            "status": "configured",
            "models": models,
            "model_count": len(models),
            "auth_method": backend_config.get("auth_method", "unknown").value,
            "auto_refresh": backend_config.get("auto_refresh", False),
        }

    except Exception as e:
        logger.error(f"Failed to get {backend_name} backend status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get backend status: {e}"
        )


@router.get("/config")
async def cliproxyapi_config(
    request: Request, provider=Depends(get_cliproxyapi_provider)
) -> Dict[str, Any]:
    """
    Get CLIProxyAPI provider configuration.

    Returns:
        JSON with configuration details
    """
    return {
        "provider": "cliproxyapi",
        "base_url": provider.config.base_url,
        "timeout": provider.config.timeout,
        "enabled": provider.config.enabled,
        "health_check_interval": provider.config.health_check_interval,
        "supported_backends": list(provider.SUPPORTED_BACKENDS.keys()),
    }
