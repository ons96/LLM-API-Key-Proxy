"""
Deployment hooks and verification utilities for automated CI/CD pipelines.
Integrates with GitHub Actions, GitLab CI, Kubernetes, and other automation platforms.
"""

import os
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/deployment", tags=["deployment"])


class DeploymentStatusResponse(BaseModel):
    status: str
    version: str
    environment: str
    timestamp: float
    checks_passed: bool
    uptime_seconds: float = Field(default_factory=lambda: time.time())


class VerificationCheck(BaseModel):
    path: str
    method: str
    expected_status: int


class DeploymentConfig:
    """Manages deployment configuration from YAML."""
    
    _instance: Optional["DeploymentConfig"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._config = self._load_config()
        self._env = os.getenv("DEPLOYMENT_ENV", "production")
        self._start_time = time.time()
        self._initialized = True
    
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration from config file."""
        config_path = (
            Path(__file__).resolve().parent.parent.parent 
            / "config" 
            / "deployment_config.yaml"
        )
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {"deployment": {"environments": {}}}
    
    @property
    def environment(self) -> str:
        return self._env
    
    def get_env_config(self) -> Dict[str, Any]:
        """Get configuration for current environment."""
        return (
            self._config.get("deployment", {})
            .get("environments", {})
            .get(self._env, {})
        )
    
    def verify_required_env_vars(self) -> List[str]:
        """Check for missing required environment variables."""
        env_config = self.get_env_config()
        required = env_config.get("required_env_vars", [])
        return [var for var in required if not os.getenv(var)]
    
    def get_version(self) -> str:
        """Get application version from environment or config."""
        meta = self._config.get("deployment", {}).get("metadata", {})
        version_source = meta.get("version_source", "env")
        
        if version_source == "env":
            env_var = meta.get("version_env_var", "APP_VERSION")
            return os.getenv(env_var, "unknown")
        return "unknown"


# Singleton instance
config = DeploymentConfig()


@router.get("/status", response_model=DeploymentStatusResponse)
async def deployment_status():
    """
    Get deployment status for CI/CD verification.
    Returns 503 if required environment variables are missing.
    """
    missing_vars = config.verify_required_env_vars()
    checks_passed = len(missing_vars) == 0
    
    status = "healthy" if checks_passed else "degraded"
    uptime = time.time() - config._start_time
    
    if not checks_passed:
        raise HTTPException(
            status_code=503,
            detail={
                "status": status,
                "missing_env_vars": missing_vars,
                "timestamp": time.time(),
                "uptime_seconds": uptime
            }
        )
    
    return DeploymentStatusResponse(
        status=status,
        version=config.get_version(),
        environment=config.environment,
        timestamp=time.time(),
        checks_passed=checks_passed,
        uptime_seconds=uptime
    )


@router.get("/config")
async def get_deployment_config():
    """
    Get deployment configuration (safe subset for CI/CD).
    Excludes sensitive values, returns health check paths and verification endpoints.
    """
    env_config = config.get_env_config()
    
    return {
        "environment": config.environment,
        "version": config.get_version(),
        "health_checks": env_config.get("health_checks", {}),
        "verification": env_config.get("verification", {}),
        "rollout": config._config.get("deployment", {}).get("rollout", {})
    }


@router.post("/verify")
async def verify_deployment():
    """
    Trigger post-deployment verification checks.
    Returns the list of endpoints that should be verified by the CI/CD pipeline.
    """
    env_config = config.get_env_config()
    verification = env_config.get("verification", {})
    endpoints = verification.get("endpoints", [])
    
    # Check required env vars first
    missing = config.verify_required_env_vars()
    if missing:
        raise HTTPException(
            status_code=503,
            detail={
                "verified": False,
                "reason": f"Missing required env vars: {', '.join(missing)}"
            }
        )
    
    return {
        "verified": True,
        "environment": config.environment,
        "version": config.get_version(),
        "checks_required": endpoints,
        "timestamp": time.time()
    }


@router.get("/health/ready")
async def readiness_probe():
    """
    Kubernetes-style readiness probe.
    Returns 200 when application is ready to receive traffic.
    """
    missing = config.verify_required_env_vars()
    if missing:
        raise HTTPException(
            status_code=503,
            detail={"ready": False, "missing_vars": missing}
        )
    return {"ready": True, "timestamp": time.time()}


@router.get("/health/live")
async def liveness_probe():
    """
    Kubernetes-style liveness probe.
    Returns 200 if application is running (not deadlocked).
    """
    return {
        "alive": True,
        "timestamp": time.time(),
        "uptime_seconds": time.time() - config._start_time
    }


def init_deployment_hooks(app):
    """
    Initialize deployment hooks with the FastAPI application.
    Call this in main.py after creating the FastAPI app.
    """
    app.include_router(router)
    
    # Log configuration status on startup
    missing = config.verify_required_env_vars()
    if missing:
        print(f"⚠️  [Deployment] Missing required env vars: {', '.join(missing)}")
    else:
        print(f"✓ [Deployment] Configuration loaded for environment: {config.environment}")
        print(f"  → Version: {config.get_version()}")
        print(f"  → Health checks: /deployment/health/ready, /deployment/health/live")
