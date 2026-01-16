import asyncio
import os
import sys
import logging
import time
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VERIFY")

# Ensure src in path
sys.path.append(os.path.join(os.getcwd(), "src"))

# Mock args to prevent TUI launch
sys.argv = ["main.py", "--host", "127.0.0.1", "--port", "8000"]

from proxy_app.router_core import RouterCore
from proxy_app.model_ranker import ModelRanker
from proxy_app.health_checker import HealthChecker
from proxy_app.rate_limiter import RateLimitTracker


async def check_dependencies():
    """Check installed dependencies."""
    logger.info("--- Phase 1: Dependency Check ---")
    required = ["fastapi", "uvicorn", "litellm", "httpx", "yaml"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            logger.info(f"✅ {pkg} installed")
        except ImportError:
            logger.error(f"❌ {pkg} missing")
            missing.append(pkg)
    return len(missing) == 0


async def check_configuration():
    """Validate configuration files."""
    logger.info("\n--- Phase 2: Configuration Validation ---")
    configs = [
        "config/virtual_models.yaml",
        "config/providers_database.yaml",
        "config/model_rankings.yaml",
    ]

    valid = True
    for cfg in configs:
        if os.path.exists(cfg):
            logger.info(f"✅ {cfg} found")
        else:
            logger.error(f"❌ {cfg} MISSING")
            valid = False

    # Test RouterCore loading
    try:
        router = RouterCore()
        models = router.virtual_models
        logger.info(f"✅ RouterCore loaded {len(models)} virtual models")
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"❌ RouterCore failed to load config: {e}")
        valid = False

    return valid


async def check_components():
    """Verify internal components."""
    logger.info("\n--- Phase 3: Component Logic Verification ---")

    # 1. Rate Limiter
    tracker = RateLimitTracker()
    await tracker.record_request("test", "model")
    stats = await tracker.get_usage_stats()
    if stats.get("test/model", {}).get("rpm") == 1:
        logger.info("✅ RateLimitTracker working")
    else:
        logger.error("❌ RateLimitTracker logic failed")
        return False

    # 2. Model Ranker
    ranker = ModelRanker()
    candidates = [{"provider": "mock", "model": "gemini-1.5-pro", "priority": 2}]
    ranked = ranker.rank_candidates("coding-smart", candidates)
    if ranked:
        logger.info("✅ ModelRanker working")
    else:
        logger.error("❌ ModelRanker failed")
        return False

    return True


async def main():
    logger.info("=== System Verification Started ===\n")

    if not await check_dependencies():
        logger.error("Verification halted due to missing dependencies.")
        return

    if not await check_configuration():
        logger.error("Verification halted due to configuration errors.")
        return

    if not await check_components():
        logger.error("Verification halted due to component failures.")
        return

    logger.info("\n=== ✅ System Verification Completed Successfully ===")
    logger.info(
        "The proxy is ready to start with Virtual Models, Health Checks, and Auto-Ranking enabled."
    )


if __name__ == "__main__":
    asyncio.run(main())
