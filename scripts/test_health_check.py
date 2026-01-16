import asyncio
import os
import sys
import httpx
from unittest.mock import MagicMock, AsyncMock

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from proxy_app.main import app
from proxy_app.health_checker import HealthChecker
from proxy_app.router_wrapper import initialize_router, get_router


async def test_health_checker():
    print("Test 1: Health Checker Logic...")

    # Mock integration and adapters
    mock_integration = MagicMock()
    mock_adapter = AsyncMock()
    mock_adapter.list_models.return_value = ["test-model"]
    mock_adapter.chat_completions.return_value = {"id": "1"}

    mock_integration.adapters = {"mock_provider": mock_adapter}

    checker = HealthChecker(mock_integration, interval_seconds=1)

    # Run one check cycle manually
    await checker._check_all_providers()

    stats = checker.get_stats()
    print(f"Stats: {stats}")

    if "mock_provider" in stats and stats["mock_provider"]["status"] == "healthy":
        print("✅ Health checker marked provider as healthy.")
    else:
        print("❌ Health checker failed.")
        return False

    return True


async def main():
    print("=== Testing Health & Stats ===\n")
    if await test_health_checker():
        print("\n✅ Health Check verification passed.")
    else:
        print("\n❌ Health Check verification failed.")


if __name__ == "__main__":
    asyncio.run(main())
