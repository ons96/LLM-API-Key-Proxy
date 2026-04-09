import asyncio
import os
import sys
import httpx
from unittest.mock import MagicMock, AsyncMock

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

# Mock command line arguments to avoid TUI launch
sys.argv = ["main.py", "--host", "127.0.0.1", "--port", "8000"]

from proxy_app.main import app, get_rotating_client
from proxy_app.health_checker import HealthChecker
from proxy_app.router_wrapper import initialize_router, get_router, RouterWrapper


async def test_health_checker_integration():
    print("Test: Health Checker Integration via RouterWrapper...")

    # Manually initialize app state components that normally run in lifespan
    # Mock rotating client
    client = MagicMock()
    # Mock router integration
    mock_integration = MagicMock()
    mock_adapter = AsyncMock()
    mock_adapter.list_models.return_value = ["test-model"]
    mock_adapter.chat_completions.return_value = {"id": "1"}
    mock_integration.adapters = {"mock_provider": mock_adapter}

    # Setup router wrapper mock
    mock_wrapper = MagicMock()
    mock_wrapper.router_integration = mock_integration

    # Patch get_router to return our mock
    # Since get_router returns a global instance, we can set it via initialize_router
    # But initialize_router creates a real RouterWrapper.
    # Let's bypass and just test HealthChecker directly with the mock integration
    # which simulates what happens in main.py

    checker = HealthChecker(mock_integration, interval_seconds=1)

    # Run check
    await checker._check_all_providers()

    stats = checker.get_stats()
    print(f"Stats: {stats}")

    if "mock_provider" in stats and stats["mock_provider"]["status"] == "healthy":
        print("✅ Health checker marked provider as healthy.")
        return True
    else:
        print("❌ Health checker failed.")
        return False


async def main():
    print("=== Testing Health Check Logic ===\n")
    if await test_health_checker_integration():
        print("\n✅ Health Check verification passed.")
    else:
        print("\n❌ Health Check verification failed.")


if __name__ == "__main__":
    asyncio.run(main())
