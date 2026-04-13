import asyncio
import os
import sys
import httpx
from typing import Dict, Any

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

# Mocking the client for testing without actual API keys if needed,
# but this script is intended to run against the *actual* proxy logic
# to verify wiring. For now, we assume the proxy is running or we import parts.
# However, importing parts is better for a self-contained test script.

from rotator_library import RotatingClient
from proxy_app.router_core import RouterCore
from proxy_app.router_integration import RouterIntegration


async def test_legacy_client_instantiation():
    """Test 1: Can we instantiate the legacy RotatingClient?"""
    print("Test 1: Instantiating RotatingClient...")
    try:
        # Mock keys
        api_keys = {"groq": ["gsk_test_key"], "gemini": ["dummy_key"]}
        client = RotatingClient(api_keys=api_keys)
        print("✅ RotatingClient instantiated successfully.")
        return True
    except Exception as e:
        print(f"❌ Failed to instantiate RotatingClient: {e}")
        return False


async def test_router_core_loading():
    """Test 2: Can RouterCore load the config/router_config.yaml?"""
    print("\nTest 2: Loading RouterCore config...")
    config_path = "config/router_config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found at {config_path}")
        return False

    try:
        router = RouterCore(config_path=config_path)
        models = router.get_model_list()
        print(f"✅ RouterCore loaded. Found {len(models)} models.")

        # Verify specific virtual model exists
        target = "router/best-coding"
        found = any(m["id"] == target for m in models)
        if found:
            print(f"✅ Virtual model '{target}' found in config.")
        else:
            print(f"❌ Virtual model '{target}' NOT found in config.")
            return False

        return True
    except Exception as e:
        print(f"❌ Failed to load RouterCore: {e}")
        return False


async def test_router_integration():
    """Test 3: Does RouterIntegration initialize adapters?"""
    print("\nTest 3: RouterIntegration Adapter Initialization...")
    try:
        # We need to mock environment variables for adapters to load
        os.environ["GROQ_API_KEY"] = "gsk_test_key"

        integration = RouterIntegration()
        adapters = integration.adapters
        print(
            f"✅ RouterIntegration initialized. Active adapters: {list(adapters.keys())}"
        )

        if "g4f" in adapters:
            print("✅ G4F adapter present (expected as it needs no key).")
        else:
            print("⚠️ G4F adapter missing.")

        return True
    except Exception as e:
        print(f"❌ RouterIntegration failed: {e}")
        return False


async def main():
    print("=== Starting Codebase Verification ===\n")

    results = []
    results.append(await test_legacy_client_instantiation())
    results.append(await test_router_core_loading())
    results.append(await test_router_integration())

    print("\n=== Verification Summary ===")
    if all(results):
        print("✅ All internal components verified successfully.")
        print("NOTE: This does not verify the running server, only the internal logic.")
    else:
        print("❌ Some components failed verification.")


if __name__ == "__main__":
    asyncio.run(main())
