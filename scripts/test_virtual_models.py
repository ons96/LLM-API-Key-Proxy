import asyncio
import os
import sys
from unittest.mock import MagicMock, AsyncMock

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from proxy_app.router_core import RouterCore
from proxy_app.router_integration import RouterIntegration
from proxy_app.router_wrapper import RouterWrapper, initialize_router, get_router


async def test_virtual_model_loading():
    """Test if RouterCore loads the virtual models from the new config."""
    print("Test 1: Loading Virtual Models...")
    try:
        router = RouterCore()
        # Force reload to be sure (though __init__ calls _initialize_components)
        router._initialize_components()

        models = router.virtual_models
        print(f"✅ Loaded {len(models)} virtual models.")

        target = "coding-smart"
        if target in models:
            print(f"✅ Virtual model '{target}' found.")
            chain = models[target].get("fallback_chain", [])
            print(f"   Chain length: {len(chain)}")
            if len(chain) > 0 and chain[0]["provider"] == "gemini":
                print("✅ Chain priority 1 is Gemini (correct).")
            else:
                print(f"❌ Unexpected chain: {chain}")
                return False
        else:
            print(f"❌ Virtual model '{target}' NOT found.")
            return False

        return True
    except Exception as e:
        print(f"❌ Failed to load virtual models: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_routing_logic():
    """Test if RouterWrapper routes a request to a virtual model correctly."""
    print("\nTest 2: Routing Request...")
    try:
        # Mock RotatingClient since we don't want real network calls
        mock_client = MagicMock()
        mock_client.acompletion = AsyncMock(
            return_value={"id": "mock-response", "choices": []}
        )

        # Initialize router with mock client
        initialize_router(mock_client)
        router_wrapper = get_router()

        # Mock adapter calls to avoid actual API calls inside RouterIntegration
        # We need to patch the adapters in router_integration
        mock_adapter = AsyncMock()
        mock_adapter.chat_completions.return_value = {
            "id": "mock-response",
            "choices": [{"message": {"content": "Mocked"}}],
        }
        mock_adapter.is_model_available.return_value = True

        # Inject mock adapters
        router_wrapper.router_integration.adapters = {
            "gemini": mock_adapter,
            "groq": mock_adapter,
            "g4f": mock_adapter,
        }

        # Make a request for "coding-smart"
        request_data = {
            "model": "coding-smart",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        mock_request = MagicMock()

        print(f"   Requesting model: {request_data['model']}")
        response = await router_wrapper.handle_chat_completions(
            request_data, mock_request
        )

        print("✅ Request handled successfully.")
        return True

    except Exception as e:
        print(f"❌ Routing failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    print("=== Testing Virtual Models ===\n")

    # Ensure env vars are set for adapters to initialize (even if mocked later)
    os.environ["GROQ_API_KEY"] = "mock"
    os.environ["GEMINI_API_KEY"] = "mock"

    r1 = await test_virtual_model_loading()
    r2 = await test_routing_logic()

    if r1 and r2:
        print("\n✅ All Virtual Model tests passed.")
    else:
        print("\n❌ Tests failed.")


if __name__ == "__main__":
    asyncio.run(main())
