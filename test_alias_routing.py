import asyncio
import logging
from unittest.mock import MagicMock, patch, AsyncMock
from src.proxy_app.router_core import RouterCore, ProviderCandidate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_alias_routing():
    print("\n=== Testing Alias Routing Logic (Real Config) ===")

    # We will ONLY mock the execution, but let it load the REAL config/aliases.yaml

    with patch(
        "src.proxy_app.router_core.litellm.acompletion", new_callable=AsyncMock
    ) as mock_llm:
        # Mock successful response
        mock_llm.return_value = {"choices": [{"message": {"content": "Success!"}}]}

        router = RouterCore()

        # Verify aliases loaded
        if "coding" in router.aliases:
            print(
                f"✅ Alias 'coding' loaded with {len(router.aliases['coding']['candidates'])} candidates"
            )
            candidates = router.aliases["coding"]["candidates"]
            print(
                f"   Top candidate: {candidates[0]['provider']}/{candidates[0]['model']}"
            )
        else:
            print("❌ Alias 'coding' NOT found in configuration")
            return

        print("1. Requesting model='coding'...")
        try:
            # We expect it to try the first candidate
            await router.route_request({"model": "coding"}, "req-real-1")

            # Check what was called
            args, _ = mock_llm.call_args
            called_model = (
                args[0].get("model") if args else None
            )  # litellm args are usually kwargs, but let's check
            if not called_model:
                # litellm.acompletion(model="...", ...)
                called_model = mock_llm.call_args.kwargs.get("model")

            print(f"2. Router called model: {called_model}")

            expected_first = f"{candidates[0]['provider']}/{candidates[0]['model']}"
            if called_model == expected_first:
                print(
                    "✅ SUCCESS: Router picked the top ranked candidate from aliases.yaml"
                )
            else:
                print(f"❌ FAILED: Expected {expected_first}, got {called_model}")

        except Exception as e:
            print(f"❌ FAILED: Router raised exception: {e}")


if __name__ == "__main__":
    asyncio.run(test_alias_routing())
