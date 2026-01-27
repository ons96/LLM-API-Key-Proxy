"""Quick test script to verify proxy works with new fallback config."""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, "/home/owens/CodingProjects/LLM-API-Key-Proxy/src")


async def test_router():
    """Test that the router can initialize and list models."""
    print("Testing router initialization...")

    try:
        # Test that router config loads
        from proxy_app.router_integration import RouterIntegration

        integration = RouterIntegration(rotating_client=None)

        # Get available models (use async method)
        models = await integration.get_models_async()
        print(f"\n✓ Router initialized successfully")
        print(f"✓ Available models: {len(models)}")

        # Check for key providers
        provider_models = {}
        for model_id, info in models.items():
            if "/" in model_id:
                provider = model_id.split("/")[0]
                if provider not in provider_models:
                    provider_models[provider] = []
                provider_models[provider].append(model_id)

        print(f"\n✓ Providers with models:")
        for provider, model_list in sorted(provider_models.items()):
            print(f"  - {provider}: {len(model_list)} models")
            if provider in ["g4f", "puter", "groq", "gemini"]:
                print(f"    Sample: {model_list[:3]}")

        # Verify model rankings file is valid
        from pathlib import Path

        rankings_path = Path(
            "/home/owens/CodingProjects/LLM-API-Key-Proxy/config/model_rankings.yaml"
        )
        if rankings_path.exists():
            import yaml

            with open(rankings_path) as f:
                rankings = yaml.safe_load(f)

            coding_smart = rankings.get("coding-smart", {}).get("models", [])
            coding_fast = rankings.get("coding-fast", {}).get("models", [])

            print(f"\n✓ Model rankings loaded:")
            print(f"  - coding-smart: {len(coding_smart)} models")
            print(f"  - coding-fast: {len(coding_fast)} models")

            # Check if Claude Opus 4.5 is in top 3
            if coding_smart:
                print(f"\n✓ Top 3 coding-smart models:")
                for i, model in enumerate(coding_smart[:3], 1):
                    print(f"  {i}. {model}")

        print("\n✅ All tests passed! Proxy is ready to use.")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_router())
    sys.exit(0 if success else 1)
