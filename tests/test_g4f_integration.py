#!/usr/bin/env python3
"""
Test script to verify G4F provider integration in the factory system.
This tests that the G4F provider can be properly instantiated and managed.
"""


import os
import sys
import asyncio
import json
from pathlib import Path

import httpx

# Add the src directory to Python path
# Current file is in tests/, so we need to go up one level to root, then into src
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rotator_library.provider_factory import (
    get_provider_class, 
    get_available_providers, 
    is_oauth_provider, 
    is_direct_provider,
    get_provider_config,
    validate_provider_config,
    OAUTH_PROVIDER_MAP,
    DIRECT_PROVIDER_MAP
)

from rotator_library.providers import PROVIDER_PLUGINS


def test_provider_factory_integration():
    """Test that G4F is properly integrated in the provider factory."""
    
    print("🔍 Testing G4F Provider Factory Integration...")
    print("=" * 50)
    
    # Test 1: Check that G4F is in available providers
    print("Test 1: Checking available providers...")
    available_providers = get_available_providers()
    print(f"Available providers: {available_providers}")
    
    assert "g4f" in available_providers, "G4F should be in available providers"
    print("✅ G4F found in available providers")
    
    # Test 2: Check that G4F is a direct provider, not OAuth
    print("\nTest 2: Checking G4F provider type...")
    assert not is_oauth_provider("g4f"), "G4F should not be an OAuth provider"
    assert is_direct_provider("g4f"), "G4F should be a direct provider"
    print("✅ G4F correctly classified as direct provider")
    
    # Test 3: Check that we can get the G4F provider class
    print("\nTest 3: Getting G4F provider class...")
    g4f_class = get_provider_class("g4f")
    print(f"G4F provider class: {g4f_class}")
    assert g4f_class is not None, "Should be able to get G4F provider class"
    print("✅ G4F provider class retrieved successfully")
    
    # Test 4: Check that G4F is in the PROVIDER_PLUGINS
    print("\nTest 4: Checking PROVIDER_PLUGINS registration...")
    assert "g4f" in PROVIDER_PLUGINS, "G4F should be in PROVIDER_PLUGINS"
    g4f_plugin = PROVIDER_PLUGINS["g4f"]
    print(f"G4F plugin class: {g4f_plugin}")
    print("✅ G4F registered in PROVIDER_PLUGINS")
    
    # Test 5: Check provider configuration loading
    print("\nTest 5: Testing G4F configuration loading...")
    config = get_provider_config("g4f")
    print(f"G4F config keys: {list(config.keys())}")
    print(f"G4F config: {config}")
    assert "default_tier_priority" in config, "G4F config should include tier priority"
    assert config["default_tier_priority"] == 5, "G4F should have fallback tier priority"
    print("✅ G4F configuration loading works")
    
    # Test 6: Check provider validation
    print("\nTest 6: Testing G4F provider validation...")
    is_valid = validate_provider_config("g4f")
    print(f"G4F validation result: {is_valid}")
    assert is_valid, "G4F should be valid"
    print("✅ G4F provider validation passes")
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! G4F integration is working correctly.")
    return True


async def test_g4f_provider_instantiation():
    """Test that we can actually instantiate a G4F provider."""
    
    print("\n🔍 Testing G4F Provider Instantiation...")
    print("=" * 50)
    
    # Set up some test environment variables
    os.environ["G4F_API_KEY"] = "test-api-key"
    os.environ["G4F_MAIN_API_BASE"] = "https://test-g4f-api.local"  # non-placeholder
    
    try:
        # Test instantiation
        from rotator_library.providers.g4f_provider import G4FProvider
        
        print("Creating G4FProvider instance...")
        provider = G4FProvider()
        
        print(f"Provider name: {provider.provider_name}")
        print(f"API key set: {'Yes' if provider.api_key else 'No'}")
        print(f"Main API base: {provider.main_api_base}")
        print(f"Default tier priority: {provider.default_tier_priority}")
        
        assert provider.provider_name == "g4f", "Provider name should be 'g4f'"
        assert provider.api_key == "test-api-key", "API key should be loaded from env"
        assert provider.main_api_base == "https://test-g4f-api.local", "Main API base should be loaded from env"
        assert provider.default_tier_priority == 5, "Default tier should be 5 (fallback)"
        
        print("✅ G4F provider instantiation successful")
        
        # Test has_custom_logic
        has_custom = provider.has_custom_logic()
        print(f"Has custom logic: {has_custom}")
        assert has_custom == True, "G4F should have custom logic"
        print("✅ G4F custom logic check passed")
        
    finally:
        # Clean up environment variables
        os.environ.pop("G4F_API_KEY", None)
        os.environ.pop("G4F_MAIN_API_BASE", None)
    
    print("\n" + "=" * 50)
    print("🎉 G4F provider instantiation test passed!")
    return True


async def test_g4f_completion_non_streaming_contract():
    """Validate G4FProvider builds correct OpenAI-style request."""

    print("\n🔍 Testing G4F Non-Streaming Completion Contract...")
    print("=" * 50)

    from rotator_library.providers.g4f_provider import G4FProvider

    captured = {"request": None}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request

        assert request.url.path == "/v1/chat/completions"
        assert request.headers.get("Authorization") == "Bearer test-key"

        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "glm-4.5"  # provider prefix stripped
        assert body["stream"] is False

        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "glm-4.5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    transport = httpx.MockTransport(handler)

    # Ensure we use public base (https://g4f.dev) and not a custom endpoint.
    old_main = os.environ.pop("G4F_MAIN_API_BASE", None)
    old_key = os.environ.pop("G4F_API_KEY", None)

    try:
        provider = G4FProvider()
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await provider.acompletion(
                client,
                model="g4f/glm-4.5",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                credential_identifier="test-key",
            )

        assert resp.model == "g4f/glm-4.5"
        assert resp.choices[0].message.content == "ok"

        print("✅ G4F non-streaming contract test passed")

    finally:
        if old_main is not None:
            os.environ["G4F_MAIN_API_BASE"] = old_main
        if old_key is not None:
            os.environ["G4F_API_KEY"] = old_key


async def test_g4f_completion_streaming_contract():
    """Validate G4FProvider streaming SSE parsing and chunk conversion."""

    print("\n🔍 Testing G4F Streaming Completion Contract...")
    print("=" * 50)

    from rotator_library.providers.g4f_provider import G4FProvider

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers.get("Authorization") == "Bearer test-key"
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "glm-4.5"
        assert body["stream"] is True

        chunk = {
            "id": "chatcmpl-stream-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "glm-4.5",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": None}
            ],
        }

        sse = f"data: {json.dumps(chunk)}\n\n" + "data: [DONE]\n\n"
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=sse.encode("utf-8"),
        )

    transport = httpx.MockTransport(handler)

    old_main = os.environ.pop("G4F_MAIN_API_BASE", None)
    old_key = os.environ.pop("G4F_API_KEY", None)

    try:
        provider = G4FProvider()
        async with httpx.AsyncClient(transport=transport) as client:
            gen = await provider.acompletion(
                client,
                model="g4f/glm-4.5",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                credential_identifier="test-key",
            )

            chunks = []
            async for chunk in gen:
                chunks.append(chunk)

        assert chunks
        assert chunks[0].model == "g4f/glm-4.5"
        chunk0 = chunks[0].model_dump()
        assert chunk0["choices"][0]["delta"]["content"] == "hi"

        print("✅ G4F streaming contract test passed")

    finally:
        if old_main is not None:
            os.environ["G4F_MAIN_API_BASE"] = old_main
        if old_key is not None:
            os.environ["G4F_API_KEY"] = old_key


def test_provider_maps():
    """Test the provider maps in the factory."""
    
    print("\n🔍 Testing Provider Maps...")
    print("=" * 50)
    
    # Check OAuth provider map
    print(f"OAuth providers: {list(OAUTH_PROVIDER_MAP.keys())}")
    assert "g4f" not in OAUTH_PROVIDER_MAP, "G4F should not be in OAuth provider map"
    
    # Check direct provider map  
    print(f"Direct providers: {list(DIRECT_PROVIDER_MAP.keys())}")
    assert "g4f" in DIRECT_PROVIDER_MAP, "G4F should be in direct provider map"
    
    print("✅ Provider maps are correctly configured")
    
    return True


def test_priority_manager():
    """Test the ProviderPriorityManager logic."""
    print("\n🔍 Testing Provider Priority Manager...")
    print("=" * 50)
    
    from rotator_library.provider_priority_manager import ProviderPriorityManager, ProviderTier
    
    # Initialize manager
    manager = ProviderPriorityManager()
    
    # Test 1: Check G4F default priority
    print("Test 1: Checking G4F priority...")
    g4f_tier = manager.get_provider_tier("g4f")
    print(f"G4F Tier: {g4f_tier.tier if g4f_tier else 'None'}")
    assert g4f_tier is not None
    assert g4f_tier.tier == ProviderTier.FALLBACK
    print("✅ G4F defaulted to FALLBACK tier")

    # Test 2: Check Fallback Chain
    print("\nTest 2: Checking Fallback Chain...")
    # Simulate available providers: OpenAI (Premium) and G4F (Fallback)
    available = ["openai", "g4f", "unused"]
    
    # Requesting OpenAI should give OpenAI -> G4F
    chain_openai = manager.get_fallback_chain("openai", available)
    print(f"Fallback chain for 'openai': {chain_openai}")
    assert chain_openai == ["openai", "g4f"]
    print("✅ OpenAI fallback chain correct")
    
    # Requesting G4F should just give G4F
    chain_g4f = manager.get_fallback_chain("g4f", available)
    print(f"Fallback chain for 'g4f': {chain_g4f}")
    assert chain_g4f == ["g4f"]
    print("✅ G4F fallback chain correct")

    return True


def main():
    """Run all tests."""
    print("G4F Provider Integration Test Suite")
    print("=" * 60)
    
    try:
        # Run synchronous tests
        test_provider_factory_integration()
        test_provider_maps()
        test_priority_manager()
        
        # Run async tests
        asyncio.run(test_g4f_provider_instantiation())
        asyncio.run(test_g4f_completion_non_streaming_contract())
        asyncio.run(test_g4f_completion_streaming_contract())
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! G4F integration is complete and working.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)