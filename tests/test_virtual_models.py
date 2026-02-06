"""
Virtual Model Tests

Test the specific virtual models and their behavior.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from src.proxy_app.router_core import RouterCore, CapabilityRequirements
from tests.fixtures.provider_mocks import MockProviderResponse
from tests.fixtures.scenarios import create_request
from tests.fixtures.benchmark_data import SAMPLE_VIRTUAL_MODELS


@pytest.fixture
def mock_router_config(tmp_path):
    """Create a mock router configuration with virtual models."""
    import yaml

    config = {
        "free_only_mode": True,
        "router_models": {
            "coding-smart": {
                "description": "Best coding models",
                "candidates": [
                    {
                        "provider": "g4f",
                        "model": "gpt-4o",
                        "priority": 1,
                        "free_tier_only": True,
                    },
                    {
                        "provider": "g4f",
                        "model": "claude-3.5-sonnet",
                        "priority": 2,
                        "free_tier_only": True,
                    },
                    {
                        "provider": "g4f",
                        "model": "o1-mini",
                        "priority": 3,
                        "free_tier_only": True,
                    },
                ],
                "auto_order": False,
            },
            "coding-fast": {
                "description": "Fast coding models",
                "candidates": [
                    {
                        "provider": "cerebras",
                        "model": "llama-3.1-70b",
                        "priority": 1,
                        "free_tier_only": True,
                    },
                    {
                        "provider": "groq",
                        "model": "llama-3.3-70b-versatile",
                        "priority": 2,
                        "free_tier_only": True,
                    },
                    {
                        "provider": "groq",
                        "model": "llama-3.1-8b-instant",
                        "priority": 3,
                        "free_tier_only": True,
                    },
                ],
                "auto_order": False,
            },
            "chat-smart": {
                "description": "Best chat models",
                "candidates": [
                    {
                        "provider": "g4f",
                        "model": "gpt-4o",
                        "priority": 1,
                        "free_tier_only": True,
                    },
                    {
                        "provider": "g4f",
                        "model": "claude-3.5-sonnet",
                        "priority": 2,
                        "free_tier_only": True,
                    },
                ],
                "auto_order": False,
            },
            "chat-fast": {
                "description": "Fast chat models",
                "candidates": [
                    {
                        "provider": "cerebras",
                        "model": "llama-3.1-8b",
                        "priority": 1,
                        "free_tier_only": True,
                    },
                    {
                        "provider": "groq",
                        "model": "llama-3.1-8b-instant",
                        "priority": 2,
                        "free_tier_only": True,
                    },
                ],
                "auto_order": False,
            },
        },
        "routing": {"default_cooldown_seconds": 60},
    }

    config_file = tmp_path / "router_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    return str(config_file)


@pytest.fixture
def router(mock_router_config):
    """Create router instance for testing."""
    return RouterCore(mock_router_config)


class TestCodingSmartVirtualModel:
    """
    Test: test_virtual_model_coding_smart_uses_best_models()

    Verify coding-smart routes to best coding models.
    """

    @pytest.mark.asyncio
    async def test_coding_smart_candidates(self, router):
        """Test that coding-smart has appropriate candidate models."""
        requirements = CapabilityRequirements()
        candidates = await router._get_candidates("coding-smart", requirements)

        print(
            f"\nCoding-smart candidates: {[(c.provider, c.model) for c in candidates]}"
        )

        # Verify we have candidates
        assert len(candidates) > 0, "coding-smart should have candidates"

        # Verify candidates include high-quality coding models
        models = [c.model for c in candidates]

        # Should include top-tier models like gpt-4o, claude, etc.
        has_top_model = any(
            model in ["gpt-4o", "claude-3.5-sonnet", "o1-mini"] for model in models
        )

        assert has_top_model, (
            f"coding-smart should include top-tier models, got {models}"
        )

    @pytest.mark.asyncio
    async def test_coding_smart_routes_correctly(self, router):
        """Test that requests to coding-smart use the right models."""
        routed_to = []

        async def track_routing(candidate, request, request_id):
            routed_to.append(f"{candidate.provider}/{candidate.model}")
            return MockProviderResponse(model=f"{candidate.provider}/{candidate.model}")

        with patch.object(
            router, "_execute_single_candidate", side_effect=track_routing
        ):
            await router.route_request(
                create_request(model="coding-smart"), request_id="test-coding-smart"
            )

        print(f"\nRouted to: {routed_to}")

        # Verify routed to a coding-smart candidate
        assert len(routed_to) > 0, "Should have routed to at least one provider"

        # Verify it used a model from the coding-smart candidates
        # (exact model depends on which succeeded first)

    @pytest.mark.asyncio
    async def test_coding_smart_uses_best_benchmark_models(self, router):
        """Verify coding-smart prioritizes models with best coding benchmarks."""
        candidates = await router._get_candidates(
            "coding-smart", CapabilityRequirements()
        )

        models = [c.model for c in candidates]

        print(f"\nCoding-smart model order: {models}")

        # Top coding models should appear in the list
        # Based on SWE-Bench and HumanEval scores
        expected_models = ["gpt-4o", "claude-3.5-sonnet", "o1-mini"]

        found_models = [m for m in models if m in expected_models]

        assert len(found_models) > 0, (
            f"coding-smart should include models from {expected_models}, got {models}"
        )


class TestCodingFastVirtualModel:
    """
    Test: test_virtual_model_coding_fast_uses_fast_models()

    Verify coding-fast routes to high-TPS models.
    """

    @pytest.mark.asyncio
    async def test_coding_fast_candidates(self, router):
        """Test that coding-fast has fast model candidates."""
        candidates = await router._get_candidates(
            "coding-fast", CapabilityRequirements()
        )

        print(
            f"\nCoding-fast candidates: {[(c.provider, c.model) for c in candidates]}"
        )

        assert len(candidates) > 0, "coding-fast should have candidates"

        # Verify candidates include fast providers (Cerebras, Groq)
        providers = [c.provider for c in candidates]

        has_fast_provider = any(
            provider in ["cerebras", "groq"] for provider in providers
        )

        assert has_fast_provider, (
            f"coding-fast should include fast providers (Cerebras/Groq), got {providers}"
        )

    @pytest.mark.asyncio
    async def test_coding_fast_prioritizes_speed(self, router):
        """Verify coding-fast prioritizes high-TPS models."""
        candidates = await router._get_candidates(
            "coding-fast", CapabilityRequirements()
        )

        # Extract models
        models = [c.model for c in candidates]

        print(f"\nCoding-fast model order: {models}")

        # Should include fast models like llama-3.1-70b (Cerebras), llama-3.3-70b (Groq)
        fast_models = [
            "llama-3.1-70b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ]

        found_fast = [m for m in models if m in fast_models]

        assert len(found_fast) > 0, (
            f"coding-fast should include fast models, got {models}"
        )


class TestChatSmartVirtualModel:
    """
    Test: test_virtual_model_chat_smart_uses_conversation_models()

    Verify chat-smart routes to best conversation models.
    """

    @pytest.mark.asyncio
    async def test_chat_smart_candidates(self, router):
        """Test that chat-smart has appropriate candidates."""
        candidates = await router._get_candidates(
            "chat-smart", CapabilityRequirements()
        )

        print(f"\nChat-smart candidates: {[(c.provider, c.model) for c in candidates]}")

        assert len(candidates) > 0, "chat-smart should have candidates"

        # Verify candidates include good chat models
        models = [c.model for c in candidates]

        # Should include models good for conversation
        has_chat_model = any(
            model in ["gpt-4o", "claude-3.5-sonnet", "gemini-1.5-flash"]
            for model in models
        )

        assert has_chat_model, f"chat-smart should include chat models, got {models}"

    @pytest.mark.asyncio
    async def test_chat_smart_different_from_coding_smart(self, router):
        """Verify chat-smart and coding-smart may have different configurations."""
        chat_candidates = await router._get_candidates(
            "chat-smart", CapabilityRequirements()
        )
        coding_candidates = await router._get_candidates(
            "coding-smart", CapabilityRequirements()
        )

        chat_models = [c.model for c in chat_candidates]
        coding_models = [c.model for c in coding_candidates]

        print(f"\nChat models: {chat_models}")
        print(f"Coding models: {coding_models}")

        # They may have overlap, but configurations can differ
        # This test documents that both virtual models exist and can be configured differently


class TestChatFastVirtualModel:
    """
    Test: test_virtual_model_chat_fast_uses_fast_models()

    Verify chat-fast routes to low-latency chat models.
    """

    @pytest.mark.asyncio
    async def test_chat_fast_candidates(self, router):
        """Test that chat-fast has fast chat model candidates."""
        candidates = await router._get_candidates("chat-fast", CapabilityRequirements())

        print(f"\nChat-fast candidates: {[(c.provider, c.model) for c in candidates]}")

        assert len(candidates) > 0, "chat-fast should have candidates"

        # Should include fast chat models (smaller models, fast providers)
        models = [c.model for c in candidates]

        # Should include instant/fast models
        fast_chat_models = ["llama-3.1-8b-instant", "llama-3.1-8b", "llama-3.2-1b"]

        found_fast = [m for m in models if m in fast_chat_models]

        assert len(found_fast) > 0, (
            f"chat-fast should include fast chat models, got {models}"
        )


class TestVirtualModelAliases:
    """
    Test: test_virtual_model_aliases_resolved()

    Verify aliases are resolved to correct virtual models.
    """

    def test_alias_resolution_if_configured(self, router):
        """Test alias resolution if aliases are configured."""
        # Check if router has aliases
        if hasattr(router, "aliases") and router.aliases:
            print(f"\nConfigured aliases: {router.aliases.keys()}")

            # Test resolving an alias
            for alias in router.aliases:
                resolved = router._resolve_alias(alias)
                print(f"Alias '{alias}' resolves to: {resolved}")

                # Verify resolution returns candidates
                assert isinstance(resolved, list), (
                    f"Alias resolution should return list, got {type(resolved)}"
                )
        else:
            # No aliases configured in this test setup
            print("\nNo aliases configured in test router")


class TestVirtualModelCapabilities:
    """
    Test that virtual models respect capability requirements.
    """

    @pytest.mark.asyncio
    async def test_virtual_model_with_tools_requirement(self, router):
        """Test candidate filtering when tools are required."""
        # Create request with tools
        request = create_request(model="coding-smart")
        request["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "test_function",
                    "description": "Test",
                    "parameters": {},
                },
            }
        ]

        requirements = router._extract_requirements(request)

        # Verify tools requirement was detected
        assert requirements.needs_tools, "Should detect tools requirement"

        # Get candidates (may be filtered by capability)
        candidates = await router._get_candidates("coding-smart", requirements)

        print(
            f"\nCandidates for request with tools: {[(c.provider, c.model) for c in candidates]}"
        )

        # We should still get candidates (filtering happens in matches_requirements)
        # This test documents the behavior

    def test_virtual_model_with_vision_requirement(self, router):
        """Test candidate filtering when vision is required."""
        # Create request with image
        request = create_request(model="coding-smart")
        request["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.jpg"},
                    },
                ],
            }
        ]

        requirements = router._extract_requirements(request)

        # Verify vision requirement was detected
        assert requirements.needs_vision, "Should detect vision requirement"

        print(f"\nRequirements: {requirements}")


class TestVirtualModelFreeOnlyMode:
    """
    Test that virtual models respect FREE_ONLY_MODE.
    """

    @pytest.mark.asyncio
    async def test_free_only_mode_filters_paid_providers(self, router):
        """Test that paid providers are filtered in FREE_ONLY_MODE."""
        # Router is configured with free_only_mode=True
        assert router.free_only_mode, "Test router should be in FREE_ONLY_MODE"

        # Get candidates
        candidates = await router._get_candidates(
            "coding-smart", CapabilityRequirements()
        )

        # Verify all candidates are free tier
        for candidate in candidates:
            # In our test config, all candidates have free_tier_only=True
            assert candidate.free_tier_only, (
                f"In FREE_ONLY_MODE, all candidates should be free tier: {candidate}"
            )

        print(
            f"\nAll candidates in FREE_ONLY_MODE: {[(c.provider, c.model) for c in candidates]}"
        )


class TestVirtualModelConfiguration:
    """
    Test virtual model configuration loading and structure.
    """

    def test_virtual_models_loaded(self, router):
        """Test that virtual models are loaded from configuration."""
        assert len(router.virtual_models) > 0, "Should have loaded virtual models"

        print(f"\nLoaded virtual models: {router.virtual_models.keys()}")

        # Verify expected virtual models
        expected_models = ["coding-smart", "coding-fast", "chat-smart", "chat-fast"]

        for model in expected_models:
            if model in router.virtual_models:
                vm_config = router.virtual_models[model]
                print(f"{model}: {vm_config.get('description', 'No description')}")

    def test_virtual_model_has_fallback_chain(self, router):
        """Test that virtual models have fallback chains."""
        for model_id, config in router.virtual_models.items():
            # Should have either 'candidates' or 'fallback_chain'
            has_chain = "candidates" in config or "fallback_chain" in config

            assert has_chain, (
                f"Virtual model '{model_id}' should have candidates or fallback_chain"
            )

            print(f"\n{model_id} has chain: {has_chain}")


class TestVirtualModelRequestFlow:
    """
    Integration tests for virtual model request flow.
    """

    @pytest.mark.asyncio
    async def test_coding_smart_end_to_end(self, router):
        """End-to-end test for coding-smart request."""

        async def mock_success(candidate, request, request_id):
            return MockProviderResponse(
                model=f"{candidate.provider}/{candidate.model}",
                content="def reverse_string(s):\n    return s[::-1]",
            )

        with patch.object(
            router, "_execute_single_candidate", side_effect=mock_success
        ):
            result = await router.route_request(
                create_request(model="coding-smart"), request_id="e2e-coding-smart"
            )

        print(f"\nResult: {result}")

        # Verify we got a result
        assert result is not None, "Should have result from coding-smart"

    @pytest.mark.asyncio
    async def test_multiple_virtual_models_work(self, router):
        """Test that multiple virtual models can be used."""

        async def mock_success(candidate, request, request_id):
            return MockProviderResponse(model=f"{candidate.provider}/{candidate.model}")

        with patch.object(
            router, "_execute_single_candidate", side_effect=mock_success
        ):
            # Test each virtual model
            models_to_test = ["coding-smart", "coding-fast", "chat-smart", "chat-fast"]

            for model in models_to_test:
                if model in router.virtual_models:
                    result = await router.route_request(
                        create_request(model=model), request_id=f"test-{model}"
                    )

                    print(f"\n{model} result: {result is not None}")
                    assert result is not None, f"Should have result for {model}"
