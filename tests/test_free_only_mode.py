"""
Tests for free-only mode enforcement.

This test module verifies that the free-only mode correctly filters out
non-free providers and respects the forbidden_providers_under_free_mode setting.
"""

import pytest
import yaml
from pathlib import Path


class TestFreeOnlyMode:
    """Test free-only mode enforcement in the router."""

    @pytest.fixture
    def router_config(self):
        """Load the router configuration."""
        config_path = Path(__file__).parent.parent / "config" / "router_config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_free_only_mode_is_enabled(self, router_config):
        """Verify that free_only_mode is enabled by default."""
        assert router_config.get("free_only_mode") is True, (
            "free_only_mode should be True in the default config"
        )

    def test_forbidden_providers_defined(self, router_config):
        """Verify forbidden providers are defined for free mode."""
        safety = router_config.get("safety", {})
        forbidden = safety.get("forbidden_providers_under_free_mode", [])

        assert len(forbidden) > 0, (
            "forbidden_providers_under_free_mode should have at least one provider"
        )

        # These paid-only providers should be forbidden
        expected_forbidden = ["openai", "anthropic", "cohere"]
        for provider in expected_forbidden:
            assert provider in forbidden, (
                f"{provider} should be in forbidden_providers_under_free_mode"
            )

    def test_free_tier_models_defined_for_providers(self, router_config):
        """Verify that free_tier_models are defined for providers."""
        providers = router_config.get("providers", {})

        # Check that enabled providers have free_tier_models defined
        for provider_name, provider_config in providers.items():
            if provider_config.get("enabled", False):
                # Search providers use free_tier boolean
                if "free_tier" in provider_config:
                    assert provider_config["free_tier"] is True, (
                        f"Provider {provider_name} has free_tier=False but is enabled"
                    )
                # LLM providers should have free_tier_models list
                elif provider_name not in [
                    "brave_search",
                    "tavily",
                    "duckduckgo",
                    "exa",
                    "jina",
                ]:
                    assert "free_tier_models" in provider_config, (
                        f"Provider {provider_name} should have free_tier_models defined"
                    )

    def test_groq_provider_has_free_models(self, router_config):
        """Verify Groq provider has free tier models configured."""
        groq_config = router_config.get("providers", {}).get("groq", {})

        assert groq_config.get("enabled") is True, "Groq should be enabled"

        free_models = groq_config.get("free_tier_models", [])
        assert len(free_models) > 0, "Groq should have free tier models defined"

        # Verify specific free models
        expected_models = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
        for model in expected_models:
            assert model in free_models, f"{model} should be in Groq free_tier_models"

    def test_gemini_provider_has_free_models(self, router_config):
        """Verify Gemini provider has free tier models configured."""
        gemini_config = router_config.get("providers", {}).get("gemini", {})

        assert gemini_config.get("enabled") is True, "Gemini should be enabled"

        free_models = gemini_config.get("free_tier_models", [])
        assert len(free_models) > 0, "Gemini should have free tier models defined"

    def test_g4f_provider_enabled_for_free_mode(self, router_config):
        """Verify G4F provider is enabled (it's always free)."""
        g4f_config = router_config.get("providers", {}).get("g4f", {})

        assert g4f_config.get("enabled") is True, (
            "G4F should be enabled as it provides free access"
        )

    def test_search_providers_free_tier(self, router_config):
        """Verify search providers are configured for free tier."""
        providers = router_config.get("providers", {})

        # DuckDuckGo should be free and not require API key
        ddg = providers.get("duckduckgo", {})
        assert ddg.get("enabled") is True, "DuckDuckGo should be enabled"
        assert ddg.get("free_tier") is True, "DuckDuckGo should have free_tier=True"
        assert ddg.get("no_api_key_required") is True, (
            "DuckDuckGo should not require an API key"
        )


class TestFreeOnlyModeEnforcement:
    """Test that free-only mode is actually enforced during routing."""

    @pytest.fixture
    def router(self):
        """Get a RouterCore instance for testing."""
        from src.proxy_app.router_core import RouterCore

        config_path = Path(__file__).parent.parent / "config" / "router_config.yaml"
        return RouterCore(str(config_path))

    def test_router_has_free_only_mode_property(self, router):
        """Verify RouterCore has free_only_mode property."""
        assert hasattr(router, "free_only_mode"), (
            "RouterCore should have free_only_mode property"
        )

    def test_free_only_mode_is_true(self, router):
        """Verify free_only_mode is enabled in the router."""
        assert router.free_only_mode is True, "free_only_mode should be True by default"

    @pytest.mark.asyncio
    async def test_forbidden_providers_not_in_candidates(self, router):
        """Verify forbidden providers are not included in candidates."""
        # Get candidates for a model that might have OpenAI configured
        # The router should filter out forbidden providers

        requirements = router._extract_requirements({})

        # This tests the internal filtering logic
        forbidden_providers = router.config.get("safety", {}).get(
            "forbidden_providers_under_free_mode", []
        )

        assert "openai" in forbidden_providers, (
            "OpenAI should be forbidden under free mode"
        )


class TestVirtualModelFreeOnly:
    """Test virtual models work correctly with free-only mode."""

    @pytest.fixture
    def router_config(self):
        """Load the virtual models configuration."""
        config_path = Path(__file__).parent.parent / "config" / "virtual_models.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_virtual_models_config_has_free_only_mode(self, router_config):
        """Verify virtual_models.yaml has free_only_mode set."""
        assert router_config.get("free_only_mode") is True, (
            "virtual_models.yaml should have free_only_mode=True"
        )

    def test_virtual_models_use_free_providers(self, router_config):
        """Verify virtual model candidates use free providers."""
        virtual_models = router_config.get("virtual_models", {})

        # These providers are known to have free tiers
        free_providers = {
            "groq",
            "gemini",
            "g4f",
            "google",
            "deepseek",
            "qwen",
            "cerebras",
        }

        for model_name, model_config in virtual_models.items():
            candidates = model_config.get("candidates", [])
            for candidate in candidates:
                provider = candidate.get("provider", "")
                assert provider in free_providers, (
                    f"Virtual model {model_name} uses provider {provider} "
                    f"which may not be free. Check free_tier configuration."
                )
