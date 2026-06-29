"""
Tests for G4F model ID handling.

This test module verifies that G4F model IDs are properly handled:
- Strip/re-add g4f/ prefix correctly
- Handle complex model IDs gracefully
- Document recommended simple model names
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx


class TestG4FModelIDHandling:
    """Test G4F model ID handling."""

    @pytest.fixture
    def g4f_provider(self):
        """Create a G4F provider instance."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        return G4FProvider()

    def test_strip_provider_prefix_simple(self, g4f_provider):
        """Test stripping g4f/ prefix from simple model ID."""
        assert g4f_provider._strip_provider_prefix("g4f/gpt-4") == "gpt-4"
        assert (
            g4f_provider._strip_provider_prefix("g4f/gpt-3.5-turbo") == "gpt-3.5-turbo"
        )
        assert g4f_provider._strip_provider_prefix("g4f/claude-3") == "claude-3"

    def test_strip_provider_prefix_complex(self, g4f_provider):
        """Test stripping g4f/ prefix from complex model IDs."""
        # These are known problematic model IDs
        assert (
            g4f_provider._strip_provider_prefix("g4f/gpt-4-32k-0613")
            == "gpt-4-32k-0613"
        )
        assert (
            g4f_provider._strip_provider_prefix("g4f/gpt-4-turbo-preview")
            == "gpt-4-turbo-preview"
        )
        assert (
            g4f_provider._strip_provider_prefix("g4f/claude-3-opus-20240229")
            == "claude-3-opus-20240229"
        )

    def test_strip_provider_prefix_without_prefix(self, g4f_provider):
        """Test that model IDs without prefix are returned unchanged."""
        assert g4f_provider._strip_provider_prefix("gpt-4") == "gpt-4"
        assert g4f_provider._strip_provider_prefix("gpt-3.5-turbo") == "gpt-3.5-turbo"

    def test_recommended_simple_model_names(self, g4f_provider):
        """Test that recommended simple model names work correctly."""
        # These are the recommended simple names that should work reliably
        recommended_models = [
            "g4f/gpt-4",
            "g4f/gpt-4o",
            "g4f/gpt-3.5-turbo",
            "g4f/claude-3",
            "g4f/claude-3.5-sonnet",
        ]

        for model in recommended_models:
            stripped = g4f_provider._strip_provider_prefix(model)
            assert stripped.startswith("gpt-") or stripped.startswith("claude-"), (
                f"Stripped model '{stripped}' should be a valid model name"
            )

    def test_has_custom_logic(self, g4f_provider):
        """Test that G4F provider has custom logic."""
        assert g4f_provider.has_custom_logic() is True


class TestG4FModelDiscovery:
    """Test G4F model discovery."""

    @pytest.fixture
    def g4f_provider(self):
        """Create a G4F provider instance."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        return G4FProvider()

    @pytest.mark.asyncio
    async def test_get_models_returns_list(self, g4f_provider):
        """Test that get_models returns a list with g4f/ prefix."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)

        # Mock the client.get response
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-4"},
                {"id": "gpt-3.5-turbo"},
                {"id": "claude-3"},
            ]
        }
        mock_client.get.return_value = mock_response

        models = await g4f_provider.get_models("test-key", mock_client)

        assert isinstance(models, list)
        assert all(m.startswith("g4f/") for m in models), (
            "All models should have g4f/ prefix"
        )


class TestG4FBaseURLSelection:
    """Test G4F base URL selection."""

    @pytest.fixture
    def g4f_provider(self):
        """Create a G4F provider instance."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        return G4FProvider()

    def test_select_base_url_returns_string(self, g4f_provider):
        """Test that _select_base_url returns a valid URL."""
        url = g4f_provider._select_base_url("g4f/gpt-4")
        assert isinstance(url, str)
        assert url.startswith("http")

    def test_select_base_url_handles_simple_models(self, g4f_provider):
        """Test base URL selection for simple model names."""
        simple_models = [
            "g4f/gpt-4",
            "g4f/gpt-3.5-turbo",
            "g4f/claude-3",
        ]

        for model in simple_models:
            url = g4f_provider._select_base_url(model)
            assert isinstance(url, str), f"Should return URL for {model}"


class TestG4FConfiguration:
    """Test G4F configuration in router_config.yaml."""

    @pytest.fixture
    def router_config(self):
        """Load the router configuration."""
        import yaml
        from pathlib import Path

        config_path = Path(__file__).parent.parent / "config" / "router_config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_g4f_provider_enabled(self, router_config):
        """Verify G4F provider is enabled."""
        g4f_config = router_config.get("providers", {}).get("g4f", {})
        assert g4f_config.get("enabled") is True, "G4F should be enabled"

    def test_g4f_used_in_router_models(self, router_config):
        """Verify G4F is used in router_models candidates."""
        router_models = router_config.get("router_models", {})
        g4f_found = False

        for model_name, model_config in router_models.items():
            candidates = model_config.get("candidates", [])
            for candidate in candidates:
                if candidate.get("provider") == "g4f":
                    g4f_found = True
                    # Verify simple model names are used
                    model = candidate.get("model", "")
                    assert "/" not in model, (
                        f"G4F model '{model}' should not contain '/' prefix in config"
                    )

        assert g4f_found, "G4F should be used in at least one router_model"


class TestG4FModelIDRecommendations:
    """Document recommended G4F model IDs."""

    # These model IDs are known to work reliably
    RECOMMENDED_MODELS = [
        "g4f/gpt-4",
        "g4f/gpt-4o",
        "g4f/gpt-3.5-turbo",
        "g4f/claude-3",
        "g4f/claude-3.5-sonnet",
        "g4f/o1-mini",
    ]

    # These model IDs may have issues
    PROBLEMATIC_MODELS = [
        "g4f/gpt-4-32k-0613",  # Complex version suffix
        "g4f/gpt-4-0125-preview",  # Preview version
        "g4f/claude-3-opus-20240229",  # Date suffix
    ]

    def test_recommended_models_are_simple(self):
        """Verify recommended models use simple naming."""
        for model in self.RECOMMENDED_MODELS:
            # Simple models should not have complex suffixes
            parts = model.split("/")
            assert len(parts) == 2, (
                f"Model should have exactly one / separator: {model}"
            )
            model_name = parts[1]
            # Simple models should not have version dates or complex suffixes
            assert "-20" not in model_name, (
                f"Simple model should not have date suffix: {model}"
            )

    def test_problematic_models_have_complex_suffixes(self):
        """Document why certain model IDs are problematic."""
        for model in self.PROBLEMATIC_MODELS:
            parts = model.split("/")
            model_name = parts[1]
            # Problematic models typically have complex versioning
            has_date = "-20" in model_name
            has_complex_version = model_name.count("-") > 1
            assert has_date or has_complex_version, (
                f"Problematic model should have complex suffix: {model}"
            )
