"""
Test suite for failover scenarios between provider tiers.
"""
import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import os
import sys
import httpx

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set up test environment with priority tiers
os.environ["PROVIDER_PRIORITY_OPENAI"] = "1"
os.environ["PROVIDER_PRIORITY_GROQ"] = "2"
os.environ["PROVIDER_PRIORITY_GEMINI"] = "3"
os.environ["PROVIDER_PRIORITY_G4F"] = "5"
os.environ["G4F_MAIN_API_BASE"] = "https://test-g4f.example.com"

from rotator_library.providers.g4f_provider import G4FProvider
from rotator_library.client import get_provider_priority
from rotator_library.usage_manager import UsageManager


class TestG4FFailoverScenarios:
    """Test G4F provider failover behavior."""
    
    def test_g4f_uses_fallback_priority(self):
        """Test that G4F is correctly configured as fallback."""
        provider = G4FProvider()
        
        # G4F should always have priority 5
        assert provider.get_credential_priority("any-key") == 5
    
    def test_g4f_uses_standard_tier(self):
        """Test that G4F credentials are in standard tier."""
        provider = G4FProvider()
        
        assert provider.get_credential_tier_name("any-key") == "standard"
    
    def test_no_tier_restrictions_for_g4f_models(self):
        """Test that G4F models have no tier requirements."""
        provider = G4FProvider()
        
        # G4F models can be used with any tier
        assert provider.get_model_tier_requirement("gpt-4") is None
        assert provider.get_model_tier_requirement("claude-3-opus") is None


class TestG4FEndpointFailover:
    """Test failover between G4F endpoints."""
    
    def test_get_endpoint_returns_none_when_main_unavailable(self):
        """Test behavior when main endpoint is not configured."""
        with patch.dict(os.environ, {
            "G4F_API_KEY": "test-key",
            # Only groq endpoint configured
            "G4F_GROQ_API_BASE": "https://groq.example.com",
        }, clear=True):
            provider = G4FProvider()
            
            # For unknown model, should route to groq (only available)
            endpoint = provider._get_endpoint_for_model("gpt-4")
            assert endpoint == "https://groq.example.com"
    
    def test_endpoint_selection_with_multiple_available(self):
        """Test endpoint selection when multiple endpoints are available."""
        provider = G4FProvider()
        
        # Model with specific routing
        groq_endpoint = provider._get_endpoint_for_model("groq/llama-3.1-70b")
        assert groq_endpoint == provider._endpoints["groq"]
        
        # Model without specific routing
        main_endpoint = provider._get_endpoint_for_model("gpt-4o")
        assert main_endpoint == provider._endpoints["main"]
    
    def test_fallback_chain_between_endpoints(self):
        """Test that the endpoint fallback chain works correctly."""
        provider = G4FProvider()
        
        # Simulate unavailable main endpoint
        provider._endpoints = {"groq": "https://groq.example.com"}
        
        # Should fall back to groq for unknown model
        endpoint = provider._get_endpoint_for_model("gpt-4")
        assert endpoint == "https://groq.example.com"


class TestG4FCredentialFailover:
    """Test credential-level failover for G4F."""
    
    def test_single_credential_priority(self):
        """Test that single credential has correct priority."""
        provider = G4FProvider()
        
        priority = provider.get_credential_priority("test-credential-123")
        assert priority == 5
    
    def test_all_credentials_same_priority(self):
        """Test that all G4F credentials have the same priority."""
        provider = G4FProvider()
        
        credentials = [
            "cred-1", "cred-2", "cred-3",
            "g4f-api-key-1", "g4f-api-key-2"
        ]
        
        for cred in credentials:
            priority = provider.get_credential_priority(cred)
            assert priority == 5, f"Credential {cred} should have priority 5"


class TestG4FErrorHandling:
    """Test error handling for G4F provider."""
    
    @pytest.mark.asyncio
    async def test_completion_without_configured_endpoint(self):
        """Test completion fails gracefully when no endpoints configured."""
        provider = G4FProvider()
        # Manually clear endpoints
        provider._endpoints = {}
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        
        # Should raise an exception when no endpoints configured
        with pytest.raises(Exception):
            await provider.acompletion(
                mock_client,
                model="g4f/gpt-4",
                messages=[{"role": "user", "content": "Hello"}]
            )
    
    @pytest.mark.asyncio
    async def test_embedding_not_supported(self):
        """Test that embeddings raise appropriate error."""
        provider = G4FProvider()
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        
        with pytest.raises(NotImplementedError) as exc_info:
            await provider.aembedding(
                mock_client,
                model="text-embedding-3-small",
                input=["test text"]
            )
        
        assert "embeddings" in str(exc_info.value).lower()


class TestG4FStreamingFailover:
    """Test streaming response handling and failover."""
    
    @pytest.mark.asyncio
    async def test_streaming_requires_endpoint(self):
        """Test that streaming fails without configured endpoint."""
        with patch.dict(os.environ, {}, clear=True):
            provider = G4FProvider()
            mock_client = AsyncMock(spec=httpx.AsyncClient)
            
            with pytest.raises(Exception):
                # Create generator and try to iterate
                gen = provider.acompletion(
                    mock_client,
                    model="g4f/gpt-4",
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=True
                )
                # Try to get first item
                await asyncio.anext(gen)
    
    def test_parse_empty_chunk(self):
        """Test parsing of empty streaming chunks."""
        provider = G4FProvider()
        
        result = provider._parse_chunk({}, "g4f/gpt-4")
        
        assert result["choices"] == []
        assert result["model"] == "g4f/gpt-4"


class TestG4FResponseFailover:
    """Test response handling and fallback scenarios."""
    
    def test_parse_error_response(self):
        """Test parsing of G4F error responses."""
        provider = G4FProvider()
        
        error_response = {
            "error": {
                "message": "Service temporarily unavailable",
                "code": 503
            }
        }
        
        with pytest.raises(Exception):
            provider._parse_g4f_response(error_response, "g4f/gpt-4")
    
    def test_parse_empty_response(self):
        """Test parsing of empty responses raises error."""
        provider = G4FProvider()
        
        empty_response = {}
        
        with pytest.raises(Exception):
            provider._parse_g4f_response(empty_response, "g4f/gpt-4")
    
    def test_parse_response_with_partial_usage(self):
        """Test parsing response with partial usage information."""
        provider = G4FProvider()
        
        response = {
            "id": "g4f-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10
                # completion_tokens and total_tokens missing
            }
        }
        
        result = provider._parse_g4f_response(response, "g4f/gpt-4")
        
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 0
        assert result.usage.total_tokens == 0


class TestG4FRotationModeFailover:
    """Test rotation mode effects on failover behavior."""
    
    def test_balanced_mode_is_default(self):
        """Test that balanced mode is the default for G4F."""
        provider = G4FProvider()
        
        assert provider.get_rotation_mode("g4f") == "balanced"
    
    def test_sequential_mode_can_be_configured(self):
        """Test that sequential mode can be configured via env var."""
        os.environ["ROTATION_MODE_G4F"] = "sequential"
        
        try:
            provider = G4FProvider()
            assert provider.get_rotation_mode("g4f") == "sequential"
        finally:
            del os.environ["ROTATION_MODE_G4F"]


class TestTierFailoverIntegration:
    """Test failover behavior across different tiers."""
    
    def test_g4f_priority_is_lower_than_all_tier1_providers(self):
        """Verify G4F priority is lower than all tier 1 providers."""
        g4f_priority = get_provider_priority("g4f")
        
        tier1_providers = ["openai"]
        for provider in tier1_providers:
            priority = get_provider_priority(provider)
            assert priority < g4f_priority, (
                f"Tier 1 provider {provider} (P{priority}) should be higher "
                f"priority than G4F (P{g4f_priority})"
            )
    
    def test_g4f_priority_is_lower_than_all_tier2_providers(self):
        """Verify G4F priority is lower than all tier 2 providers."""
        g4f_priority = get_provider_priority("g4f")
        
        tier2_providers = ["groq"]
        for provider in tier2_providers:
            priority = get_provider_priority(provider)
            assert priority < g4f_priority, (
                f"Tier 2 provider {provider} (P{priority}) should be higher "
                f"priority than G4F (P{g4f_priority})"
            )
    
    def test_failover_order_is_correct(self):
        """Test that the failover order follows priority tiers."""
        priorities = {
            "openai": get_provider_priority("openai"),
            "groq": get_provider_priority("groq"),
            "gemini": get_provider_priority("gemini"),
            "g4f": get_provider_priority("g4f"),
        }
        
        # Sort by priority
        sorted_providers = sorted(priorities.items(), key=lambda x: x[1])
        
        # Verify order: openai < groq < gemini < g4f
        assert sorted_providers[0][1] == 1  # openai
        assert sorted_providers[1][1] == 2  # groq
        assert sorted_providers[2][1] == 3  # gemini
        assert sorted_providers[3][1] == 5  # g4f


class TestG4FConfigurationFailover:
    """Test configuration-based failover scenarios."""
    
    def test_partial_endpoint_configuration(self):
        """Test behavior with only some endpoints configured."""
        with patch.dict(os.environ, {
            "G4F_API_KEY": "test-key",
            "G4F_GROQ_API_BASE": "https://groq.example.com",
            # Other endpoints not set
        }, clear=True):
            provider = G4FProvider()
            
            # Should have groq endpoint
            assert "groq" in provider._endpoints
            
            # For model without specific routing, should use groq
            endpoint = provider._get_endpoint_for_model("gpt-4")
            assert endpoint == "https://groq.example.com"
    
    def test_api_key_not_required(self):
        """Test that G4F can work without API key."""
        with patch.dict(os.environ, {
            "G4F_MAIN_API_BASE": "https://g4f.example.com",
            # No G4F_API_KEY set
        }, clear=True):
            provider = G4FProvider()
            
            assert provider._api_key is None
            headers = provider._get_auth_header()
            assert headers == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
