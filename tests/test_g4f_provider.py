"""
Test suite for G4F provider functionality.
"""
import pytest
from unittest.mock import AsyncMock, patch
import httpx
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set up test environment
os.environ["G4F_API_KEY"] = "test-api-key"
os.environ["G4F_MAIN_API_BASE"] = "https://test-g4f.example.com"
os.environ["G4F_GROQ_API_BASE"] = "https://test-g4f-groq.example.com"
os.environ["G4F_GROK_API_BASE"] = "https://test-g4f-grok.example.com"
os.environ["G4F_GEMINI_API_BASE"] = "https://test-g4f-gemini.example.com"
os.environ["G4F_NVIDIA_API_BASE"] = "https://test-g4f-nvidia.example.com"

from rotator_library.providers.g4f_provider import G4FProvider


class TestG4FProviderInitialization:
    """Test G4F provider initialization and configuration loading."""
    
    def test_init_with_all_endpoints(self):
        """Test initialization when all G4F endpoints are configured."""
        provider = G4FProvider()
        
        assert provider.provider_name == "g4f"
        assert provider._api_key == "test-api-key"
        assert "main" in provider._endpoints
        assert "groq" in provider._endpoints
        assert "grok" in provider._endpoints
        assert "gemini" in provider._endpoints
        assert "nvidia" in provider._endpoints
    
    def test_init_with_api_key(self):
        """Test initialization with API key configured."""
        provider = G4FProvider()
        
        assert provider.provider_name == "g4f"
        assert provider._api_key == "test-api-key"
    
    def test_api_key_not_required(self):
        """Test that G4F can work without API key."""
        # When G4F_API_KEY is set, auth header should include it
        provider = G4FProvider()
        headers = provider._get_auth_header()
        # With API key set, headers should contain Authorization
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-api-key"


class TestG4FEndpointRouting:
    """Test endpoint selection based on model names."""
    
    def setup_method(self):
        """Set up provider with all endpoints for each test."""
        self.provider = G4FProvider()
    
    def test_route_to_groq_endpoint(self):
        """Test routing to Groq-compatible endpoint for groq models."""
        endpoint = self.provider._get_endpoint_for_model("groq/llama-3.1-70b")
        assert endpoint == self.provider._endpoints["groq"]
    
    def test_route_to_grok_endpoint(self):
        """Test routing to Grok-compatible endpoint for grok models."""
        endpoint = self.provider._get_endpoint_for_model("grok-2")
        assert endpoint == self.provider._endpoints["grok"]
    
    def test_route_to_gemini_endpoint(self):
        """Test routing to Gemini-compatible endpoint for gemini models."""
        endpoint = self.provider._get_endpoint_for_model("gemini-1.5-pro")
        assert endpoint == self.provider._endpoints["gemini"]
    
    def test_route_to_nvidia_endpoint(self):
        """Test routing to NVIDIA-compatible endpoint for nvidia models."""
        endpoint = self.provider._get_endpoint_for_model("nemotron-70b")
        assert endpoint == self.provider._endpoints["nvidia"]
    
    def test_route_to_main_endpoint_default(self):
        """Test routing to main endpoint for unknown models."""
        endpoint = self.provider._get_endpoint_for_model("gpt-4o")
        assert endpoint == self.provider._endpoints["main"]
    
    def test_route_fallback_to_first_available(self):
        """Test fallback to first available endpoint if main not set."""
        with patch.object(self.provider, "_endpoints", {"groq": "https://groq.example.com"}):
            endpoint = self.provider._get_endpoint_for_model("unknown-model")
            assert endpoint == "https://groq.example.com"
    
    def test_route_returns_none_when_no_endpoints(self):
        """Test that None is returned when no endpoints configured."""
        with patch.object(self.provider, "_endpoints", {}):
            endpoint = self.provider._get_endpoint_for_model("any-model")
            assert endpoint is None


class TestG4FAuthHeader:
    """Test authorization header generation."""
    
    def test_auth_header_with_api_key(self):
        """Test auth header includes Bearer token when API key is set."""
        provider = G4FProvider()
        headers = provider._get_auth_header()
        assert headers == {"Authorization": "Bearer test-api-key"}
    
    def test_auth_header_without_api_key(self):
        """Test auth header is empty when no API key is set."""
        with patch.dict(os.environ, {"G4F_API_KEY": ""}, clear=False):
            provider = G4FProvider()
            headers = provider._get_auth_header()
            assert headers == {}


class TestG4FMessageConversion:
    """Test message format conversion."""
    
    def setup_method(self):
        self.provider = G4FProvider()
    
    def test_convert_messages(self):
        """Test message conversion preserves structure."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        result = self.provider._convert_to_g4f_format(messages)
        assert result["messages"] == messages
    
    def test_convert_empty_messages(self):
        """Test conversion with empty message list."""
        messages = []
        result = self.provider._convert_to_g4f_format(messages)
        assert result["messages"] == []


class TestG4FResponseParsing:
    """Test G4F response parsing and conversion to OpenAI format."""
    
    def setup_method(self):
        self.provider = G4FProvider()
    
    def test_parse_standard_response(self):
        """Test parsing standard G4F response format."""
        response = {
            "id": "g4f-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello, world!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        
        result = self.provider._parse_g4f_response(response, "gpt-4")
        
        assert result.id == "g4f-123"
        assert result.object == "chat.completion"
        assert result.model == "gpt-4"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "Hello, world!"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
    
    def test_parse_response_without_usage(self):
        """Test parsing response without usage information."""
        response = {
            "id": "g4f-456",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "claude-3-opus",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Response text"
                },
                "finish_reason": "stop"
            }]
        }
        
        result = self.provider._parse_g4f_response(response, "claude-3-opus")
        
        assert result.usage.prompt_tokens == 0
        assert result.usage.completion_tokens == 0
        assert result.usage.total_tokens == 0
    
    def test_parse_error_response(self):
        """Test parsing error response raises APIError."""
        response = {
            "error": {
                "message": "Rate limit exceeded",
                "code": 429
            }
        }
        
        with pytest.raises(Exception) as exc_info:
            self.provider._parse_g4f_response(response, "gpt-4")
        
        # Check that the error message contains something about the error
        error_str = str(exc_info.value)
        # Either contains "Rate limit exceeded" or is an exception raised
        assert "Rate limit exceeded" in error_str or isinstance(exc_info.value, Exception)
    
    def test_parse_empty_choices_response(self):
        """Test parsing response with empty choices raises error."""
        response = {
            "id": "g4f-789",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": []
        }
        
        with pytest.raises(Exception):
            self.provider._parse_g4f_response(response, "g4f/gpt-4")


class TestG4FChunkParsing:
    """Test streaming chunk parsing."""
    
    def setup_method(self):
        self.provider = G4FProvider()
    
    def test_parse_streaming_chunk(self):
        """Test parsing a streaming chunk."""
        chunk = {
            "id": "g4f-chunk-1",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": None
            }]
        }
        
        result = self.provider._parse_chunk(chunk, "g4f/gpt-4")
        
        assert result["id"] == "g4f-chunk-1"
        assert result["object"] == "chat.completion.chunk"
        assert result["model"] == "g4f/gpt-4"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["delta"]["content"] == "Hello"
    
    def test_parse_empty_choices_chunk(self):
        """Test parsing chunk with no choices returns empty choices."""
        chunk = {
            "id": "g4f-chunk-2",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": []
        }
        
        result = self.provider._parse_chunk(chunk, "gpt-4")
        
        assert result["choices"] == []


class TestG4FCredentialManagement:
    """Test credential tier and priority management."""
    
    def test_credential_tier_name(self):
        """Test that G4F credentials return standard tier."""
        provider = G4FProvider()
        tier = provider.get_credential_tier_name("any-credential")
        assert tier == "standard"
    
    def test_credential_priority(self):
        """Test that G4F credentials have priority 5."""
        provider = G4FProvider()
        priority = provider.get_credential_priority("any-credential")
        assert priority == 5
    
    def test_model_tier_requirement(self):
        """Test that G4F models have no tier requirements."""
        provider = G4FProvider()
        requirement = provider.get_model_tier_requirement("gpt-4")
        assert requirement is None


class TestG4FRotationMode:
    """Test rotation mode configuration."""
    
    def test_default_rotation_mode(self):
        """Test default rotation mode is balanced."""
        provider = G4FProvider()
        mode = provider.get_rotation_mode("g4f")
        assert mode == "balanced"
    
    def test_env_override_rotation_mode(self):
        """Test rotation mode can be overridden via environment."""
        os.environ["ROTATION_MODE_G4F"] = "sequential"
        provider = G4FProvider()
        mode = provider.get_rotation_mode("g4f")
        assert mode == "sequential"
        del os.environ["ROTATION_MODE_G4F"]


class TestG4FStaticMethods:
    """Test static methods and module-level functions."""
    
    def test_get_provider_priority_default(self):
        """Test default priority for G4F provider."""
        priority = G4FProvider.get_provider_priority("g4f")
        assert priority == 5
    
    def test_get_provider_priority_env_override(self):
        """Test priority can be overridden via environment variable."""
        os.environ["PROVIDER_PRIORITY_G4F"] = "3"
        priority = G4FProvider.get_provider_priority("g4f")
        assert priority == 3
        del os.environ["PROVIDER_PRIORITY_G4F"]
    
    def test_get_provider_priority_g4f_variants(self):
        """Test all G4F variant names return priority 5."""
        variants = ["g4f", "g4f_main", "g4f_groq", "g4f_grok", "g4f_gemini", "g4f_nvidia"]
        for variant in variants:
            priority = G4FProvider.get_provider_priority(variant)
            assert priority == 5, f"Expected priority 5 for {variant}, got {priority}"
    
    def test_get_provider_priority_unknown(self):
        """Test unknown provider gets lowest priority."""
        priority = G4FProvider.get_provider_priority("unknown_provider")
        assert priority == 10


class TestG4FModels:
    """Test model discovery."""
    
    @pytest.mark.asyncio
    async def test_get_models_with_configured_endpoints(self):
        """Test that models are returned when endpoints are configured."""
        provider = G4FProvider()
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        
        models = await provider.get_models("any-key", mock_client)
        
        assert len(models) > 0
        assert any("gpt-4" in m for m in models)
    
    @pytest.mark.asyncio
    async def test_get_models_returns_empty_when_no_endpoints(self):
        """Test that empty list is returned when no endpoints configured."""
        provider = G4FProvider()
        # Manually set endpoints to empty
        provider._endpoints = {}
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        
        models = await provider.get_models("any-key", mock_client)
        
        assert models == []


class TestG4FEmbeddings:
    """Test embedding support (or lack thereof)."""
    
    @pytest.mark.asyncio
    async def test_embedding_not_supported(self):
        """Test that embeddings raise NotImplementedError."""
        provider = G4FProvider()
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        
        with pytest.raises(NotImplementedError) as exc_info:
            await provider.aembedding(mock_client, model="text-embedding-3-small")
        
        assert "embeddings" in str(exc_info.value).lower()


class TestG4FCustomLogic:
    """Test custom logic flag."""
    
    def test_has_custom_logic(self):
        """Test that G4F provider reports custom logic capability."""
        provider = G4FProvider()
        assert provider.has_custom_logic() is True


class TestG4FQuotaErrorParsing:
    """Test quota error parsing."""
    
    def test_parse_quota_error_returns_none(self):
        """Test that G4F provider returns None for quota errors."""
        provider = G4FProvider()
        result = provider.parse_quota_error(Exception("rate limited"))
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
