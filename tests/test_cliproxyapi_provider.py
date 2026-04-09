"""
Unit tests for CLIProxyAPI Provider.

Tests cover:
- Provider initialization
- Model discovery
- Chat completion (streaming and non-streaming)
- Health check
- Error handling
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from src.rotator_library.providers.cliproxyapi_provider import (
    CLIProxyAPIProvider,
    CLIProxyAPIProviderConfig,
    CLIProxyAPIAuthMethod,
)


@pytest.fixture
def provider():
    """Create CLIProxyAPI provider with default config."""
    return CLIProxyAPIProvider()


@pytest.fixture
def provider_with_config():
    """Create CLIProxyAPI provider with custom config."""
    config = CLIProxyAPIProviderConfig(
        base_url="http://test.local:8317",
        timeout=60,
        enabled=True,
    )
    return CLIProxyAPIProvider(config=config)


class TestCLIProxyAPIProviderInit:
    """Test provider initialization."""

    def test_default_initialization(self, provider):
        """Test provider initializes with default config."""
        assert provider.config.base_url == "http://127.0.0.1:8317"
        assert provider.config.timeout == 120
        assert provider.config.enabled is True

    def test_custom_config_initialization(self, provider_with_config):
        """Test provider initializes with custom config."""
        assert provider_with_config.config.base_url == "http://test.local:8317"
        assert provider_with_config.config.timeout == 60

    def test_skip_cost_calculation(self, provider):
        """Test that cost calculation is skipped."""
        assert provider.skip_cost_calculation is True

    def test_has_custom_logic(self, provider):
        """Test that custom logic flag is True."""
        assert provider.has_custom_logic() is True

    def test_supported_backends_defined(self, provider):
        """Test that supported backends are defined."""
        assert "gemini" in provider.SUPPORTED_BACKENDS
        assert "iflow" in provider.SUPPORTED_BACKENDS
        assert "antigravity" in provider.SUPPORTED_BACKENDS
        assert "qwen" in provider.SUPPORTED_BACKENDS

    def test_iflow_uses_cookie_auth(self, provider):
        """Test that iFlow backend uses cookie authentication."""
        iflow_config = provider.SUPPORTED_BACKENDS["iflow"]
        assert iflow_config["auth_method"] == CLIProxyAPIAuthMethod.COOKIE
        assert iflow_config["auto_refresh"] is True


class TestGetModels:
    """Test model discovery."""

    @pytest.mark.asyncio
    async def test_get_models_success(self, provider):
        """Test successful model discovery."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "gemini/gemini-2.5-pro"},
                {"id": "iflow/glm-4-plus"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            models = await provider.get_models("unused", mock_client)

            assert len(models) == 2
            assert "gemini/gemini-2.5-pro" in models
            assert "iflow/glm-4-plus" in models

    @pytest.mark.asyncio
    async def test_get_models_fallback_on_error(self, provider):
        """Test fallback to static model list on connection error."""
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )
            mock_get_client.return_value = mock_client

            models = await provider.get_models("unused", mock_client)

            assert len(models) > 0
            assert any("gemini/" in m for m in models)
            assert any("iflow/" in m for m in models)


class TestChatCompletion:
    """Test chat completion functionality."""

    @pytest.mark.asyncio
    async def test_non_stream_completion(self, provider):
        """Test non-streaming chat completion."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "test-123",
            "choices": [
                {
                    "index": 0,
                    "message": {"content": "Hello!", "role": "assistant"},
                    "finish_reason": "stop",
                }
            ],
            "created": 1234567890,
            "model": "gemini/gemini-2.5-pro",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await provider.acompletion(
                client=AsyncMock(),
                model="gemini/gemini-2.5-pro",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

            assert result.id == "test-123"
            assert len(result.choices) == 1
            assert result.choices[0].message.content == "Hello!"

    @pytest.mark.asyncio
    async def test_stream_completion(self, provider):
        """Test streaming chat completion."""
        sse_lines = [
            b'data: {"id": "test-123", "choices": [{"index": 0, "delta": {"content": "Hel"}}]}',
            b'data: {"id": "test-123", "choices": [{"index": 0, "delta": {"content": "lo!"}}]}',
            b"data: [DONE]",
        ]

        async def mock_aiter_lines():
            for line in sse_lines:
                yield line

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()

            async def mock_stream(*args, **kwargs):
                return mock_response

            mock_client.stream = mock_stream
            mock_get_client.return_value = mock_client

            messages = [{"role": "user", "content": "Hi"}]
            kwargs = {
                "model": "gemini/gemini-2.5-pro",
                "messages": messages,
                "stream": True,
            }

            gen = provider.acompletion(client=AsyncMock(), **kwargs)
            chunks = []
            async for chunk in gen:
                chunks.append(chunk)

            assert len(chunks) == 2
            assert chunks[0].choices[0].delta.content == "Hel"

    @pytest.mark.asyncio
    async def test_completion_with_backend_prefix(self, provider):
        """Test that backend prefix is handled correctly."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "test-123",
            "choices": [
                {"index": 0, "message": {"content": "OK"}, "finish_reason": "stop"}
            ],
            "created": 1234567890,
            "model": "iflow/glm-4-plus",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await provider.acompletion(
                client=AsyncMock(),
                model="iflow/glm-4-plus",
                messages=[{"role": "user", "content": "Test"}],
                stream=False,
            )

            assert result is not None


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, provider):
        """Test health check when CLIProxyAPI is healthy."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "uptime": "1h"}

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await provider.health_check()

            assert result["status"] == "healthy"
            assert "cliproxyapi" in result

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, provider):
        """Test health check when CLIProxyAPI returns error."""
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await provider.health_check()

            assert result["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self, provider):
        """Test health check on connection error."""
        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(
                side_effect=httpx.RequestError("Connection refused")
            )
            mock_get_client.return_value = mock_client

            result = await provider.health_check()

            assert result["status"] == "error"
            assert "Connection refused" in result["error"]


class TestErrorHandling:
    """Test error handling."""

    def test_parse_quota_error_quota_exhausted(self, provider):
        """Test parsing quota exhausted error."""
        error_body = json.dumps(
            {
                "error": {
                    "message": "Quota exhausted for this account",
                    "retry_after": 3600,
                }
            }
        )

        result = provider.parse_quota_error(Exception(), error_body)

        assert result is not None
        assert result["reason"] == "QUOTA_EXHAUSTED"
        assert result["retry_after"] == 3600

    def test_parse_quota_error_rate_limited(self, provider):
        """Test parsing rate limit error."""
        error_body = json.dumps(
            {
                "error": {
                    "message": "Rate limit exceeded",
                    "retry_after": 60,
                }
            }
        )

        result = provider.parse_quota_error(Exception(), error_body)

        assert result is not None
        assert result["reason"] == "RATE_LIMITED"

    def test_parse_quota_error_no_error(self, provider):
        """Test parsing when no quota error."""
        error_body = json.dumps(
            {
                "error": {
                    "message": "Invalid API key",
                }
            }
        )

        result = provider.parse_quota_error(Exception(), error_body)

        assert result is None


class TestTierConfiguration:
    """Test tier and priority configuration."""

    def test_tier_priorities_defined(self, provider):
        """Test that tier priorities are defined."""
        assert "free" in provider.tier_priorities
        assert provider.tier_priorities["free"] == 5

    def test_get_credential_tier_name(self, provider):
        """Test getting tier name for credential."""
        tier = provider.get_credential_tier_name("some-credential")
        assert tier == "free"

    def test_usage_reset_config_defined(self, provider):
        """Test that usage reset config is defined."""
        assert provider.usage_reset_configs is not None
        assert frozenset({5}) in provider.usage_reset_configs


class TestProviderStatus:
    """Test provider status functionality."""

    @pytest.mark.asyncio
    async def test_get_provider_status(self, provider):
        """Test getting provider status."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "gemini/gemini-2.5-pro"},
                {"id": "gemini/gemini-2.5-flash"},
                {"id": "iflow/glm-4-plus"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await provider.get_provider_status()

            assert result["status"] == "healthy"
            assert "providers" in result
            assert "gemini" in result["providers"]
            assert "iflow" in result["providers"]
            assert result["total_models"] == 3
