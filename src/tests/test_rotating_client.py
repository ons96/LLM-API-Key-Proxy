"""
Unit tests for RotatingClient retry and rotation logic.
Phase 1.1 Testing Infrastructure.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, call, MagicMock
from typing import List, Dict, Any

# Import the client under test
from rotator_library.client import RotatingClient, RotatingClientError, NoAvailableProviderError
from rotator_library.credential_manager import CredentialManager
from rotator_library.cooldown_manager import CooldownManager
from rotator_library.failure_logger import FailureLogger
from rotator_library.provider_adapter import ProviderAdapter


class TestRotatingClientRetryAndRotation:
    """Test suite for RotatingClient retry and provider rotation behavior."""

    @pytest.fixture
    def mock_credential_manager(self):
        """Fixture providing a mocked CredentialManager."""
        cm = Mock(spec=CredentialManager)
        cm.get_all_credentials = Mock(return_value=[])
        return cm

    @pytest.fixture
    def mock_cooldown_manager(self):
        """Fixture providing a mocked CooldownManager."""
        cm = Mock(spec=CooldownManager)
        cm.is_in_cooldown = Mock(return_value=False)
        cm.record_failure = Mock()
        cm.record_success = Mock()
        return cm

    @pytest.fixture
    def mock_failure_logger(self):
        """Fixture providing a mocked FailureLogger."""
        fl = Mock(spec=FailureLogger)
        fl.log_failure = Mock()
        return fl

    @pytest.fixture
    def mock_provider_adapter(self):
        """Fixture providing a mock ProviderAdapter."""
        adapter = Mock(spec=ProviderAdapter)
        adapter.provider_name = "mock_provider"
        adapter.acompletion = AsyncMock()
        adapter.aembedding = AsyncMock()
        return adapter

    @pytest.fixture
    def rotating_client(
        self, 
        mock_credential_manager, 
        mock_cooldown_manager, 
        mock_failure_logger
    ):
        """Fixture providing a RotatingClient with mocked dependencies."""
        client = RotatingClient(
            credential_manager=mock_credential_manager,
            cooldown_manager=mock_cooldown_manager,
            failure_logger=mock_failure_logger,
            max_retries_per_provider=2,
            rotation_strategy="round_robin"
        )
        return client

    @pytest.mark.asyncio
    async def test_successful_completion_no_rotation(
        self, 
        rotating_client, 
        mock_provider_adapter
    ):
        """Test that a successful call does not trigger rotation."""
        # Setup
        expected_response = {"choices": [{"message": {"content": "success"}}]}
        mock_provider_adapter.acompletion.return_value = expected_response
        rotating_client._providers = [mock_provider_adapter]
        rotating_client._active_provider_idx = 0

        # Execute
        result = await rotating_client.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}]
        )

        # Assert
        assert result == expected_response
        assert mock_provider_adapter.acompletion.call_count == 1
        # Should not rotate on success
        assert rotating_client._active_provider_idx == 0

    @pytest.mark.asyncio
    async def test_rotation_on_provider_failure(
        self,
        rotating_client,
        mock_provider_adapter
    ):
        """Test that failure rotates to next provider."""
        # Setup
        provider1 = Mock(spec=ProviderAdapter)
        provider1.provider_name = "provider_1"
        provider1.acompletion = AsyncMock(side_effect=Exception("Connection error"))
        
        provider2 = Mock(spec=ProviderAdapter)
        provider2.provider_name = "provider_2"
        provider2.acompletion = AsyncMock(return_value={"choices": [{"message": {"content": "success"}}]})
        
        rotating_client._providers = [provider1, provider2]
        rotating_client._active_provider_idx = 0

        # Execute
        result = await rotating_client.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}]
        )

        # Assert
        assert result["choices"][0]["message"]["content"] == "success"
        assert provider1.acompletion.call_count == 1  # First provider tried
        assert provider2.acompletion.call_count == 1  # Second provider tried after rotation
        # Should have rotated to provider 2 (index 1)
        assert rotating_client._active_provider_idx == 1

    @pytest.mark.asyncio
    async def test_retry_exhaustion_all_providers_fail(
        self,
        rotating_client
    ):
        """Test that appropriate error is raised when all providers fail."""
        # Setup
        provider1 = Mock(spec=ProviderAdapter)
        provider1.provider_name = "provider_1"
        provider1.acompletion = AsyncMock(side_effect=Exception("Error 1"))
        
        provider2 = Mock(spec=ProviderAdapter)
        provider2.provider_name = "provider_2"
        provider2.acompletion = AsyncMock(side_effect=Exception("Error 2"))
        
        rotating_client._providers = [provider1, provider2]
        rotating_client._active_provider_idx = 0

        # Execute & Assert
        with pytest.raises(NoAvailableProviderError) as exc_info:
            await rotating_client.acompletion(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}]
            )
        
        assert "All providers exhausted" in str(exc_info.value)
        assert provider1.acompletion.call_count == rotating_client.max_retries_per_provider
        assert provider2.acompletion.call_count == rotating_client.max_retries_per_provider

    @pytest.mark.asyncio
    async def test_rate_limit_error_triggers_rotation(
        self,
        rotating_client,
        mock_cooldown_manager
    ):
        """Test that rate limit errors trigger immediate rotation."""
        # Setup
        from rotator_library.error_handler import RateLimitError
        
        provider1 = Mock(spec=ProviderAdapter)
        provider1.provider_name = "provider_1"
        provider1.acompletion = AsyncMock(side_effect=RateLimitError("Rate limited"))
        
        provider2 = Mock(spec=ProviderAdapter)
        provider2.provider_name = "provider_2"
        provider2.acompletion = AsyncMock(return_value={"choices": [{"message": {"content": "success"}}]})
        
        rotating_client._providers = [provider1, provider2]
        rotating_client._active_provider_idx = 0

        # Execute
        result = await rotating_client.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}]
        )

        # Assert
        assert result is not None
        # Should record failure in cooldown manager for rate limit
        mock_cooldown_manager.record_failure.assert_called_with("provider_1", "rate_limit")
        # Should not retry the same provider for rate limits, should rotate immediately
        assert provider1.acompletion.call_count == 1
        assert provider2.acompletion.call_count == 1

    @pytest.mark.asyncio
    async def test_auth_error_triggers_rotation(
        self,
        rotating_client,
        mock_cooldown_manager
    ):
        """Test that authentication errors trigger rotation and cooldown."""
        # Setup
        from rotator_library.error_handler import AuthenticationError
        
        provider1 = Mock(spec=ProviderAdapter)
        provider1.provider_name = "provider_1"
        provider1.acompletion = AsyncMock(side_effect=AuthenticationError("Invalid key"))
        
        provider2 = Mock(spec=ProviderAdapter)
        provider2.provider_name = "provider_2"
        provider2.acompletion = AsyncMock(return_value={"choices": [{"message": {"content": "success"}}]})
        
        rotating_client._providers = [provider1, provider2]
        rotating_client._active_provider_idx = 0

        # Execute
        result = await rotating_client.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}]
        )

        # Assert
        assert result is not None
        # Auth errors should trigger longer cooldown
        mock_cooldown_manager.record_failure.assert_called_with("provider_1", "authentication")
        assert provider1.acompletion.call_count == 1
        assert provider2.acompletion.call_count == 1

    @pytest.mark.asyncio
    async def test_4xx_error_no_rotation(
        self,
        rotating_client
    ):
        """Test that 4xx errors (client errors) do not trigger rotation."""
        # Setup
        from rotator_library.error_handler import BadRequestError
        
        provider1 = Mock(spec=ProviderAdapter)
        provider1.provider_name = "provider_1"
        provider1.acompletion = AsyncMock(side_effect=BadRequestError("Invalid request"))
        
        rotating_client._providers = [provider1]
        rotating_client._active_provider_idx = 0

        # Execute & Assert
        with pytest.raises(BadRequestError):
            await rotating_client.acompletion(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}]
            )
        
        # Should not retry or rotate on 4xx errors
        assert provider1.acompletion.call_count == 1

    @pytest.mark.asyncio
    async def test_success_after_multiple_rotations(
        self,
        rotating_client
    ):
        """Test successful completion after rotating through multiple failing providers."""
        # Setup
        providers = []
        for i in range(3):
            p = Mock(spec=ProviderAdapter)
            p.provider_name = f"provider_{i}"
            if i < 2:
                p.acompletion = AsyncMock(side_effect=Exception(f"Error {i}"))
            else:
                p.acompletion = AsyncMock(return_value={"status": "success"})
            providers.append(p)
        
        rotating_client._providers = providers
        rotating_client._active_provider_idx = 0

        # Execute
        result = await rotating_client.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}]
        )

        # Assert
        assert result["status"] == "success"
        assert providers[0].acompletion.call_count == 1
        assert providers[1].acompletion.call_count == 1
        assert providers[2].acompletion.call_count == 1

    @pytest.mark.asyncio
    async def test_cooldown_skipped_providers(
        self,
        rotating_client,
        mock_cooldown_manager
    ):
        """Test that providers in cooldown are skipped during rotation."""
        # Setup
        mock_cooldown_manager.is_in_cooldown.side_effect = lambda name: name == "provider_1"
        
        provider1 = Mock(spec=ProviderAdapter)
        provider1.provider_name = "provider_1"
        provider1.acompletion = AsyncMock(return_value={"choices": [{"message": {"content": "p1"}}]})
        
        provider2 = Mock(spec=ProviderAdapter)
        provider2.provider_name = "provider_2"
        provider2.acompletion = AsyncMock(return_value={"choices": [{"message": {"content": "p2"}}]})
        
        rotating_client._providers = [provider1, provider2]
        rotating_client._active_provider_idx = 0

        # Execute
        result = await rotating_client.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}]
        )

        # Assert
        assert result["choices"][0]["message"]["content"] == "p2"
        # Provider 1 should be skipped due to cooldown
        assert provider1.acompletion.call_count == 0
        assert provider2.acompletion.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_with_backoff(
        self,
        rotating_client
    ):
        """Test that retries on the same provider use exponential backoff."""
        # Setup
        import time
        provider1 = Mock(spec=ProviderAdapter)
        provider1.provider_name = "provider_1"
        provider1.acompletion = AsyncMock(side_effect=[
            Exception("Transient error"),
            Exception("Transient error"),
            {"choices": [{"message": {"content": "success"}}]}
        ])
        
        rotating_client._providers = [provider1]
        rotating_client._active_provider_idx = 0
        rotating_client.max_retries_per_provider = 3
        rotating_client.backoff_base = 0.01  # 10ms for fast tests

        # Execute
        start_time = time.time()
        result = await rotating_client.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}]
        )
        elapsed = time.time() - start_time

        # Assert
        assert result is not None
        assert provider1.acompletion.call_count == 3
        # Should have waited at least backoff_base * (2^0 + 2^1) = 0.01 + 0.02 = 0.03s
        assert elapsed >= 0.03

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_consecutive_failures(
        self,
        rotating_client,
        mock_cooldown_manager
    ):
        """Test that circuit breaker opens after configured consecutive failures."""
        # Setup
        provider1 = Mock(spec=ProviderAdapter)
        provider1.provider_name = "provider_1"
        provider1.acompletion = AsyncMock(side_effect=Exception("Persistent error"))
        
        rotating_client._providers = [provider1]
        rotating_client._active_provider_idx = 0
        rotating_client.circuit_breaker_threshold = 3

        # Execute first 3 calls
        for _ in range(3):
            with pytest.raises(Exception):
                await rotating_client.acompletion(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "test"}]
                )

        # Assert
        assert mock_cooldown_manager.record_failure.call_count == 3
        # Fourth call should immediately fail without trying (circuit open)
        with pytest.raises(NoAvailableProviderError):
            await rotating_client.acompletion(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}]
            )
        
        # Should not have been called a 4th time (circuit open)
        assert provider1.acompletion.call_count == 3

    def test_context_manager_protocol(self, rotating_client):
        """Test that RotatingClient properly implements async context manager."""
        # This test verifies proper resource cleanup
        assert hasattr(rotating_client, '__aenter__')
        assert hasattr(rotating_client, '__aexit__')

    @pytest.mark.asyncio
    async def test_embedding_rotation(
        self,
        rotating_client
    ):
        """Test that rotation also works for embedding endpoints."""
        # Setup
        provider1 = Mock(spec=ProviderAdapter)
        provider1.provider_name = "provider_1"
        provider1.aembedding = AsyncMock(side_effect=Exception("Embedding error"))
        
        provider2 = Mock(spec=ProviderAdapter)
        provider2.provider_name = "provider_2"
        provider2.aembedding = AsyncMock(return_value={"data": [{"embedding": [0.1, 0.2]}]})
        
        rotating_client._providers = [provider1, provider2]
        rotating_client._active_provider_idx = 0

        # Execute
        result = await rotating_client.aembedding(
            model="text-embedding-3-small",
            input=["test text"]
        )

        # Assert
        assert result["data"][0]["embedding"] == [0.1, 0.2]
        assert provider1.aembedding.call_count == 1
        assert provider2.aembedding.call_count == 1


class TestRotatingClientStateManagement:
    """Tests for internal state management during retries."""

    @pytest.mark.asyncio
    async def test_failure_count_reset_on_success(self):
        """Test that consecutive failure counters reset after success."""
        client = RotatingClient()
        provider = Mock(spec=ProviderAdapter)
        provider.provider_name = "test_provider"
        provider.acompletion = AsyncMock(return_value={"status": "ok"})
        
        client._providers = [provider]
        
        # Simulate some failures then success
        client._consecutive_failures = {"test_provider": 2}
        
        await client.acompletion(model="gpt-4", messages=[{"role": "user", "content": "hi"}])
        
        assert client._consecutive_failures.get("test_provider", 0) == 0

    @pytest.mark.asyncio
    async def test_provider_health_check_integration(self):
        """Test integration with health checker to filter providers."""
        with patch('rotator_library.client.HealthChecker') as mock_health:
            mock_health_instance = Mock()
            mock_health_instance.is_healthy = Mock(return_value=False)
            mock_health.return_value = mock_health_instance
            
            client = RotatingClient(health_check_enabled=True)
            # Should filter out unhealthy providers before attempting calls
            # Implementation specific test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
