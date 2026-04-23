"""
Integration Tests for G4F Provider

Tests the G4F (GPT4Free) provider integration including:
- Provider initialization
- Chat completions (non-streaming)
- Streaming chat completions
- Error handling
- Model support
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import AsyncGenerator


# Test configuration
TEST_MODEL = "gpt-3.5-turbo"
TEST_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
]
TEST_TEMPERATURE = 0.7
TEST_MAX_TOKENS = 100


class TestG4FProviderInitialization:
    """Tests for G4F provider initialization."""

    @pytest.mark.asyncio
    async def test_provider_initialization_without_api_key(self):
        """Test provider can be initialized without API key."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        provider = G4FProvider()

        # Should not be initialized yet
        assert not provider._initialized
        assert provider.provider_name == "g4f"

    @pytest.mark.asyncio
    async def test_provider_initialization_with_custom_base_url(self):
        """Test provider with custom base URL."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        custom_url = "https://custom-g4f-api.example.com"
        provider = G4FProvider(base_url=custom_url)

        assert provider.base_url == custom_url

    @pytest.mark.asyncio
    async def test_provider_initialize_success(self):
        """Test successful provider initialization."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        with patch('src.rotator_library.providers.g4f_provider.g4f') as mock_g4f:
            provider = G4FProvider()
            await provider.initialize()

            assert provider._initialized
            assert provider.g4f is mock_g4f

    @pytest.mark.asyncio
    async def test_provider_initialize_twice(self):
        """Test that initializing twice doesn't cause issues."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        with patch('src.rotator_library.providers.g4f_provider.g4f'):
            provider = G4FProvider()
            await provider.initialize()
            first_init_time = provider._initialized

            # Initialize again
            await provider.initialize()
            assert provider._initialized

    @pytest.mark.asyncio
    async def test_provider_initialize_missing_package(self):
        """Test initialization fails gracefully when g4f is not installed."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        with patch.dict('sys.modules', {'g4f': None}):
            # Re-import to trigger the error
            import importlib
            import sys

            # Remove cached module
            if 'g4f' in sys.modules:
                del sys.modules['g4f']

            provider = G4FProvider()

            with patch('builtins.__import__', side_effect=ImportError("No module named 'g4f'")):
                with pytest.raises(RuntimeError, match="g4f package is required"):
                    await provider.initialize()


class TestG4FProviderChatCompletions:
    """Tests for G4F chat completions."""

    @pytest.mark.asyncio
    async def test_chat_completions_basic(self):
        """Test basic chat completion request."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        mock_response = "Hello! I'm doing well, thank you for asking."

        with patch('src.rotator_library.providers.g4f_provider.g4f') as mock_g4f:
            mock_g4f.ChatCompletion.create = Mock(return_value=mock_response)

            provider = G4FProvider()
            await provider.initialize()

            response = await provider.chat_completions(
                messages=TEST_MESSAGES,
                model=TEST_MODEL,
                temperature=TEST_TEMPERATURE,
                max_tokens=TEST_MAX_TOKENS
            )

            # Verify response structure
            assert "id" in response
            assert "object" in response
            assert response["object"] == "chat.completion"
            assert "model" in response
            assert "choices" in response
            assert len(response["choices"]) > 0
            assert "message" in response["choices"][0]
            assert response["choices"][0]["message"]["content"] == mock_response

    @pytest.mark.asyncio
    async def test_chat_completions_with_model_object_response(self):
        """Test chat completion with model object response."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        # Create mock response object similar to actual g4f response
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Test response content"
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        with patch('src.rotator_library.providers.g4f_provider.g4f') as mock_g4f:
            mock_g4f.ChatCompletion.create = Mock(return_value=mock_response)

            provider = G4FProvider()
            await provider.initialize()

            response = await provider.chat_completions(
                messages=TEST_MESSAGES,
                model=TEST_MODEL
            )

            assert response["choices"][0]["message"]["content"] == "Test response content"

    @pytest.mark.asyncio
    async def test_chat_completions_parameters_passed(self):
        """Test that parameters are correctly passed to g4f."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        with patch('src.rotator_library.providers.g4f_provider.g4f') as mock_g4f:
            mock_g4f.ChatCompletion.create = Mock(return_value="test")

            provider = G4FProvider()
            await provider.initialize()

            await provider.chat_completions(
                messages=TEST_MESSAGES,
                model="gpt-4",
                temperature=0.5,
                max_tokens=50,
                custom_param="custom_value"
            )

            # Verify the call was made with correct parameters
            mock_g4f.ChatCompletion.create.assert_called_once()
            call_kwargs = mock_g4f.ChatCompletion.create.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4"
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 50
            assert call_kwargs["custom_param"] == "custom_value"

    @pytest.mark.asyncio
    async def test_chat_completions_auto_initializes(self):
        """Test that chat_completions auto-initializes if not initialized."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        with patch('src.rotator_library.providers.g4f_provider.g4f') as mock_g4f:
            mock_g4f.ChatCompletion.create = Mock(return_value="test")

            provider = G4FProvider()
            # Don't call initialize explicitly

            await provider.chat_completions(messages=TEST_MESSAGES)

            assert provider._initialized


class TestG4FProviderStreaming:
    """Tests for G4F streaming chat completions."""

    @pytest.mark.asyncio
    async def test_chat_completions_stream_basic(self):
        """Test basic streaming chat completion."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        # Mock streaming response
        async def mock_stream():
            yield "Hello"
            yield " "
            yield "world"

        with patch('src.rotator_library.providers.g4f_provider.g4f') as mock_g4f:
            mock_g4f.ChatCompletion.create = Mock(return_value=mock_stream())

            provider = G4FProvider()
            await provider.initialize()

            chunks = []
            async for chunk in provider.chat_completions_stream(
                messages=TEST_MESSAGES,
                model=TEST_MODEL
            ):
                chunks.append(chunk)

            assert len(chunks) > 0
            # Verify chunk structure
            for chunk in chunks:
                assert "id" in chunk
                assert "object" in chunk
                assert chunk["object"] == "chat.completion.chunk"
                assert "choices" in chunk
                assert len(chunk["choices"]) > 0

    @pytest.mark.asyncio
    async def test_chat_completions_stream_with_model_object(self):
        """Test streaming with model object chunks."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        # Create mock chunk objects
        mock_chunk1 = Mock()
        mock_delta1 = Mock()
        mock_delta1.content = "Hello"
        mock_chunk1.choices = [Mock(delta=mock_delta1)]

        mock_chunk2 = Mock()
        mock_delta2 = Mock()
        mock_delta2.content = " world"
        mock_chunk2.choices = [Mock(delta=mock_delta2)]

        async def mock_stream():
            yield mock_chunk1
            yield mock_chunk2

        with patch('src.rotator_library.providers.g4f_provider.g4f') as mock_g4f:
            mock_g4f.ChatCompletion.create = Mock(return_value=mock_stream())

            provider = G4FProvider()
            await provider.initialize()

            chunks = []
            async for chunk in provider.chat_completions_stream(
                messages=TEST_MESSAGES,
                model=TEST_MODEL
            ):
                chunks.append(chunk)

            assert len(chunks) == 2
            assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
            assert chunks[1]["choices"][0]["delta"]["content"] == " world"

    @pytest.mark.asyncio
    async def test_chat_completions_stream_auto_initializes(self):
        """Test that streaming auto-initializes if not initialized."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        async def mock_stream():
            yield "test"

        with patch('src.rotator_library.providers.g4f_provider.g4f') as mock_g4f:
            mock_g4f.ChatCompletion.create = Mock(return_value=mock_stream())

            provider = G4FProvider()

            async for _ in provider.chat_completions_stream(messages=TEST_MESSAGES):
                pass

            assert provider._initialized


class TestG4FProviderInfo:
    """Tests for G4F provider information."""

    @pytest.mark.asyncio
    async def test_supported_models(self):
        """Test supported models list."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        provider = G4FProvider()
        models = provider.supported_models

        assert isinstance(models, list)
        assert len(models) > 0
        assert "gpt-3.5-turbo" in models
        assert "gpt-4" in models

    @pytest.mark.asyncio
    async def test_provider_info(self):
        """Test provider info returns correct structure."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        provider = G4FProvider()
        info = provider.get_provider_info()

        assert "name" in info
        assert info["name"] == "G4F"
        assert "version" in info
        assert "models" in info
        assert "features" in info
        assert "streaming" in info["features"]


class TestG4FProviderErrorHandling:
    """Tests for G4F provider error handling."""

    @pytest.mark.asyncio
    async def test_chat_completions_error_logging(self):
        """Test that errors are properly logged."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        with patch('src.rotator_library.providers.g4f_provider.g4f') as mock_g4f:
            mock_g4f.ChatCompletion.create = Mock(side_effect=Exception("API Error"))

            provider = G4FProvider()
            await provider.initialize()

            with pytest.raises(Exception):
                await provider.chat_completions(messages=TEST_MESSAGES)

    @pytest.mark.asyncio
    async def test_stream_error_handling(self):
        """Test streaming error handling."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        async def mock_stream():
            yield "Hello"
            raise Exception("Stream interrupted")

        with patch('src.rotator_library.providers.g4f_provider.g4f') as mock_g4f:
            mock_g4f.ChatCompletion.create = Mock(return_value=mock_stream())

            provider = G4FProvider()
            await provider.initialize()

            chunks = []
            with pytest.raises(Exception):
                async for chunk in provider.chat_completions_stream(
                    messages=TEST_MESSAGES
                ):
                    chunks.append(chunk)


class TestG4FProviderClose:
    """Tests for G4F provider close/cleanup."""

    @pytest.mark.asyncio
    async def test_close_sets_initialized_false(self):
        """Test that close sets initialized to False."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        with patch('src.rotator_library.providers.g4f_provider.g4f'):
            provider = G4FProvider()
            await provider.initialize()

            assert provider._initialized is True

            await provider.close()

            assert provider._initialized is False


class TestG4FProviderIntegration:
    """Integration tests for end-to-end G4F provider usage."""

    @pytest.mark.asyncio
    async def test_full_chat_completion_flow(self):
        """Test complete chat completion flow."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        mock_response = "This is a test response from the model."

        with patch('src.rotator_library.providers.g4f_provider.g4f') as mock_g4f:
            mock_g4f.ChatCompletion.create = Mock(return_value=mock_response)

            # Initialize provider
            provider = G4FProvider()
            await provider.initialize()

            # Make request
            response = await provider.chat_completions(
                messages=TEST_MESSAGES,
                model=TEST_MODEL,
                temperature=0.8,
                max_tokens=150
            )

            # Verify response
            assert response["model"] == TEST_MODEL
            assert len(response["choices"]) == 1
            assert response["choices"][0]["message"]["role"] == "assistant"
            assert "usage" in response

            # Cleanup
            await provider.close()

    @pytest.mark.asyncio
    async def test_full_streaming_flow(self):
        """Test complete streaming flow."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        async def mock_stream():
            for word in ["Hello", " ", "there", "!"]:
                yield word

        with patch('src.rotator_library.providers.g4f_provider.g4f') as mock_g4f:
            mock_g4f.ChatCompletion.create = Mock(return_value=mock_stream())

            provider = G4FProvider()
            await provider.initialize()

            chunks = []
            async for chunk in provider.chat_completions_stream(
                messages=TEST_MESSAGES,
                model=TEST_MODEL
            ):
                chunks.append(chunk)
                if len(chunks) >= 4:  # Limit for test
                    break

            assert len(chunks) > 0

            await provider.close()

    @pytest.mark.asyncio
    async def test_multiple_requests_same_instance(self):
        """Test multiple requests using the same provider instance."""
        from src.rotator_library.providers.g4f_provider import G4FProvider

        with patch('src.rotator_library.providers.g4f_provider.g4f') as mock_g4f:
            mock_g4f.ChatCompletion.create = Mock(return_value="Response")

            provider = G4FProvider()
            await provider.initialize()

            # Make multiple requests
            for i in range(3):
                response = await provider.chat_completions(
                    messages=TEST_MESSAGES,
                    model=TEST_MODEL
                )
                assert response is not None
                assert "choices" in response

            # Verify g4f was called 3 times
            assert mock_g4f.ChatCompletion.create.call_count == 3
