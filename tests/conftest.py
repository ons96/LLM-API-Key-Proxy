"""
Pytest configuration and fixtures for the test suite.
"""
import pytest
import pytest_asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock
import httpx

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as an async test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


@pytest.fixture(scope="session")
def anyio_backend():
    """Set the async backend for pytest-asyncio."""
    return "asyncio"


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx AsyncClient."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


@pytest.fixture
def mock_httpx_response():
    """Create a mock httpx response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.content = b'{"id": "test", "object":", "choices": [] "chat.completion}'
    response.json.return_value = {
        "id": "test-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }
    return response


@pytest.fixture
def sample_messages():
    """Sample messages for chat completion tests."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]


@pytest.fixture
def sample_chat_completion_response():
    """Sample OpenAI-style chat completion response."""
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! I'm doing well, thank you for asking."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 12,
            "total_tokens": 25
        }
    }


@pytest.fixture
def sample_streaming_chunks():
    """Sample streaming chunks for testing."""
    import time
    
    return [
        {
            "id": f"g4f-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": None
            }]
        },
        {
            "id": f"g4f-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {"content": " world"},
                "finish_reason": None
            }]
        },
        {
            "id": f"g4f-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {"content": "!"},
                "finish_reason": "stop"
            }]
        },
        "[DONE]"
    ]


@pytest.fixture
def g4f_provider_config():
    """Sample G4F provider configuration."""
    return {
        "G4F_API_KEY": "test-api-key",
        "G4F_MAIN_API_BASE": "https://g4f-api.example.com",
        "G4F_GROQ_API_BASE": "https://g4f-groq.example.com",
        "G4F_GROK_API_BASE": "https://g4f-grok.example.com",
        "G4F_GEMINI_API_BASE": "https://g4f-gemini.example.com",
        "G4F_NVIDIA_API_BASE": "https://g4f-nvidia.example.com",
    }


@pytest.fixture
def priority_config():
    """Sample provider priority configuration."""
    return {
        "PROVIDER_PRIORITY_OPENAI": "1",
        "PROVIDER_PRIORITY_ANTHROPIC": "1",
        "PROVIDER_PRIORITY_GROQ": "2",
        "PROVIDER_PRIORITY_OPENROUTER": "2",
        "PROVIDER_PRIORITY_GEMINI": "3",
        "PROVIDER_PRIORITY_MISTRAL": "3",
        "PROVIDER_PRIORITY_G4F": "5",
    }
