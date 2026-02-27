"""
Pytest configuration and fixtures for G4F provider tests.
"""

import pytest
import asyncio
from typing import AsyncGenerator


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_messages():
    """Standard test messages for chat completion tests."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]


@pytest.fixture
def test_model():
    """Standard test model identifier."""
    return "gpt-3.5-turbo"


@pytest.fixture
def test_temperature():
    """Standard test temperature."""
    return 0.7


@pytest.fixture
def test_max_tokens():
    """Standard test max tokens."""
    return 100


@pytest.fixture
def mock_g4f_module():
    """Mock g4f module for testing."""
    from unittest.mock import Mock

    mock_module = Mock()
    mock_module.ChatCompletion = Mock()
    mock_module.ChatCompletion.create = Mock(return_value="Test response")

    return mock_module


@pytest.fixture
async def g4f_provider():
    """Create and initialize a G4F provider for testing."""
    from src.rotator_library.providers.g4f_provider import G4FProvider

    with pytest.mock.patch('src.rotator_library.providers.g4f_provider.g4f') as mock_g4f:
        provider = G4FProvider()
        await provider.initialize()
        yield provider
        await provider.close()


@pytest.fixture
def provider_config():
    """Provider configuration for testing."""
    return {
        "provider_name": "g4f",
        "base_url": "https://api.g4f.pro",
        "timeout": 30,
        "max_retries": 3
    }
