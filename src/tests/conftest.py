"""
Pytest configuration and shared fixtures.
"""
import pytest
import asyncio
from unittest.mock import Mock


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture to set environment variables for testing."""
    env_vars = {
        "PROXY_API_KEY": "test-api-key-12345",
        "LITELLM_LOG": "DEBUG"
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars
