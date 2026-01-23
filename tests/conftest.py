"""
Pytest configuration and shared fixtures for router tests.
"""

import pytest
import sys
import os
from pathlib import Path

# Add src to Python path for imports
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Prevent launcher from starting during tests
os.environ["SKIP_LAUNCHER"] = "true"


@pytest.fixture
def sample_request():
    """Sample request for testing."""
    return {
        "model": "coding-smart",
        "messages": [
            {"role": "user", "content": "Write a Python function to reverse a string"}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }


@pytest.fixture
def sample_streaming_request():
    """Sample streaming request for testing."""
    return {
        "model": "coding-smart",
        "messages": [
            {"role": "user", "content": "Explain async/await in Python"}
        ],
        "stream": True,
        "temperature": 0.7
    }


@pytest.fixture
def sample_tools_request():
    """Sample request with tools for testing."""
    return {
        "model": "coding-smart",
        "messages": [
            {"role": "user", "content": "What's the weather in San Francisco?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
    }


@pytest.fixture
def sample_vision_request():
    """Sample request with vision for testing."""
    return {
        "model": "coding-smart",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/image.jpg"
                        }
                    }
                ]
            }
        ]
    }


# Configure pytest-asyncio
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "load: marks tests as load/stress tests"
    )


# Pytest asyncio configuration
@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for async tests."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()
