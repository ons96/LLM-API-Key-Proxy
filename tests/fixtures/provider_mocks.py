"""Mock provider responses for testing."""

import asyncio
import time
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock


class MockProviderResponse:
    """Mock response object that mimics LiteLLM response structure."""
    
    def __init__(
        self,
        content: str = "Test response",
        model: str = "test-model",
        tokens: int = 100,
        delay: float = 0.1,
        usage: Optional[Dict[str, int]] = None
    ):
        self.id = f"chatcmpl-{int(time.time() * 1000)}"
        self.object = "chat.completion"
        self.created = int(time.time())
        self.model = model
        self.delay = delay
        
        if usage is None:
            usage = {
                "prompt_tokens": 50,
                "completion_tokens": tokens,
                "total_tokens": 50 + tokens
            }
        
        self.choices = [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ]
        self.usage = usage
    
    async def wait_for_response(self):
        """Simulate network delay."""
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        return self


class MockProviderError(Exception):
    """Base class for mock provider errors."""
    pass


class RateLimitError(MockProviderError):
    """Mock rate limit error (429)."""
    def __init__(self, retry_after: int = 60):
        super().__init__(f"Rate limit exceeded, retry after {retry_after}s")
        self.retry_after = retry_after
        
    class response:
        headers = {}
        
    def __init__(self, retry_after: int = 60):
        super().__init__(f"Rate limit exceeded, retry after {retry_after}s")
        self.retry_after = retry_after
        self.response = type('obj', (object,), {
            'headers': {'retry-after': str(retry_after)}
        })


class TimeoutError(MockProviderError):
    """Mock timeout error."""
    def __init__(self):
        super().__init__("Request timeout")


class AuthError(MockProviderError):
    """Mock authentication error (401)."""
    def __init__(self):
        super().__init__("Invalid API key")


class ConnectionError(MockProviderError):
    """Mock connection error."""
    def __init__(self):
        super().__init__("Connection failed")


class InvalidRequestError(MockProviderError):
    """Mock invalid request error (400)."""
    def __init__(self):
        super().__init__("Invalid request format")


def create_mock_provider(
    responses: list = None,
    errors: list = None,
    delay: float = 0.1
):
    """
    Create a mock provider that returns specified responses or raises errors.
    
    Args:
        responses: List of MockProviderResponse objects or strings
        errors: List of exception classes to raise
        delay: Default delay for responses
    
    Returns:
        AsyncMock that can be used as a provider
    """
    if responses is None and errors is None:
        responses = [MockProviderResponse(delay=delay)]
    
    call_count = [0]
    
    async def mock_call(*args, **kwargs):
        """Mock call handler."""
        idx = call_count[0]
        call_count[0] += 1
        
        # Check if we should raise an error
        if errors and idx < len(errors):
            error = errors[idx]
            if isinstance(error, type):
                raise error()
            raise error
        
        # Return response
        if responses:
            response_idx = idx if idx < len(responses) else -1
            response = responses[response_idx]
            
            if isinstance(response, str):
                response = MockProviderResponse(content=response, delay=delay)
            
            if hasattr(response, 'wait_for_response'):
                return await response.wait_for_response()
            return response
        
        # Default response
        return await MockProviderResponse(delay=delay).wait_for_response()
    
    mock = AsyncMock(side_effect=mock_call)
    return mock


def create_slow_mock_provider(delay: float = 5.0):
    """Create a mock provider that responds slowly."""
    return create_mock_provider(
        responses=[MockProviderResponse(delay=delay)],
        delay=delay
    )


def create_fast_mock_provider(delay: float = 0.1):
    """Create a mock provider that responds quickly."""
    return create_mock_provider(
        responses=[MockProviderResponse(delay=delay)],
        delay=delay
    )


def create_failing_mock_provider(
    error_type: type = RateLimitError,
    num_failures: int = 1,
    then_succeed: bool = True
):
    """Create a mock provider that fails then optionally succeeds."""
    errors = [error_type for _ in range(num_failures)]
    
    if then_succeed:
        responses = [MockProviderResponse()]
        # Need to make errors list length match attempts
        return create_mock_provider(responses=responses, errors=errors)
    
    return create_mock_provider(errors=errors)


def create_concurrent_safe_mock(base_mock):
    """
    Wrap a mock to be safe for concurrent calls.
    Each call gets independent state.
    """
    return base_mock  # AsyncMock is already thread-safe in asyncio


# Provider-specific mocks

def create_groq_mock(model: str = "llama-3.3-70b-versatile"):
    """Create a mock for Groq provider."""
    return create_mock_provider(
        responses=[MockProviderResponse(
            model=f"groq/{model}",
            content=f"Response from Groq {model}",
            delay=0.05,  # Fast response
            tokens=150
        )]
    )


def create_cerebras_mock(model: str = "llama-3.1-70b"):
    """Create a mock for Cerebras provider."""
    return create_mock_provider(
        responses=[MockProviderResponse(
            model=f"cerebras/{model}",
            content=f"Response from Cerebras {model}",
            delay=0.02,  # Very fast response
            tokens=200
        )]
    )


def create_g4f_mock(model: str = "gpt-4o"):
    """Create a mock for G4F provider."""
    return create_mock_provider(
        responses=[MockProviderResponse(
            model=f"g4f/{model}",
            content=f"Response from G4F {model}",
            delay=0.25,  # Slower response
            tokens=180
        )]
    )


def create_gemini_mock(model: str = "gemini-1.5-flash"):
    """Create a mock for Gemini provider."""
    return create_mock_provider(
        responses=[MockProviderResponse(
            model=f"google/{model}",
            content=f"Response from Gemini {model}",
            delay=0.15,
            tokens=120
        )]
    )
