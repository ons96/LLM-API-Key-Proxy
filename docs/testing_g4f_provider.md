# G4F Provider Testing Guide

## Overview

This document describes the integration testing approach for the G4F (GPT4Free) provider.

## Test Structure

### Unit Tests (`test_g4f_provider.py`)

- **TestG4FProviderInitialization**: Tests for provider initialization
- **TestG4FProviderChatCompletions**: Tests for non-streaming chat completions
- **TestG4FProviderStreaming**: Tests for streaming chat completions
- **TestG4FProviderInfo**: Tests for provider information
- **TestG4FProviderErrorHandling**: Tests for error handling
- **TestG4FProviderClose**: Tests for cleanup/close operations
- **TestG4FProviderIntegration**: End-to-end integration tests

### Integration Tests (`test_g4f_integration.py`)

- **TestG4FProviderFactory**: Tests for provider factory integration
- **TestG4FProviderAdapter**: Tests for adapter integration
- **TestG4FRouterIntegration**: Tests for router integration
- **TestG4FProviderEnvironment**: Tests for environment configuration

## Running Tests

### Run all G4F tests:
```bash
pytest src/tests/test_g4f_provider.py -v
```

### Run specific test class:
```bash
pytest src/tests/test_g4f_provider.py::TestG4FProviderChatCompletions -v
```

### Run integration tests:
```bash
pytest src/tests/test_g4f_integration.py -v
```

### Run with coverage:
```bash
pytest src/tests/test_g4f_provider.py --cov=src.rotator_library.providers.g4f_provider --cov-report=html
```

## Test Fixtures

Defined in `conftest.py`:

- `event_loop`: Async event loop for tests
- `test_messages`: Standard test message list
- `test_model`: Standard test model
- `test_temperature`: Standard temperature setting
- `test_max_tokens`: Standard max tokens
- `mock_g4f_module`: Mocked g4f module
- `g4f_provider`: Pre-initialized provider instance
- `provider_config`: Provider configuration

## Mocking Strategy

The tests use mocking to avoid actual API calls:

- `unittest.mock.Mock` for g4f module
- `unittest.mock.patch` to replace g4f imports
- AsyncMock for async methods

## Expected Behavior

1. **No API Key Required**: G4F provider should work without an API key
2. **Auto-initialization**: Provider should auto-initialize on first use
3. **OpenAI Compatibility**: Responses should be in OpenAI format
4. **Streaming Support**: Should support streaming responses
5. **Error Handling**: Should handle and log errors properly
