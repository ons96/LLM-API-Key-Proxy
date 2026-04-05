# AGENTS.md - Unified LLM Provider Gateway

## 1. Role/Mission

### Mission
Build a **Unified LLM Provider Gateway** - a robust abstraction layer that standardizes interactions across multiple LLM providers (OpenAI, Anthropic, Google AI, Ollama, etc.). The system enables:

- **Provider Agnosticism**: Write code once, switch providers via configuration
- **Automatic Fallback**: Graceful degradation when primary providers fail
- **Unified Interface**: Single API for chat completions, embeddings, and model listing
- **Provider Adapters**: Extensible architecture for adding new providers
- **Configuration Management**: Environment-driven provider selection and credentials

### Role for Autonomous Agent
You are the lead developer responsible for designing, implementing, testing, and documenting this gateway system. Make independent technical decisions, write production-quality code, and ensure the system is extensible for future providers.

---

## 2. Technical Stack

### Core Technologies
| Category | Technology | Rationale |
|----------|------------|------------|
| **Language** | Python 3.11+ | Rich async ecosystem, type safety |
| **HTTP Client** | httpx | Async-native, sync/async support |
| **Configuration** | python-dotenv + pydantic-settings | Type-safe config management |
| **Async Runtime** | asyncio | Concurrent provider requests |
| **Testing** | pytest + pytest-asyncio | Async test support |
| **Mocking** | respx / unittest.mock | HTTP mocking for tests |
| **Type Checking** | mypy | Static type validation |
| **Linting** | ruff | Fast linting/formatting |

### Provider SDKs (Free Tiers / Local)
| Provider | SDK/Method | Free Resource |
|----------|------------|------------|
| **Ollama** | Direct HTTP | Local LLaMA models |
| **LM Studio** | Direct HTTP | Local models |
| **OpenAI** | openai Python SDK | Free tier (limited) |
| **Google AI** | google-generativeai | Free tier available |
| **Groq** | Direct HTTP | Free ultra-fast inference |

### Excluded (Not Free)
- Anthropic (requires paid account)
- OpenAI paid tier
- Azure OpenAI

---

## 3. Requirements (Numbered)

### Core Requirements

1. **Unified API Interface**
   - Implement `LLMClient` abstract base class with methods:
     - `chat_completion(messages: list[Message]) -> ChatResponse`
     - `stream_chat_completion(messages: list[Message]) -> AsyncIterator[Chunk]`
     - `list_models() -> list[Model]`
     - `embeddings(text: str | list[str]) -> list[Embedding]`
   - All providers share identical interface

2. **Provider Adapter System**
   - Create adapter classes for each supported provider:
     - `OllamaAdapter` (local LLLaMA)
     - `OpenAIAdapter` (GPT models)
     - `GoogleAIAdapter` (Gemini)
     - `GroqAdapter` (free fast inference)
   - Adapters inherit from `LLMClient` and implement provider-specific HTTP calls
   - Factory pattern: `get_provider(provider_name: str) -> LLMClient`

3. **Configuration Management**
   - Use pydantic-settings for type-safe configuration
   - Support environment variables:
     - `LLM_PROVIDER` (primary provider)
     - `LLM_FALLBACK_PROVIDER` (fallback provider)
     - `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`
     - `OLLAMA_BASE_URL` (default: http://localhost:11434)
   - Configuration validation at startup

4. **Fallback Handling**
   - Implement automatic fallback when primary provider fails:
     - Network errors → fallback
     - Rate limiting → fallback
     - API errors (5xx) → fallback
     - Timeout → fallback
   - Circuit breaker pattern for repeated failures
   - Configurable fallback behavior (enabled/disabled)

5. **Request/Response Models**
   - Pydantic models for all request/response types:
     - `Message` (role: str, content: str)
     - `ChatRequest` (model: str, messages: list[Message], temperature: float, etc.)
     - `ChatResponse` (model: str, content: str, usage: Usage, finish_reason: str)
     - `Model` (id: str, name: str, provider: str, metadata: dict)
     - `Embedding` (index: int, embedding: list[float])

6. **Streaming Support**
   - Async generator for streaming responses
   - Proper token-by-token yield
   - Connection handling and cleanup

7. **Error Handling**
   - Custom exception hierarchy:
     - `LLMProviderError` (base)
     - `LLMAuthenticationError`
     - `LLMRateLimitError`
     - `LLMTimeoutError`
     - `LLMModelNotFoundError`
   - Provider-specific error mapping

8. **Retry Logic**
   - Exponential backoff for transient failures
   - Configurable retry count and delays
   - Jitter to prevent thundering herd

---

## 4. File Structure

```
llm_gateway/
├── __init__.py              # Package exports
├── VERSION                  # Version file
│
├── core/
│   ├── __init__.py
│   ├── base.py              # LLMClient abstract base class
│   ├── exceptions.py        # Custom exception hierarchy
│   ├── models.py           # Pydantic request/response models
│   └── factory.py          # Provider factory
│
├── providers/
│   ├── __init__.py
│   ├── base.py             # Base provider implementation
│   ├── ollama.py           # Ollama adapter
│   ├── openai.py           # OpenAI adapter
│   ├── google.py           # Google AI adapter
│   └── groq.py             # Groq adapter
│
├── config/
│   ├── __init__.py
│   └── settings.py        # Pydantic-settings configuration
│
├── middleware/
│   ├── __init__.py
│   ├── fallback.py       # Fallback handler
│   ├── retry.py           # Retry logic
│   └── circuit_breaker.py # Circuit breaker
│
├── utils/
│   ├── __init__.py
│   ├── logging.py        # Logging configuration
│   └── http.py           # HTTP client utilities
│
├── cli/
│   ├── __init__.py
│   └── main.py           # CLI entry point
│
└── tests/
    ├── __init__.py
    ├── conftest.py       # Pytest fixtures
    ├── unit/
    │   ├── __init__.py
    │   ├── test_providers/
    │   │   ├── __init__.py
    │   │   ├── test_ollama.py
    │   │   ├── test_openai.py
    │   │   └── test_google.py
    │   └── test_middleware/
    │       ├── __init__.py
    │       ├── test_fallback.py
    │       └── test_circuit_breaker.py
    └── integration/
        ├── __init__.py
        └── test_integration.py
```

---

## 5. Testing Requirements

### Testing Strategy

1. **Unit Tests (Required)**
   - Test each provider adapter in isolation
   - Mock all HTTP calls using `respx` or `unittest.mock`
   - Test request/response model serialization
   - Test exception handling paths
   - Minimum coverage target: **80%**

2. **Integration Tests (Required)**
   - Test with at least one free provider (Ollama local or Groq)
   - Test fallback behavior with simulated failures
   - Test configuration loading from environment

3. **Mocking Requirements**
   - Use `respx` for HTTP route mocking
   - Create deterministic test fixtures for provider responses
   - Simulate network errors, timeouts, rate limits

### Test Fixtures
```python
# conftest.py must include:
- mock_provider_responses: Fixtures for mocked API responses
- mock_env_vars: Fixture to override environment variables
- sample_messages: Standard test conversation
- mock_error_responses: Various error scenario mocks
```

### Running Tests
```bash
# All tests
pytest

# With coverage
pytest --cov=llm_gateway --cov-report=term-missing

# Specific provider
pytest tests/unit/test_providers/test_ollama.py

# Integration only
pytest tests/integration/
```

---

## 6. Git Protocol

### Commit Guidelines

1. **Commit Messages**
   - Format: