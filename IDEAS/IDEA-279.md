# AGENTS.md - LLM Fallback and Rotation System

## 1. Role/Mission

**Mission:** Build a robust LLM Fallback and Rotation System that automatically manages API calls across multiple LLM providers, ensuring high availability by seamlessly rotating through fallback models when primary providers fail.

**Primary Objectives:**
- Maintain a configurable priority list of LLM providers and models
- Automatically detect and handle failures including rate limits, provider downtime, invalid API keys, and malformed responses
- Implement intelligent retry mechanisms with exponential backoff
- Track provider health to optimize rotation decisions
- Provide a unified interface for making LLM calls regardless of the underlying provider

## 2. Technical Stack

**Language:** Python 3.10+ (free, cross-platform)

**Core Dependencies:**
- `openai` - OpenAI API client (free tier available)
- `anthropic` - Anthropic Claude API client (free tier available)
- `requests` - HTTP library for API calls
- `tenacity` - Retry logic library (free)
- `pydantic` - Data validation (free)
- `python-dotenv` - Environment variable management (free)
- `pytest` - Testing framework (free)
- `pytest-asyncio` - Async testing (free)
- `build` - Package building (free)

**Optional Local Provider:**
- `ollama` - Local LLM inference (free, self-hosted)

**Infrastructure:**
- GitHub Actions for CI/CD (free for public repos)
- GitHub Secrets for API key management (free)

## 3. Requirements

### 3.1 Provider Management
1. Create a `ProviderConfig` class to define provider settings (name, API base URL, API key env var, models, priority)
2. Implement a `ProviderRegistry` to manage a prioritized list of providers
3. Support at minimum: OpenAI, Anthropic (Claude), and Ollama (local) providers
4. Allow providers to be enabled/disabled via configuration
5. Store provider priority order in a YAML configuration file

### 3.2 Error Detection
6. Implement `LLMError` base exception class
7. Create specific exception types: `RateLimitError`, `ProviderDownError`, `InvalidKeyError`, `MalformedResponseError`, `TimeoutError`
8. Build error detection logic that parses provider error responses to determine error type
9. Handle both HTTP status codes and response body analysis for error identification

### 3.3 Retry Mechanism
10. Implement a retry decorator/function using exponential backoff
11. Configure maximum retry attempts per request (default: 3)
12. Configure base delay and max delay for exponential backoff (default: 1s base, 60s max)
13. Add jitter to backoff calculations to prevent thundering herd
14. Only retry on transient errors (rate limits, timeouts, provider downtime); do not retry onInvalidKeyError

### 3.4 Fallback Rotation
15. Implement a `LLMClient` class that manages the fallback rotation
16. On failure, automatically attempt the next provider in the priority list
17. Track which providers have been attempted for the current request
18. Return the successful response from any provider in the fallback chain

### 3.5 Health Tracking
19. Implement `ProviderHealth` class to track provider metrics (success rate, average latency, consecutive failures)
20. Implement health-based provider filtering (disable providers with poor health)
21. Track success/failure per-request and maintain rolling statistics
22. Log provider health metrics for monitoring

### 3.6 Unified Interface
23. Create a `.chat_complete()` method that accepts: messages, model (optional), temperature, max_tokens
24. Support both sync and async interfaces
25. Return responses in a standardized format regardless of provider
26. Include provider information in the response metadata

### 3.7 Configuration
27. Load configuration from `config.yaml` file
28. Support environment variable override for API keys
29. Allow customization of retry settings, timeouts, and provider priorities via config

### 3.8 Logging
30. Implement structured logging with provider context
31. Log all retry attempts with backoff details
32. Log provider health changes

## 4. File Structure

```
llm-fallback-system/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI workflow
├── config/
│   └── providers.yaml           # Provider configuration
├── src/
│   └── llm_fallback/
│       ├── __init__.py
│       ├── client.py           # Main LLMClient class
│       ├── config.py           # Configuration loading
│       ├── exceptions.py        # Custom exception classes
│       ├── health.py            # Provider health tracking
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py         # Base provider class
│       │   ├── openai.py       # OpenAI provider
│       │   ├── anthropic.py    # Anthropic provider
│       │   └── ollama.py        # Ollama provider
│       ├── registry.py          # Provider registry
│       └── retry.py             # Retry logic
├── tests/
│   ├── __init__.py
│   ├── test_exceptions.py
│   ├── test_health.py
│   ├── test_providers/
│   │   ├── __init__.py
│   │   ├── test_openai.py
│   │   └── test_anthropic.py
│   ├── test_client.py
│   ├── test_retries.py
│   └── fixtures/
│       └── mock_responses.py   # Mock response data
├── pyproject.toml              # Project configuration
├── README.md
├── QUESTIONS.md                # Autonomous agent questions
└── .env.example               # Example environment variables
```

## 5. Testing Requirements

### 5.1 Test Coverage Goals
- Minimum 80% code coverage required
- All public methods must have unit tests
- Integration tests for provider clients (can use mocks)

### 5.2 Unit Tests
1. Test exception classes correctly identify error types
2. Test provider configuration loading from YAML
3. Test registry returns providers in priority order
4. Test exponential backoff calculation accuracy
5. Test health tracking updates on success/failure
6. Test fallback rotation cycles through all providers

### 5.3 Mock Tests
7. Mock provider responses to test error detection
8. Mock rate limit responses to test retry logic
9. Mock provider downtime to test fallback behavior

### 5.4 Integration Tests (Optional)
10. Test against mock API servers (e.g., using `responses` library)
11. Test configuration file parsing
12. Test environment variable loading

### 5.5 Test Execution
- Run tests with: `pytest -v --cov=src --cov-report=html`
- All tests must pass before merging to main
- Run linting: `ruff check src tests`

## 6. Git Protocol

### 6.1 Branch Strategy
- `main` - Production-ready code (protected)
- `develop` - Integration branch (protected)
- Feature branches: `feature