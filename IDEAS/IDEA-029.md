# AGENTS.md - All-in-One LLM Router with Fallbacks

## 1. Role/Mission

**Mission:** Build an intelligent LLM (Large Language Model) router system that automatically routes requests to available LLM providers and gracefully falls back to alternative providers when failures occur.

**Core Objectives:**
- Create a unified API interface that abstracts multiple LLM providers
- Implement automatic health monitoring of all configured providers
- Build robust fallback logic that switches providers on failure without disrupting user experience
- Add intelligent retry mechanisms with exponential backoff
- Optimize for cost and reliability by preferring healthy, cost-effective providers

**Autonomous Agent Guidelines:**
- Make all technical decisions independently without asking for confirmation
- Use free/available resources only - do not purchase any services
- If you encounter blockers, document questions in `QUESTIONS.md` and proceed with the best available alternative
- Prioritize working, testable code over perfect code

---

## 2. Technical Stack

**Core Language:** Python 3.10+

**Key Dependencies:**
- `httpx` - Async HTTP client for provider requests
- `pydantic` - Data validation and settings management
- `python-dotenv` - Environment variable management
- `tenacity` - Retry logic with exponential backoff
- `aiohttp` - Alternative async operations
- `pytest` + `pytest-asyncio` - Testing framework
- `pytest-mock` - Mocking for tests

**LLM Providers (supported interfaces):**
- OpenAI (GPT models)
- Anthropic (Claude models)
- Google Gemini
- Mistral AI
- Local/custom endpoints

**Infrastructure:**
- GitHub Actions for CI/CD
- Environment变量 for API key management
- Local file-based health state (for free operation without external databases)

---

## 3. Requirements

### 3.1 Core Router Implementation

1. **Unified Provider Interface**
   - Create abstract base class `LLMProvider` defining the contract
   - Implement `chat_completion()` method with consistent signature across all providers
   - Standardize response format using Pydantic models

2. **Multi-Provider Configuration**
   - Support adding multiple providers via configuration
   - Allow priority/ranking of providers (preferred order)
   - Support per-provider model selection and parameters

3. **Health Check System**
   - Implement lightweight health check for each provider
   - Store health status in memory or file-based cache
   - Configurable health check interval (default: 60 seconds)
   - Mark providers as unhealthy after consecutive failures

4. **Fallback Logic**
   - On provider failure, automatically attempt next available provider
   - Preserve conversation context when switching providers
   - Log all fallback events for debugging
   - Maximum fallback attempts (configurable, default: 3)

5. **Auto-Retry with Backoff**
   - Implement retry logic for transient failures
   - Use exponential backoff (default: 1s, 2s, 4s)
   - Configurable retry count (default: 2)
   - Distinguish retryable vs. non-retryable errors

### 3.2 Request/Response Handling

6. **Unified Request Format**
   - Support standard OpenAI-compatible chat format
   - Allow provider-specific parameters passthrough
   - Handle streaming responses (where supported)

7. **Response Normalization**
   - Convert provider-specific responses to unified format
   - Include metadata (provider used, latency, tokens)

8. **Error Handling**
   - Create custom exception classes
   - Include provider-specific error messages
   - Rate limiting detection and handling

### 3.3 System Features

9. **Configuration Management**
   - Use environment variables for API keys
   - Support YAML config file for provider settings
   - Allow runtime reconfiguration

10. **Logging & Observability**
    - Structured logging with levels
    - Log provider selection and fallbacks
    - Track latency and success rates
    - Performance metrics collection

11. **Circuit Breaker Pattern**
    - Track failure rates per provider
    - Temporarily disable providers exceeding failure threshold
    - Auto-re-enable after recovery period

---

## 4. File Structure

```
llm-router/
├── AGENTS.md                    # This file
├── QUESTIONS.md                 # Questions for future investigation
├── README.md                     # Project overview
├── pyproject.toml               # Python project config
├── .env.example                 # Example environment variables
├── .gitignore                   # Git ignore rules
│
├── src/
│   └── llm_router/
│       ├── __init__.py           # Package init
│       ├── main.py               # Main router class
│       ├── types.py              # Type definitions
│       ├── exceptions.py         # Custom exceptions
│       ├── config.py             # Configuration management
│       ├── metrics.py            # Metrics collection
│       │
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py            # Abstract base provider
│       │   ├── openai.py          # OpenAI provider
│       │   ├── anthropic.py       # Anthropic provider
│       │   ├── google.py          # Google Gemini provider
│       │   ├── mistral.py         # Mistral AI provider
│       │   └── mock.py            # Mock provider for testing
│       │
│       ├── health/
│       │   ├── __init__.py
│       │   ├── checker.py         # Health check implementation
│       │   └── state.py           # Health state management
│       │
│       └── router/
│           ├── __init__.py
│           ├── selector.py        # Provider selection logic
│           ├── fallback.py        # Fallback orchestration
│           └── retry.py           # Retry logic
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # Pytest fixtures
│   ├── test_providers/
│   │   ├── __init__.py
│   │   ├── test_base.py
│   │   ├── test_openai.py
│   │   └── test_mock.py
│   │
│   ├── test_health/
│   │   ├── __init__.py
│   │   └── test_checker.py
│   │
│   ├── test_router/
│   │   ├── __init__.py
│   │   ├── test_selector.py
│   │   ├── test_fallback.py
│   │   └── test_integration.py
│   │
│   └── test_utils/
│       ├── __init__.py
│       └── fixtures.py
│
├── scripts/
│   ├── test_auth.sh               # Test authentication
│   └── benchmark.py              # Benchmark script
│
└── examples/
    └── basic_usage.py            # Basic usage examples
```

---

## 5. Testing Requirements

### 5.1 Test Strategy

**Mock-Based Testing (Primary):**
- Use mock providers for all unit tests to avoid API costs
- Mock provider responses must simulate real provider behavior
- Test both success and failure scenarios

**Test Coverage Goals:**
- Minimum 80% code coverage required
- All public methods must have corresponding tests
- Edge cases and error conditions must be tested

### 5.2 Required Tests

1. **Provider Tests**
   - [ ] Test each provider's `chat_completion()` method
   - [ ] Test provider-specific error handling
   - [ ] Test request formatting for each provider

2. **Health Check Tests**
   - [ ] Test health check execution
   - [ ] Test health state updates
   - [ ] Test unhealthy provider detection

3. **Router Tests**
   - [ ] Test successful provider selection
   - [ ] Test fallback on provider failure
   - [ ] Test max fallback limit enforcement
   - [ ] Test context preservation during fallback

4. **Retry Tests**
   - [ ] Test retry on transient failure
   - [ ] Test exponential backoff timing
   - [ ] Test non-retryable error handling

5. **Integration Tests**
   - [ ] Full end-to-end router flow
   - [ ] Multiple sequential fallbacks
   - [ ] Concurrent request handling

### 5.3 Test Execution

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_providers/
pytest tests/test_router/
pytest tests/test_integration.py

# Run with verbose output
pytest -v --tb=short
```

---

## 6. Git Protocol

### 6.1 Branch Strategy

- **main** - Main branch, always deployable
- **develop** - Integration branch for new features
- **feature/* - Feature branches for new functionality

### 6.2 Commit Messages

Use conventional commits:
```
feat: Add OpenAI provider support
fix: Correct retry backoff timing
docs: Update README with examples
test: Add provider fallback tests
refactor: Improve health check efficiency
```

### 6.3 Pull Request Workflow

1. Create feature branch from `develop`
2. Make changes with clear, focused commits
3. Ensure all tests pass
4. Update documentation if needed
5. Create PR to `develop`
6. Request review (self-review if solo)

### 6.4 Handling Questions/Blockers

If you encounter questions or blockers:
1. Document in `QUESTIONS.md` with:
   - The question or issue
   - What you tried
   - Suggested resolution
2. Proceed with the best alternative if possible
3. Continue working on other independent tasks

---

## 7. Completion Criteria

### 7.1 Functional Requirements

- [ ] **Unified Router API**: Single interface to access all configured providers
- [ ] **Multiple Provider Support**: At least 3 providers implemented (OpenAI, Anthropic, Mock)
- [ ] **Health Monitoring**: Automatic health checks running at configured intervals
- [ ] **Automatic Fallbacks**: Seamless fallback to next provider on failure
- [ ] **Retry Logic**: Automatic retry with exponential backoff for transient failures
- [ ] **Error Handling**: Proper error propagation with contextual information
- [ ] **Configuration**: Environment-based configuration for API keys and settings

### 7.2 Code Quality Requirements

- [ ] Minimum 80% test coverage
- [ ] All tests passing (`pytest` passes)
- [ ] No linting errors (`ruff` or `flake8`)
- [ ] Proper docstrings for all public classes and methods
- [ ] Type hints for all function signatures

### 7.3 Deliverables

- [ ] Source code in `src/llm_router/` directory
- [ ] Test suite in `tests/` directory
- [ ] Working examples in `examples/` directory
- [ ] Configuration example in `.env.example`
- [ ] README with usage instructions
- [ ] This AGENTS.md completed

### 7.4 Verification Commands

```bash
# Code compiles without errors
python -c "from llm_router import LLMRouter"

# All tests pass
pytest -v

# Tests meet coverage threshold
pytest --cov=src --cov-report=term-missing

# Package installs correctly
pip install -e .
```

---

## Important Notes for Autonomous Agent

1. **Use Mock Providers**: Do not attempt to configure real API keys - use the mock provider for testing and demonstration

2. **Document as You Go**: Keep `QUESTIONS.md` updated with any questions that arise

3. **Incremental Development**: Implement features incrementally - get working code first, then refine

4. **Free Resources Only**: Do not purchase any services - use free tiers, local resources, or mocks

5. **Default to Best Practice**: When facing decisions, choose the approach that is most standard, maintainable, and testable

6. **Completion Over Perfection**: A working, tested implementation is better than an incomplete perfect one

---

*Last Updated: 2024*
*This AGENTS.md serves as the complete specification for the autonomous agent.*