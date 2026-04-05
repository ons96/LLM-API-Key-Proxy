# LLM API Gateway System - AGENTS.md

## 1. Role/Mission

**Mission**: Build an autonomous LLM API Gateway that intelligently manages multiple LLM API providers, handles dynamic fallbacks, optimizes routing based on performance metrics, and provides reliable access to coding and chat models with automatic failover capabilities.

**Core Objectives**:
- Aggregate multiple LLM API providers (free and paid) into a unified gateway
- Implement intelligent routing for coding-smart, coding-fast, chat-smart, chat-fast model categories
- Enable dynamic provider discovery, signup automation, and configuration
- Ensure 99.9% availability through automatic failover and health checking
- Maintain high code quality through rigorous testing and validation

---

## 2. Technical Stack

**Primary Language**: Python 3.11+

**Framework & Libraries**:
- **FastAPI** - High-performance async API framework
- **httpx** - Async HTTP client for provider requests
- **aiohttp** - Additional async HTTP handling
- **pydantic** - Data validation and settings management
- **redis** - Caching and rate limit counters
- **SQLAlchemy** - Database ORM for configurations
- **tenacity** - Retry and backoff logic
- **structlog** - Structured logging
- **pytest** - Testing framework
- **pytest-asyncio** - Async test support

**Infrastructure**:
- **Docker** - Containerization
- **GitHub Actions** - CI/CD pipeline
- **Health check endpoints** - Provider monitoring

**External Services**:
- Multiple LLM API providers (OpenAI, Anthropic, Google, Ollama, LM Studio, Groq, Mistral, Cohere, etc.)
- Redis for distributed rate limiting
- PostgreSQL for configuration storage

---

## 3. Requirements

### 3.1 Core Gateway Infrastructure
1. Create a FastAPI-based gateway server with async request handling
2. Implement a unified `/v1/chat/completions` endpoint compatible with OpenAI API format
3. Support both streaming and non-streaming responses
4. Implement request/response middleware for logging and metrics

### 3.2 Provider Management
5. Create a provider abstraction layer with interface for adding new providers
6. Implement provider configuration system (API keys, endpoints, model mappings)
7. Build a provider health check system that monitors latency and availability
8. Implement automatic provider failover when a provider fails or exceeds rate limits

### 3.3 Model Routing & Categorization
9. Define and implement model categories:
   - `coding-smart`: Best coding models (e.g., Claude Opus, GPT-4)
   - `coding-fast`: Fast coding models (e.g., Claude Haiku, GPT-3.5)
   - `chat-smart`: Best conversational models
   - `chat-fast`: Fast conversational models
10. Implement dynamic model ordering based on:
    - Historical response times (latency)
    - Success rate
    - Quality scores (if available)
    - Cost efficiency

### 3.4 Rate Limiting & Load Balancing
11. Implement per-provider rate limit tracking and management
12. Create load balancing strategies (round-robin, least-latency, weighted)
13. Implement request queuing with timeout handling
14. Add circuit breaker pattern for failing providers

### 3.5 Dynamic Provider Discovery (Stretch Goal)
15. Research free LLM API providers and document discovery process
16. Create automation scripts for provider signup if APIs are available
17. Implement configuration auto-update mechanism for new providers

### 3.6 Quality Assurance
18. Implement comprehensive unit tests for all core components
19. Create integration tests that actually call provider APIs (with mocks for paid services)
20. Build a validation script that tests the gateway end-to-end
21. Document all bugs found during testing in a tracking system

---

## 4. File Structure

```
llm-gateway/
├── agents.md                    # This file
├── QUESTIONS.md                 # Questions for human review
├── README.md                    # Project documentation
├── pyproject.toml               # Python project configuration
├── poetry.lock                 # Dependency lock file
├── .env.example                # Environment variables template
├── docker-compose.yml          # Local development setup
├── Dockerfile                  # Container build
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
├── src/
│   └── llm_gateway/
│       ├── __init__.py
│       ├── main.py             # FastAPI application entry point
│       ├── config.py           # Configuration management
│       ├── models/
│       │   ├── __init__.py
│       │   ├── requests.py     # Request models (Pydantic)
│       │   └── responses.py    # Response models (Pydantic)
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py         # Base provider interface
│       │   ├── openai.py       # OpenAI provider implementation
│       │   ├── anthropic.py    # Anthropic provider implementation
│       │   ├── ollama.py       # Ollama local provider
│       │   ├── groq.py         # Groq provider
│       │   ├── lmstudio.py     # LM Studio local provider
│       │   └── registry.py     # Provider registry and factory
│       ├── router/
│       │   ├── __init__.py
│       │   ├── routing.py      # Core routing logic
│       │   ├── categories.py   # Model categorization
│       │   └── metrics.py     # Performance metrics collection
│       ├── failover/
│       │   ├── __init__.py
│       │   ├── circuit_breaker.py  # Circuit breaker implementation
│       │   └── health_check.py     # Health monitoring
│       ├── rate_limiter/
│       │   ├── __init__.py
│       │   └── limiter.py     # Rate limiting logic
│       ├── gateway/
│       │   ├── __init__.py
│       │   ├── chat.py         # Chat completions handler
│       │   └── middleware.py  # Request/response middleware
│       └── utils/
│           ├── __init__.py
│           ├── logger.py      # Structured logging setup
│           └── validators.py  # Input validation helpers
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Pytest fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_providers/
│   │   │   ├── __init__.py
│   │   │   └── test_base_provider.py
│   │   ├── test_router/
│   │   │   ├── __init__.py
│   │   │   └── test_routing.py
│   │   ├── test_failover/
│   │   │   ├── __init__.py
│   │   │   └── test_circuit_breaker.py
│   │   └── test_rate_limiter/
│   │       ├── __init__.py
│   │       └── test_limiter.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_gateway.py    # End-to-end gateway tests
│   │   └── test_providers.py  # Provider integration tests
│   └── fixtures/
│       ├── mock_responses.json
│       └── provider_configs.yaml
├── scripts/
│   ├── setup_dev.py           # Development environment setup
│   ├── validate_gateway.py    # Gateway validation script
│   ├── discover_providers.py  # Provider discovery (stretch)
│   └── test_routing.py        # Routing test script
└── docs/
    ├── architecture.md        # System architecture
    ├── provider_setup.md      # Provider configuration guide
    └── api_reference.md       # API documentation
```

---

## 5. Testing Requirements

### 5.1 Unit Tests (Required)
- All provider implementations must have >80% code coverage
- Test all routing logic with mocked providers
- Test circuit breaker states (closed, open, half-open)
- Test rate limiter boundaries and edge cases
- Test model categorization logic

### 5.2 Integration Tests (Required)
- Test gateway with at least one live free provider (Ollama/LM Studio)
- Test failover behavior by simulating provider failures
- Test streaming responses end-to-end
- Test rate limit handling

### 5.3 Validation Requirements (Critical)
**YOU MUST RUN THESE VALIDATIONS BEFORE CONSIDERING ANY TASK COMPLETE:**

1. **Syntax Validation**: All Python files must pass `python -m py_compile`
2. **Import Validation**: All modules must import without errors
3. **Type Checking**: Run `mypy src/` if type hints are added
4. **Linting**: Pass `ruff check src/` with no errors
5. **Gateway Startup**: The FastAPI app must start without errors
6. **Endpoint Tests**: Test the `/health` and `/v1/chat/completions` endpoints
7. **Failover Test**: Verify failover works by testing with mock failures
8. **Documentation**: Ensure all public APIs have docstrings

### 5.4 Test Execution Commands
```bash
# Run all tests
pytest tests/ -v --tb=short

# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests (requires live providers)
pytest tests/integration/ -v

# Run validation script
python scripts/validate_gateway.py

# Check code quality
ruff check src/
mypy src/ --ignore-missing-imports
```

---

## 6. Git Protocol

### 6.1 Branch Strategy
- `main` - Stable, production-ready code
- `develop` - Integration branch for features
- `feature/*` - Feature branches (e.g., `feature/provider-failover`)
- `bugfix/*` - Bug fix branches (e.g., `bugfix/rate-limit-fix`)

### 6.2 Commit Messages
Use conventional commits format:
```
<type>(<scope>): <description>

Types: feat, fix, docs, test, refactor, chore
Examples:
  feat(provider): add Groq provider support
  fix(router): handle empty provider list gracefully
  test(failover): add circuit breaker state tests
```

### 6.3 Pull Request Requirements
- All tests must pass
- Code must pass linting and type checking
- At least one reviewer approval required
- All CI checks must pass
- Update documentation if needed

### 6.4 CI Pipeline (GitHub Actions)
The CI pipeline must:
1. Run linting (ruff)
2. Run type checking (mypy)
3. Run unit tests with coverage
4. Run integration tests (if credentials available)
5. Build Docker image

---

## 7. Completion Criteria

### 7.1 Functional Requirements
- [ ] Gateway server starts and accepts requests
- [ ] `/v1/chat/completions` endpoint works with at least 2 providers
- [ ] Streaming responses work correctly
- [ ] Provider failover triggers on provider failure
- [ ] Rate limiting prevents provider overuse
- [ ] Model routing selects appropriate provider based on category
- [ ] Health check endpoint returns provider statuses

### 7.2 Code Quality
- [ ] All code passes linting (ruff)
- [ ] All code passes type checking (mypy)
- [ ] Unit test coverage > 80%
- [ ] No critical or high severity bugs in code
- [ ] All public APIs documented

### 7.3 Validation Checklist
Before marking any task complete, verify:
1. `python -m py_compile src/llm_gateway/main.py` - No syntax errors
2. `python -c "from llm_gateway import main"` - No import errors
3. `pytest tests/unit/ -v` - All unit tests pass
4. Gateway starts: `uvicorn llm_gateway.main:app --host 0.0.0.0 --port 8000`
5. Health check: `curl http://localhost:8000/health`
6. Chat completion test with mock or live provider

### 7.4 Documentation
- [ ] README.md with setup instructions
- [ ] API documentation for endpoints
- [ ] Provider configuration guide
- [ ] Architecture documentation

### 7.5 Known Limitations to Document
If you cannot complete a requirement, document it in QUESTIONS.md with:
- What you attempted
- Why it failed
- Suggested approach for future work
- Resources that might help

---

## Important Notes for Autonomous Agent

1. **Use Free Resources Only**: Do not require paid API keys for core functionality. Use Ollama, LM Studio, or other free/local providers for testing.

2. **Test Reality, Not Mocks**: While mocks are useful for unit tests, you MUST validate with real providers where possible. Do not assume code works because tests pass - verify actual functionality.

3. **Fix Bugs Found**: If testing reveals bugs, you MUST fix them before completion. Do not leave known issues.

4. **Ask Questions**: If you encounter blockers, unclear requirements, or need clarification, create a QUESTION.md file with your specific questions rather than making assumptions.

5. **Validate End-to-End**: The final validation must demonstrate the gateway working with actual LLM providers, not just unit tests passing.

6. **No Fake Work**: Do not claim completion if the gateway doesn't actually route requests to providers and return responses. Show actual test output proving it works.