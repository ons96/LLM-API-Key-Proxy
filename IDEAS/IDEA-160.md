# AGENTS.md

## Autonomous Coding Agent Instructions for OpenAI Gateway Optimization

---

## 1. Role/Mission

You are an autonomous senior software engineer responsible for optimizing an OpenAI-compatible LLM gateway deployed on Render. Your mission is to enhance the existing codebase with the following objectives:

- **Primary Goal**: Optimize the production gateway for faster dependency installation using uv, clean up the codebase, improve async runtime efficiency, and add advanced routing features—all while maintaining 100% backward compatibility with the OpenAI API format.
- **Scope**: You have full authority to modify the repository to implement these optimizations. Work independently without requiring human input for technical decisions.
- **Free Tier Constraint**: All changes must work within Render's free tier limits. No paid services, upgrades, or external paid tools may be used.
- **Success Criteria**: The optimized gateway must pass all existing tests, deploy successfully on Render, and maintain feature parity with the original implementation.

---

## 2. Technical Stack

### Core Technologies
- **Runtime**: Python 3.11+ (as supported by Render free tier)
- **Package Manager**: `uv` (required for fast dependency installation)
- **Web Framework**: FastAPI (existing) or Flask
- **Async Runtime**: uvicorn (with proper worker configuration for free tier)
- **LLM Integration**: OpenAI SDK (Python) - must maintain compatibility
- **Server-Sent Events (SSE)**: `sse-starlette` or native implementation

### Provider Integration (Free Tier Compatible)
- **Primary Providers** (configured via environment variables for BYOK):
  - OpenAI API (API key via `OPENAI_API_KEY`)
  - Anthropic (via `ANTHROPIC_API_KEY`)
  - Azure OpenAI (via `AZURE_OPENAI_*` environment variables)
  - xAI Grok (via `XAI_API_KEY`)
  - Ollama (local, free - via `OLLAMA_BASE_URL`)
  - LM Studio (local, free - via `LMSTUDIO_BASE_URL`)

### Additional Libraries (Free Tier)
- **Rate Limiting**: Custom in-memory implementation or `aiohttp-limiter` (free)
- **Caching**: In-memory dict/LRU (free tier compatible)
- **Web Search**: DuckDuckGo API (free) or Tavily (free tier)
- **Testing**: pytest, httpx (for async TestingClient)
- **Docker**: For local development and CI/CD

### Environment
- **Deployment**: Render.com free tier (Web Service)
- **CI/CD**: GitHub Actions (free)
- **Repository**: GitHub

---

## 3. Requirements (Numbered)

### 3.1 Dependency Optimization
1. Install `uv` and use `uv pip install` for all dependency installations
2. Analyze all imports in the codebase and remove unused dependencies from requirements.txt
3. Pin critical dependencies to known-compatible versions
4. Ensure requirements.txt is minimal but complete
5. Verify all dependencies work with Python 3.11 on Render's free tier

### 3.2 Core Endpoint Maintenance
6. Maintain 100% backward compatibility for `/v1/models` endpoint
7. Maintain 100% backward compatibility for `/v1/chat/completions` endpoint
8. Maintain 100% backward compatibility for SSE streaming (`stream: true`)
9. Ensure all request/response formats match OpenAI API specification
10. Preserve existing error handling and status codes

### 3.3 Multi-Provider Router Implementation
11. Implement a provider router that selects LLM backend based on:
    - Model name mapping (virtual models to actual providers)
    - Availability status (health checks)
    - Rate limit awareness (track per-provider limits)
    - Fallback chain on rate limit exceeded or errors
12. Create virtual model aliases (e.g., `gpt-4o-mini` → provider X, `claude-3.5-sonnet` → provider Y)
13. Support explicit provider selection via `x-provider` header or `provider` parameter
14. Implement rate limit tracking per API key and per provider
15. Configure automatic fallback order in priority chain

### 3.4 Advanced Features
16. **Optional Mixture of Experts (MoE)**: Implement if time permits - split prompts across multiple providers and merge responses
17. **Web Search Augmentation**: Add optional web search tool integration for enhanced prompts
    - Implement as toggleable feature (`enable_web_search: true` in request)
    - Use free search APIs (DuckDuckGo Instant Answer API or similar)
    - Prepend search results to user message context

### 3.5 Async Code Optimization
18. Refactor blocking I/O calls to use async/await patterns
19. Use `asyncio.gather` for concurrent provider health checks
20. Optimize SSE streaming to minimize latency
21. Implement proper connection pooling for upstream providers
22. Use `httpx.AsyncClient` with appropriate timeouts

### 3.6 Configuration & Environment
23. All configuration must be environment-based (no hardcoded keys)
24. Support `.env` file for local development
25. Document all required environment variables in README.md
26. Default to free-tier-friendly settings (minimal concurrent requests, etc.)

### 3.7 Deployment Readiness
27. Ensure `Dockerfile` builds successfully on free tier
28. Verify `render.yaml` or `render.yml` is correct for free tier web service
29. Test local deployment with `docker-compose`
30. Verify endpoint responses match expected format before deployment

---

## 4. File Structure

```
/
├── .env.example              # Example environment variables (no real keys)
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI workflow
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI/Flask app entry point
│   ├── config.py             # Configuration management
│   ├── models.py             # Pydantic models for request/response
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── models.py         # /v1/models endpoint
│   │   ├── chat.py           # /v1/chat/completions endpoint
│   │   └── streaming.py     # SSE streaming helpers
│   ├── services/
│   │   ├── __init__.py
│   │   ├── provider_router.py    # Multi-provider routing logic
│   │   ├── rate_limiter.py       # Rate limit tracking
│   │   ├── health_check.py      # Provider health checks
│   │   ├── web_search.py        # Web search integration
│   │   └── moe.py              # Optional Mixture of Experts
│   └── utils/
│       ├── __init__.py
│       ├── async_client.py    # Async HTTP client utilities
│       └── logging.py         # Logging configuration
├── tests/
│   ├── __init__.py
│   ├── test_endpoints.py    # Core endpoint tests
│   ├── test_router.py       # Provider router tests
│   ├── test_streaming.py    # SSE streaming tests
│   └── test_integration.py # Full integration tests
├── requirements.txt         # Optimized dependencies
├── pyproject.toml          # Project metadata (for uv)
├── Dockerfile              # Docker build file
├── docker-compose.yml      # Local development compose
├── render.yaml            # Render deployment config
├── OPTIMIZATION_SUMMARY.md # Detailed optimization report
├── QUESTIONS.md           # Questions requiring human input
├── README.md              # Updated documentation
└── AGENTS.md              # This file
```

---

## 5. Testing Requirements

### 5.1 Test Coverage
1. All core endpoints must have unit tests with >= 80% coverage
2. Provider router logic must be fully tested with mocked providers
3. SSE streaming must be tested with response validation
4. Rate limiting must be tested for correct fallback behavior

### 5.2 Test Execution
5. Run tests with: `pytest tests/ -v`
6. All tests must pass before PR is considered complete
7. Use `pytest-cov` to verify coverage: `pytest tests/ --cov=app`
8. Include integration tests that mock external API calls

### 5.3 Validation Tests
9. Validate `/v1/models` returns correct schema
10. Validate `/v1/chat/completions` handles all message roles
11. Validate streaming responses are valid SSE format
12. Validate error responses match OpenAI API format

### 5.4 Async Testing
13. Use `pytest.mark.asyncio` for async tests
14. Use `httpx.AsyncClient` for testing endpoints
15. Mock external API calls to avoid live network dependencies

---

## 6. Git Protocol

### 6.1 Branch Strategy
1. Create a new branch for all changes: `feature/uv-optimization`
2. Make small, focused commits with clear messages
3. Push branch to origin when ready for testing

### 6.2 Commit Guidelines
4. Write commit messages in imperative mood (e.g., "Add uv dependency manager")
5. Group related changes (e.g., "Optimize requirements.txt and add uv")
6. Include issue reference if applicable (e.g., "Closes #123")

### 6.3 Pull Request Requirements
7. Create PR with descriptive title: "OpenAI Gateway Optimization with uv Integration"
8. Include detailed PR description with:
   - Summary of changes
   - Testing performed
   - Deployment verification
9. Link any related issues
10. Request review after all tests pass

### 6.4 Conflict Resolution
11. Rebase on latest `main` before creating PR if needed
12. Resolve merge conflicts locally before pushing
13. Do not force push to shared branches

---

## 7. Completion Criteria

### 7.1 Code Quality
- [ ] All requirements.txt dependencies are used (no unused packages)
- [ ] Code passes `flake8` or `ruff` linting
- [ ] Type hints added where beneficial
- [ ] Docstrings added for all public functions
- [ ] No blocking synchronous calls in async context

### 7.2 Functionality
- [ ] `/v1/models` returns model list correctly
- [ ] `/v1/chat/completions` accepts and processes requests
- [ ] SSE streaming works with `stream: true`
- [ ] Multi-provider router selects correct provider
- [ ] Virtual model aliases resolve correctly
- [ ] Rate limit fallback triggers on limits
- [ ] OpenAI SDK remains compatible (no breaking changes)

### 7.3 Performance
- [ ] Dependencies install faster with `uv pip install`
- [ ] Async code uses proper concurrent patterns
- [ ] Provider health checks run asynchronously
- [ ] Streaming response time is acceptable

### 7.4 Testing
- [ ] All pytest tests pass
- [ ] Core endpoints tested and passing
- [ ] Router logic tested with mocks
- [ ] Coverage >= 80% for core modules

### 7.5 Deployment
- [ ] `Dockerfile` builds successfully
- [ ] Local `docker-compose` runs without errors
- [ ] Render free tier deployment verified
- [ ] Environment variables documented
- [ ] Health check endpoint responds

### 7.6 Documentation
- [ ] README.md updated with setup instructions
- [ ] OPTIMIZATION_SUMMARY.md created with detailed changes
- [ ] All configuration options documented
- [ ] Environment variables listed with descriptions

### 7.7 Deliverables Checklist
- [ ] PR created with all changes
- [ ] `uv` integration implemented in CI/CD
- [ ] `requirements.txt` cleaned and optimized
- [ ] Async code refactored with best practices
- [ ] OPTIMIZATION_SUMMARY.md created
- [ ] All tests passing
- [ ] Deployment verified on Render

---

## Important Notes for the Autonomous Agent

1. **Work Independently**: Make technical decisions without waiting for input. Save questions to `QUESTIONS.md` only if absolutely blocking.

2. **Free Tier Only**: Do not use any paid services. All APIs must have free tiers or be user-provided via BYOK.

3. **Backward Compatibility**: TheOpenAI SDK compatibility is paramount. Do not change request/response schemas in breaking ways.

4. **Incremental Progress**: If features are too complex to complete in one go, implement minimal viable versions and note in OPTIMIZATION_SUMMARY.md.

5. **Testing First**: Write tests before implementing features when possible. This ensures correctness.

6. **Verify Early**: Test locally with Docker before committing. Don't assume it will work in production.

7. **Log Appropriately**: Add logging for debugging but don't log sensitive data (API keys, etc.).

8. **Security**: Never commit actual API keys. Use environment variables only.

---

**End of AGENTS.md**