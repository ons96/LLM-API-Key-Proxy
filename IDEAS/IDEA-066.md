# AGENTS.md - LLM API Gateway with Provider Rotation

## 1. Role/Mission

You are an autonomous coding agent tasked with building an LLM API Gateway that serves as a unified interface for routing requests across multiple free LLM providers. Your mission is to create a robust proxy system that automatically manages provider rotation based on rate limits, tracks usage metrics, handles authentication, and provides high availability by seamlessly switching between providers when one becomes unavailable or rate-limited.

**Primary Objectives:**
- Build a unified API gateway that abstracts away differences between multiple LLM providers
- Implement intelligent provider rotation with automatic failover
- Track usage limits (active days, token counts, API calls) per provider
- Handle authentication and API key management securely
- Make the system resilient to rate limits and provider outages

## 2. Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.10+ | Core implementation |
| HTTP Server | FastAPI | API gateway service |
| LLM Providers | g4f, litellm, mirrorel | Free LLM provider integration |
| Routing | Custom Round-Robin/Rotation | Provider load balancing |
| Authentication | JWT tokens + API keys | Secure access control |
| Data Storage | SQLite (for simplicity) | Usage tracking database |
| Testing | pytest, httpx | API testing |
| Deployment | Docker | Containerization |

### Key Dependencies
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `g4f` - Free GPT API wrapper
- `litellm` - Unified LLM API
- `mirrorel` - Alternative LLM provider
- `pydantic` - Data validation
- `python-jose` - JWT handling
- `passlib` - Password hashing
- `aiosqlite` - Async SQLite

## 3. Requirements

### 3.1 Core Functionality

1. **Unified API Endpoint**
   - Create a single REST API endpoint (`/v1/chat/completions`) that accepts standard OpenAI-compatible requests
   - Support streaming and non-streaming responses
   - Maintain compatibility with OpenAI API format for easy integration

2. **Provider Rotation Engine**
   - Implement a configurable provider pool with priority-based selection
   - Track provider health status and maintain a "healthy" provider list
   - Automatically rotate to the next available provider on rate limit errors
   - Implement retry logic with exponential backoff

3. **Usage Tracking**
   - Track daily active users per provider
   - Maintain token count metrics per request and cumulative totals
   - Record API call counts with timestamps
   - Store usage data in SQLite with daily aggregation
   - Expose `/usage` endpoint for clients to check remaining limits

4. **Authentication System**
   - Implement API key-based authentication
   - Support JWT token generation and validation
   - Allow rate limiting per API key
   - Include key rotation and expiration capabilities

5. **Provider Management**
   - Support dynamic enable/disable of providers
   - Configure per-provider rate limits
   - Track provider-specific configurations
   - Handle provider-specific request formats

### 3.2 Error Handling

6. **Graceful Degradation**
   - Implement circuit breaker pattern for failing providers
   - Return meaningful error messages to clients
   - Log all failures for debugging
   - Implement timeout handling (30s default)

### 3.3 Configuration

7. **Environment-Based Config**
   - Use environment variables for all sensitive configs
   - Support `.env` file loading
   - Include default values for non-sensitive settings

8. **Provider Configuration**
   - Support multiple model names per provider
   - Map internal model names to provider-specific models
   - Allow custom provider endpoints

### 3.4 Observability

9. **Logging and Monitoring**
   - Implement structured logging
   - Track request latency per provider
   - Log provider switches and reasons
   - Include request correlation IDs

## 4. File Structure

```
llm-gateway/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI workflow
├── src/
│   └── gateway/
│       ├── __init__.py
│       ├── main.py              # FastAPI application entry point
│       ├── config.py             # Configuration management
│       ├── models/
│       │   ├── __init__.py
│       │   ├── requests.py      # Pydantic request models
│       │   └── responses.py     # Pydantic response models
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py          # Base provider抽象
│       │   ├── g4f_provider.py  # g4f implementation
│       │   ├── litellm_provider.py
│       │   ├── mirror_provider.py
│       │   └── pool.py          # Provider rotation pool
│       ├── auth/
│       │   ├── __init__.py
│       │   ├── api_keys.py       # API key management
│       │   └── jwt_handler.py   # JWT handling
│       ├── tracking/
│       │   ├── __init__.py
│       │   ├── usage.py         # Usage tracking service
│       │   └── database.py     # SQLite operations
│       ├── middleware/
│       │   ├── __init__.py
│       │   ├── logging.py       # Request logging
│       │   └── rate_limit.py   # Rate limiting
│       └── utils/
│           ├── __init__.py
│           └── helpers.py       # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_providers.py       # Provider unit tests
│   ├── test_auth.py           # Authentication tests
│   ├── test_tracking.py       # Usage tracking tests
│   ├── test_integration.py   # Integration tests
│   └── conftest.py            # Pytest fixtures
├── docker/
│   └── Dockerfile
├── .env.example               # Example environment file
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Project metadata
├── AGENTS.md               # This file
└── QUESTIONS.md            # Questions for human review
```

## 5. Testing Requirements

### 5.1 Test Coverage Goals

| Category | Minimum Coverage | Focus Areas |
|----------|------------------|-------------|
| Unit Tests | 80% | Provider classes, auth, tracking |
| Integration | 70% | API endpoints, provider pool |
| Mock Tests | 100% | External API calls |

### 5.2 Test Types

1. **Unit Tests**
   - Test each provider class in isolation with mocked responses
   - Test authentication logic (key validation, JWT handling)
   - Test usage tracking calculations
   - Test provider pool rotation logic

2. **Integration Tests**
   - Test API endpoints with realistic scenarios
   - Test provider failover behavior
   - Test streaming responses

3. **Contract Tests**
   - Verify OpenAI-compatible response format
   - Validate error response structures

### 5.3 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src.gateway --cov-report=html

# Run specific test file
pytest tests/test_providers.py

# Run with verbose output
pytest -v
```

## 6. Git Protocol

### 6.1 Branch Strategy

- `main` - Production-ready code only
- `develop` - Integration branch for features
- `feature/*` - Individual feature branches
- `fix/*` - Bug fix branches

### 6.2 Commit Convention

```
