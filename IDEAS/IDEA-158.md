# AGENTS.md - Intelligent LLM Gateway/Router

---

## 1. Role/Mission

### Project Overview
Build an **Intelligent LLM Gateway/Router** - an all-in-one API gateway that abstracts multiple LLM providers behind a unified interface. The system automatically selects the optimal model for each request, handles fallback scenarios gracefully, and optimizes for free tier usage while maintaining high reliability.

### Mission Statement
Create a production-ready, self-healing LLM gateway that:
- Provides a unified API endpoint for multiple LLM providers (OpenAI, Anthropic, Google, Ollama, etc.)
- Intelligently routes requests based on task type, cost, rate limits, and availability
- Automatically falls back to alternative models/providers when primary options fail
- Supports multiple operation modes (coding, planning, chat, multi-LLM reasoning)
- Maximizes free tier usage through intelligent负载均衡 and provider rotation

### Target Users
- Developers building AI-powered applications
- Autonomous coding agents
- AI chatbots and conversational interfaces
- Any system requiring reliable LLM access without vendor lock-in

---

## 2. Technical Stack

### Core Framework
- **FastAPI** - Modern, high-performance Python web framework
- **Python 3.10+** - For async/await support and modern type hints

### Key Dependencies
```
fastapi>=0.109.0
uvicorn>=0.27.0
httpx>=0.26.0          # Async HTTP client
pydantic>=2.5.0        # Data validation
python-dotenv>=1.0.0   # Environment variable management
aiohttp>=3.9.0         # Alternative async HTTP
loguru>=0.7.0          # Enhanced logging
tenacity>=8.2.0        # Retry logic
```

### LLM Provider Support (Free Tiers First)
1. **Ollama** - Local/hosted open-source models (free)
2. **OpenAI** - GPT-3.5 Turbo (free tier available)
3. **Anthropic** - Claude 3 Haiku (free tier available)
4. **Google AI** - Gemini Pro (free tier available)
5. **Groq** - Fast inference (free tier available)
6. **Cohere** - Command R (free tier available)
7. **Mistral** - Open-source models (free)

### Configuration
- Environment-based configuration (`.env` files)
- YAML support for complex routing rules

---

## 3. Requirements (Numbered)

### 3.1 Core Gateway Functionality
- [ ] **Unified REST API** - Single endpoint handling all LLM requests
- [ ] **Request Transformation** - Normalize requests across different provider formats
- [ ] **Response Transformation** - Standardize responses regardless of provider
- [ ] **Streaming Support** - Full Server-Sent Events (SSE) implementation for all providers

### 3.2 Model Selection & Routing
- [ ] **Auto Mode Selection** - Automatically choose best model based on task type
- [ ] **Manual Model Override** - Allow explicit model/provider selection
- [ ] **Task Classification** - Detect coding, planning, chat, or multi-LLM tasks
- [ ] **Mode Switching** - Support distinct modes: `coding-planning`, `coding-generation`, `chat-fast`, `chat-smartest`, `multi`

### 3.3 Fallback & Resilience
- [ ] **Automatic Fallback** - Retry failed requests with alternative models
- [ ] **Provider Health Checking** - Monitor provider availability
- [ ] **Rate Limit Handling** - Detect and bypass rate limits automatically
- [ ] **Circuit Breaker** - Temporarily disable failing providers
- [ ] **Exponential Backoff** - Smart retry timing

### 3.4 Free Usage Optimization
- [ ] **Usage Tracking** - Monitor daily/monthly API usage per provider
- [ ] **Provider Rotation** - Distribute requests across free tiers
- [ ] **Cost Awareness** - Prioritize free models when available
- [ ] **Token Budgeting** - Implement per-user or global token limits

### 3.5 Multi-LLM Reasoning
- [ ] **LLM Counsel Mode** - Query multiple models and merge perspectives
- [ ] **Expert Voting** - Aggregate responses from specialized models
- [ ] **Reasoning Comparison** - Compare thought processes across models

### 3.6 Developer Experience
- [ ] **Interactive Docs** - FastAPI automatic documentation (Swagger/ReDoc)
- [ ] **Request Logging** - Comprehensive audit trail
- [ ] **Metrics Endpoint** - Provider usage statistics
- [ ] **Health Check** - Gateway and provider status endpoint

### 3.7 Security & Configuration
- [ ] **API Key Management** - Secure storage and rotation
- [ ] **Request Validation** - Input sanitization
- [ ] **Rate Limiting** - Per-client rate limits on gateway level
- [ ] **Environment Configuration** - No hardcoded credentials

---

## 4. File Structure

```
llm-gateway/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI/CD
├── .env.example                    # Example environment variables
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project metadata
├── README.md                       # Project documentation
├── AGENTS.md                       # This file
├── QUESTIONS.md                    # Questions for human review
├── app/
│   ├── __init__.py                 # Package marker
│   ├── main.py                      # FastAPI application entry point
│   ├── config.py                    # Configuration management
│   ├── logging.py                  # Logging setup
│   ├── exceptions.py               # Custom exceptions
│   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── requests.py         # Request schemas
│   │   │   ├── responses.py        # Response schemas
│   │   │   └── config.py           # Configuration schemas
│   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── router.py           # Request routing logic
│   │   │   ├── selector.py         # Model selection logic
│   │   │   ├── fallback.py         # Fallback handling
│   │   │   └── tracker.py          # Usage tracking
│   ├── providers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # Abstract provider base class
│   │   │   ├── openai.py           # OpenAI provider
│   │   │   ├── anthropic.py        # Anthropic provider
│   │   │   ├── google.py          # Google AI provider
│   │   │   ├── ollama.py           # Ollama provider
│   │   │   ├── groq.py             # Groq provider
│   │   │   └── cohere.py           # Cohere provider
│   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py             # Chat endpoint
│   │   │   ├── completions.py      # Completion endpoint
│   │   │   ├── streaming.py        # Streaming endpoint
│   │   │   └── health.py          # Health check endpoint
│   └── utils/
│       │   ├── __init__.py
│       ├── retry.py                # Retry utilities
│       └── validators.py          # Input validation
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Test fixtures
│   ├── test_providers/
│   │   │   ├── __init__.py
│   │   │   ├── test_base.py
│   │   │   └── test_providers.py
│   ├── test_routes/
│   │   │   ├── __init__.py
│   │   │   └── test_chat.py
│   └── test_integration/
│       ├── __init__.py
│       └── test_gateway.py
├── scripts/
│   ├── setup_env.sh               # Environment setup script
│   └── health_check.py            # Health check utility
└── docs/
    ├── api_reference.md
    ├── provider_setup.md
    └── configuration.md
```

---

## 5. Testing Requirements

### 5.1 Test Coverage Goals
- **Minimum 80% code coverage** required
- All critical paths must have unit tests
- All API routes must have integration tests

### 5.2 Test Types Required

#### Unit Tests
- Provider adapter classes
- Model selection logic
- Fallback mechanisms
- Request/response transformation
- Usage tracking logic

#### Integration Tests
- Full request/response cycle for each mode
- Fallback chain verification
- Multi-LLM reasoning mode
- Streaming response handling
- Rate limiting behavior

#### Contract Tests
- Response schema consistency
- Error response format consistency

### 5.3 Test Infrastructure
```python
# Test dependencies
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
httpx>=0.26.0           # For async test clients
respx>=0.20.0          # Mock HTTP responses
aioresponses>=0.7.6     # Mock async HTTP
faker>=22.0.0          # Fake data generation
```

### 5.4 Running Tests
```bash
# Run all tests with coverage
pytest --cov=app --cov-report=html

# Run specific test categories
pytest tests/test_providers/
pytest tests/test_routes/

# Run with verbose output
pytest -v

# Run only fast tests (skip integration)
pytest -m "not integration"
```

### 5.5 Mocking Strategy
- Use environment variables for provider API keys
- Mock external HTTP calls in unit tests
- Use actual providers only in designated integration tests
- Implement mock provider for development testing

---

## 6. Git Protocol

### 6.1 Branch Strategy
```
main                    # Production-ready code
├── develop            # Integration branch
├── feature/*          # Feature branches
├── bugfix/*           # Bug fix branches
└── hotfix/*          # Emergency fixes
```

### 6.2 Commit Message Format
```
