# AGENTS.md - LLM API Proxy Gateway with Dynamic Fallback

## 1. Role/Mission

You are an autonomous software engineer tasked with building a **LLM API Proxy Gateway with Dynamic Fallback**. This system acts as an intelligent middleware that:

- **Routes LLM API requests** across multiple LLM providers (OpenAI, Anthropic, Google, xAI, etc.)
- **Automatically falls back** to alternative providers when a primary provider fails
- **Optimizes provider ordering** using a weighted scoring algorithm that considers:
  - Benchmark scores (MMLU, HumanEval, GSM8K, etc.)
  - Token generation speeds (tokens/second)
  - Latency (time to first token)
  - Rate limits and usage quotas
- **Maps virtual model names** (e.g., `coding-smart`, `chat-fast`) to actual provider endpoints
- **Provides health monitoring** via REST endpoints

Your mission is to implement this gateway with minimal external dependencies, using free resources where possible, and to make all architectural decisions independently.

---

## 2. Technical Stack

### Core Framework
- **Primary**: Node.js with Express.js (TypeScript preferred for type safety)
- **Alternative**: Python with Flask (if agent prefers Python)
- Choose one and stick with it consistently

### Provider SDKs
- OpenAI SDK (`openai` npm package or `openai` pip package)
- Anthropic SDK (`@anthropic-ai/sdk` or `anthropic` pip package)
- Google Generative AI SDK (`@google/generative-ai`)
- xAI SDK (or direct REST API calls)

### Configuration
- YAML-based configuration file (`config.yaml` or `config.yml`)
- Dotenv for environment variables (`.env`)

### Logging & Monitoring
- `winston` (Node.js) or `python-logging` (Python)
- Simple in-memory metrics storage (no external DB required)

### Testing
- Jest (Node.js) or pytest (Python)
- Mock HTTP responses for provider testing

---

## 3. Requirements (Numbered)

### 3.1 Core Gateway Functionality
1. **Express/Flask Server**: Implement a REST API server listening on port 3000 (configurable via `PORT` env var)
2. **POST /v1/chat/completions**: Proxy chat completion requests to configured providers
3. **POST /v1/completions**: Proxy text completion requests to configured providers
4. **GET /models**: Return list of available virtual models and their mappings
5. **GET /health**: Return health status of all configured providers
6. **GET /metrics**: Return simple usage metrics (requests per provider, errors, latency percentiles)

### 3.2 Provider Configuration
7. **Multi-Provider Setup**: Support at least 3 providers (OpenAI, Anthropic, Google)
8. **API Key Management**: Load API keys from environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
9. **Provider Metadata**: Store benchmark scores, token speeds, and rate limits in `config.yaml`

### 3.3 Virtual Model Mapping
10. **Virtual Model Registry**: Define virtual models like:
    - `coding-smart` → maps to best coding benchmark model
    - `chat-fast` → maps to fastest response model
    - `reasoning-deep` → maps to best reasoning benchmark model
    - `balanced` → maps to best overall weighted model
11. **Model Alias Resolution**: Resolve virtual model names to actual provider models before routing

### 3.4 Dynamic Fallback System
12. **Fallback Chain**: Each virtual model has an ordered list of provider fallbacks
13. **Provider Scoring Algorithm**: Implement weighted scoring:
    ```
    Score = (BenchmarkScore * 0.4) + (TokenSpeedNorm * 0.3) + (LatencyInvNorm * 0.2) + (RateLimitNorm * 0.1)
    ```
    - Normalize each metric to 0-1 range
    - For latency, use inverse (lower is better)
14. **Automatic Reordering**: Re-sort fallback chain based on current health and recent performance
15. **Retry Logic**: Retry failed requests up to 3 times with exponential backoff (1s, 2s, 4s)

### 3.5 Health Monitoring
16. **Health Check Endpoint**: `GET /health` returns provider status (healthy/degraded/unhealthy)
17. **Periodic Health Checks**: Run lightweight health checks every 60 seconds
18. **Circuit Breaker**: Mark provider as unavailable after 5 consecutive failures
19. **Auto-Recovery**: Re-enable provider after 3 successful health checks

### 3.6 Error Handling
20. **Graceful Degradation**: Return meaningful error messages when all providers fail
21. **Request Timeout**: Implement 30-second timeout for all provider requests
22. **Rate Limit Handling**: Detect rate limit errors (429) and immediately try next provider

### 3.7 Documentation
23. **API Documentation**: Auto-generate OpenAPI/Swagger docs at `/api-docs`
24. **Configuration Guide**: Document how to add new providers and virtual models

---

## 4. File Structure

```
llm-proxy-gateway/
├── AGENTS.md
├── QUESTIONS.md
├── README.md
├── .env.example
├── config.yaml
├── package.json              # Node.js (or requirements.txt for Python)
├── tsconfig.json           # Node.js TypeScript config
├── jest.config.js          # Jest config
├── src/
│   ├── index.ts           # Main entry point
│   ├── app.ts             # Express app setup
│   ├── config/
│   │   ├── loader.ts      # Config loader
│   │   └── types.ts       # Config types
│   ├── providers/
│   │   ├── base.ts       # Base provider interface
│   │   ├── openai.ts     # OpenAI provider
│   │   ├── anthropic.ts # Anthropic provider
│   │   ├── google.ts     # Google provider
│   │   └── registry.ts   # Provider registry
│   ├── routing/
│   │   ├── router.ts     # Request router
│   │   ├── scorer.ts     # Provider scoring algorithm
│   │   └── fallback.ts   # Fallback chain logic
│   ├── models/
│   │   ├── mapper.ts     # Virtual model mapper
│   │   └── registry.ts   # Model registry
│   ├── health/
│   │   ├── checker.ts    # Health check logic
│   │   ├── monitor.ts    # Health monitor service
│   │   └── circuit.ts    # Circuit breaker
│   ├── middleware/
│   │   ├── logging.ts    # Request logging
│   │   └── timeout.ts    # Timeout handling
│   ├── routes/
│   │   ├── chat.ts       # Chat completions route
│   │   ├── completions.ts# Text completions route
│   │   ├── models.ts     # Models route
│   │   ├── health.ts     # Health route
│   │   └── metrics.ts    # Metrics route
│   ├── utils/
│   │   ├── logger.ts    # Logger setup
│   │   └── metrics.ts   # Metrics collector
│   └── types/
│       └── index.ts      # Shared types
├── tests/
│   ├── unit/
│   │   ├── scorer.test.ts
│   │   ├── mapper.test.ts
│   │   └── router.test.ts
│   ├── integration/
│   │   ├── providers.test.ts
│   │   └── fallback.test.ts
│   └── mocks/
│       ├── openai.mock.ts
│       ├── anthropic.mock.ts
│       └── google.mock.ts
└── docs/
    └── api.md
```

---

## 5. Testing Requirements

### 5.1 Unit Tests (Required)
1. **Scorer Tests**: Verify weighted scoring algorithm produces expected rankings
2. **Mapper Tests**: Verify virtual model names resolve to correct provider models
3. **Router Tests**: Verify requests route to correct provider based on configuration
4. **Fallback Tests**: Verify fallback chain iterates correctly on failures

### 5.2 Integration Tests (Required)
1. **Provider Mock Tests**: Test provider switching using mocked responses
2. **Error Handling Tests**: Test graceful degradation when all providers fail
3. **Timeout Tests**: Test request timeout and fallback trigger
4. **Circuit Breaker Tests**: Test circuit opens after failure threshold

### 5.3 Test Coverage
- Maintain minimum **80% code coverage** for core routing logic
- All critical paths must have test coverage

### 5.4 Testing with Free Resources
- Use mock providers (create mock HTTP servers) for testing
- No actual API calls to paid providers during test runs
- Environment variable `USE_MOCKS=true` enables mock mode

---

## 6. Git Protocol

### 6.1 Branch Strategy
- **Main branch**: `main` - stable, deployable code
- **Development branch**: `develop` - integration branch
- **Feature branches**: `feature