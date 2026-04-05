# AGENTS.md - Free LLM API Provider Aggregator

## 1. Role/Mission

The autonomous agent will build a **Free LLM API Provider Aggregator** — a unified gateway that combines multiple free LLM API providers into a single, reliable API endpoint with automatic failover, health monitoring, and usage tracking.

### Mission Objectives

- Create a RESTful API gateway that abstracts away the complexity of multiple LLM providers
- Implement automatic failover to switch to alternative providers when one fails
- Build health monitoring to track provider status, latency, and availability
- Implement usage tracking to log token consumption and API call metrics
- Provide a unified response format regardless of which underlying provider handles the request
- Prioritize free-tier providers and free trial allocations

### Success Criteria

- External applications can make a single API call to this gateway
- The gateway intelligently routes requests to available free providers
- If a provider fails or becomes unavailable, the system automatically retries with an alternative provider
- All provider interactions are logged for monitoring and debugging

---

## 2. Technical Stack

### Runtime & Language
- **Runtime**: Node.js 20+ (LTS)
- **Language**: TypeScript

### Framework & Libraries
- **Web Framework**: Fastify (preferred for performance) or Express.js
- **HTTP Client**: Axios or got (for provider API calls)
- **Validation**: Zod (for request/response validation)
- **Logging**: Pino (for structured logging)
- **Rate Limiting**: express-rate-limit or custom implementation
- **Metrics**: prom-client (for Prometheus metrics export)

### Provider APIs (Free Tier Target)
- **OpenRouter.ai** — Aggregator with free tier access
- **NovitaAI** — Free tier available
- **ModelScope** — Free endpoints for specific models
- **Cerebras** — Free tier access
- **Hugging Face** — Free inference endpoints
- **iFlow** — Free tier available

### DevOps & Infrastructure
- **CI/CD**: GitHub Actions
- **Hosting**: Local development / Self-hosted
- **Container**: Docker for consistent deployment

### Additional Tools
- **Testing**: Vitest or Jest
- **Type Checking**: TypeScript strict mode

---

## 3. Requirements (Numbered)

### Core Requirements

1. **Unified API Gateway**
   - Expose a single RESTful endpoint (e.g., `POST /v1/chat/completions`)
   - Accept OpenAI-compatible request format
   - Return standardized responses in OpenAI format

2. **Provider Abstraction Layer**
   - Create provider adapter interfaces for each supported provider
   - Implement adapters for: OpenRouter, NovitaAI, ModelScope, Cerebras, Hugging Face, iFlow
   - Normalize request format to each provider's expected structure
   - Normalize responses to unified format

3. **Automatic Failover**
   - Implement a provider selection strategy (round-robin, least-latency, priority-based)
   - If a provider returns an error or times out, automatically attempt the next provider
   - Configure max retry attempts per request (default: 2)
   - Configure timeout threshold (default: 30 seconds)

4. **Health Monitoring**
   - Implement a health check endpoint (`GET /health`)
   - Run periodic health checks for each provider (every 60 seconds)
   - Track per-provider metrics: latency, success rate, error rate
   - Maintain provider status (healthy, degraded, unavailable)
   - Expose provider metrics via `/metrics` endpoint

5. **Usage Tracking**
   - Log all API requests with timestamp, provider Used, model, token count
   - Track input tokens, output tokens, total tokens
   - Store logs in-memory or file-based (JSON Lines format)
   - Provide an endpoint to retrieve usage stats (`GET /usage`)

6. **Configuration Management**
   - Use environment variables for all provider API keys and endpoints
   - Use a configuration file (`config.yaml` or `config.json`) for non-secret settings
   - Support provider priority weights

### Secondary Requirements

7. **Rate Limiting**
   - Implement per-IP rate limiting (prevent abuse)
   - Implement global rate limiting based on free tier limits

8. **Request Caching (Optional)**
   - Cache exact duplicate requests for a short period (60 seconds)
   - Use simple in-memory cache

9. **Error Handling**
   - Return meaningful error messages to clients
   - Map provider errors to standard error codes

10. **CORS Support**
    - Enable CORS for web application integration

---

## 4. File Structure

```
free-llm-aggregator/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI workflow
├── src/
│   ├── index.ts                # Application entry point
│   ├── config/
│   │   ├── index.ts            # Configuration loader
│   │   └── schema.ts           # Config validation schema
│   ├── providers/
│   │   ├── base.ts             # Base provider interface
│   │   ├── openrouter.ts       # OpenRouter adapter
│   │   ├── novita.ts           # NovitaAI adapter
│   │   ├── modelscope.ts       # ModelScope adapter
│   │   ├── cerebras.ts         # Cerebras adapter
│   │   ├── huggingface.ts      # Hugging Face adapter
│   │   └── iflow.ts            # iFlow adapter
│   ├── router/
│   │   ├── index.ts            # Main router setup
│   │   ├── chat.ts             # Chat completions route
│   │   ├── health.ts           # Health check route
│   │   ├── metrics.ts          # Metrics route
│   │   └── usage.ts            # Usage stats route
│   ├── middleware/
│   │   ├── error.ts           # Error handling middleware
│   │   ├── rate-limit.ts      # Rate limiting middleware
│   │   └── cors.ts             # CORS middleware
│   ├── services/
│   │   ├── aggregator.ts      # Main aggregator service
│   │   ├── failover.ts        # Failover logic service
│   │   ├── monitor.ts          # Health monitoring service
│   │   └── tracker.ts         # Usage tracking service
│   ├── types/
│   │   └── index.ts           # Shared TypeScript types
│   └── utils/
│       ├── logger.ts         # Logger setup
│       └── http.ts           # HTTP client utilities
├── tests/
│   ├── providers/
│   │   └── providers.test.ts
│   ├── services/
│   │   ├── aggregator.test.ts
│   │   └── monitor.test.ts
│   └── integration/
│       └── chat.test.ts
├── config/
│   └── default.json          # Default configuration
├── Dockerfile
├── package.json
├── tsconfig.json
├── .env.example              # Example environment variables
├── README.md
└── AGENTS.md                 # This file
```

---

## 5. Testing Requirements

### Unit Tests
- Test each provider adapter for correct request/response transformation
- Test configuration loading and validation
- Test failover logic with mocked providers

### Integration Tests
- Test the full gateway flow with mocked provider responses
- Test health endpoint returns correct provider statuses
- Test rate limiting correctly blocks excessive requests

### Testing Strategy
- Use **Vitest** for test framework
- Mock provider HTTP responses using Nock or similar
- Maintain >80% code coverage
- Run tests on every push via GitHub Actions

### CI Pipeline
- Lint and type-check on every PR
- Run full test suite on every push to `main` and PRs
- Build Docker image on push to `main`

---

## 6. Git Protocol

### Branch Strategy
- **Main branch**: `main` — Always deployable, contains stable code
- **Development branch**: `develop` — Integration branch for features
- **Feature branches**: `feature