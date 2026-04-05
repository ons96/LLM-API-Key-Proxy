# AGENTS.md - Free LLM API Gateway

## 1. Role/Mission

**Mission**: Build a self-hosted, free LLM API gateway that provides unified REST API access to multiple LLM providers (OpenAI, Anthropic, Google, Ollama, etc.) with BYOK (Bring Your Own Key) support. The gateway acts as a proxy layer enabling external tools to use a single API endpoint while managing multiple provider credentials securely.

**Key Objectives**:
- Create a lightweight, self-hostable API gateway that runs on free infrastructure (e.g., Fly.io, Render free tier, Railway, or local Docker)
- Support BYOK model where users provide their own API keys for providers
- Provide unified REST API interface mapping to various LLM provider APIs
- Implement API key management system for user credentials
- Handle proxy routing, request transformation, and response normalization

---

## 2. Technical Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| **Runtime** | Node.js (v18+) | Cross-platform, extensive HTTP library support |
| **Framework** | Express.js | Lightweight, flexible HTTP server |
| **Validation** | Zod | Schema validation for requests/responses |
| **Configuration** | dotenv + convict | Environment-based config management |
| **Security** | helmet, express-rate-limit | Security headers + rate limiting |
| **Encryption** | node:crypto (AES-256-GCM) | Encrypt stored user API keys |
| **Testing** | Jest + Supertest | Unit and integration testing |
| **Containerization** | Docker | Self-hosting support |
| **Secrets** | Environment variables | No external secrets manager required |

---

## 3. Requirements (Numbered)

### 3.1 Core Gateway Functionality
- [ ] **R1.1** Implement Express.js REST API server on configurable port (default: 3000)
- [ ] **R1.2** Create `/v1/chat/completions` endpoint compatible with OpenAI API format
- [ ] **R1.3** Create `/v1/models` endpoint listing available models from all providers
- [ ] **R1.4** Implement provider proxy routing - detect model and route to appropriate provider
- [ ] **R1.5** Add request transformation layer to convert OpenAI-format requests to provider-specific formats
- [ ] **R1.6** Add response normalization layer to return OpenAI-format responses

### 3.2 BYOK API Key Management
- [ ] **R2.1** Create encrypted storage for user-provided API keys (AES-256-GCM)
- [ ] **R2.2** Implement POST `/keys` endpoint for users to add their API keys
- [ ] **R2.3** Implement GET `/keys` endpoint for users to list their keys (masked)
- [ ] **R2.4** Implement DELETE `/keys/:id` endpoint for users to remove keys
- [ ] **R2.5** Implement PUT `/keys/:id` endpoint for users to update keys
- [ ] **R2.6** Support multiple providers: OpenAI, Anthropic, Google AI, Ollama (local)

### 3.3 Authentication & Authorization
- [ ] **R3.1** Implement API key-based authentication for gateway access
- [ ] **R3.2** Create POST `/auth/register` endpoint for gateway user registration
- [ ] **R3.3** Create POST `/auth/login` endpoint for gateway authentication returning JWT
- [ ] **R3.4** Implement JWT middleware for protected endpoints
- [ ] **R3.5** Add rate limiting per user to prevent abuse

### 3.4 Configuration & Environment
- [ ] **R4.1** Support configuration via environment variables
- [ ] **R4.2** Support `.env` file for local development
- [ ] **R4.3** Require `JWT_SECRET`, `ENCRYPTION_KEY`, `PORT` as required env vars
- [ ] **R4.4** Support optional `DATABASE_URL` for persistent key storage (SQLite for free tier)

### 3.5 Docker & Self-Hosting
- [ ] **R5.1** Create `Dockerfile` for containerized deployment
- [ ] **R5.2** Create `docker-compose.yml` for local development
- [ ] **R5.3** Create `start.sh` script for quick deployment

### 3.6 Logging & Monitoring
- [ ] **R6.1** Implement structured logging (JSON format)
- [ ] **R6.2** Log all API requests with timing and status
- [ ] **R6.3** Add error logging with stack traces
- [ ] **R6.4** Expose `/health` endpoint for health checks

---

## 4. File Structure

```
llm-gateway/
├── src/
│   ├── index.js                 # Entry point
│   ├── app.js                   # Express app configuration
│   ├── config/
│   │   └── index.js             # Configuration loader
│   ├── routes/
│   │   ├── auth.js              # Auth endpoints
│   │   ├── keys.js              # API key management
│   │   ├── chat.js              # Chat completions proxy
│   │   ├── models.js            # Models listing
│   │   └── health.js            # Health check
│   ├── middleware/
│   │   ├── auth.js              # JWT authentication
│   │   ├── rateLimiter.js       # Rate limiting
│   │   └── errorHandler.js      # Error handling
│   ├── services/
│   │   ├── proxy.js             # Provider proxy service
│   │   ├── transformers.js      # Request/response transformers
│   │   └── encryption.js       # Key encryption service
│   ├── providers/
│   │   ├── openai.js            # OpenAI provider
│   │   ├── anthropic.js        # Anthropic provider
│   │   ├── google.js            # Google AI provider
│   │   └── ollama.js            # Ollama provider
│   ├── db/
│   │   └── sqlite.js            # SQLite initialization
│   └── utils/
│       └── logger.js           # Logger utility
├── tests/
│   ├── routes/
│   │   ├── auth.test.js
│   │   ├── keys.test.js
│   │   └── chat.test.js
│   ├── services/
│   │   ├── proxy.test.js
│   │   └── transformers.test.js
│   └── providers/
│       ├── openai.test.js
│       └── anthropic.test.js
├── scripts/
│   ├── start.sh                 # Deployment script
│   └── seed.js                  # Seed test data
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── package.json
├── jest.config.js
├── README.md
└── AGENTS.md
```

---

## 5. Testing Requirements

### 5.1 Test Coverage Goals
- **Minimum 80% code coverage** required for all src files
- **100% coverage** required for: transformers.js, proxy.js, encryption.js

### 5.2 Test Types

| Test Type | Coverage Target | Tool |
|-----------|-----------------|------|
| Unit Tests | Core services, utilities | Jest |
| Integration Tests | API endpoints | Supertest |
| Provider Mock Tests | All provider adapters | Jest mocks |

### 5.3 Required Tests

- [ ] **T5.3.1** Test all route handlers return correct status codes
- [ ] **T5.3.2** Test JWT auth middleware rejects invalid tokens
- [ ] **T5.3.3** Test API key encryption/decryption roundtrip
- [ ] **T5.3.4** Test request transformation for all providers
- [ ] **T5.3.5** Test response normalization returns valid OpenAI format
- [ ] **T5.3.6** Test proxy routing selects correct provider based on model
- [ ] **T5.3.7** Test rate limiting blocks excessive requests
- [ ] **T5.3.8** Test health endpoint returns 200

### 5.4 Running Tests

```bash
# Run all tests with coverage
npm test -- --coverage

# Run tests in watch mode
npm test -- --watch

# Run specific test file
npm test -- tests/routes/auth.test.js
```

---

## 6. Git Protocol

### 6.1 Branch Strategy
- **Main branch**: `main` - Production-ready code only
- **Development branch**: `develop` - Integration branch
- **Feature branches**: `feature