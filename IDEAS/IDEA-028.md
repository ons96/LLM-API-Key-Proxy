# AGENTS.md - Free LLM API Gateway

## 1. Role/Mission

**Mission:** Build a unified API gateway that provides secure, rate-limited access to free LLM services via OpenRouter. The gateway will act as a middleware layer between client applications and OpenRouter's free LLM endpoints, handling authentication, request validation, rate limiting, and response caching.

**Autonomous Agent Guidelines:**
- Make independent technical decisions when specifications are unclear
- Prioritize using free-tier resources and services only
- Save any clarifying questions to `QUESTIONS.md` in the root directory
- Proceed with best practices unless explicitly directed otherwise
- Self-validate code before committing

---

## 2. Technical Stack

| Component | Technology |
|-----------|------------|
| Runtime | Node.js (v18+) |
| Framework | Express.js |
| API Integration | OpenRouter AI API (free endpoints) |
| Rate Limiting | `express-rate-limit` |
| Request Validation | `zod` |
| Environment Variables | `dotenv` |
| HTTP Client | `axios` |
| Testing | Jest + Supertest |
| Logging | `winston` |
| CORS | `cors` |

---

## 3. Requirements

1. **Basic Server Setup**
   - Create an Express.js server running on port 3000 (default) or PORT from env
   - Implement health check endpoint at `GET /health`
   - Enable CORS for all origins (configurable via env)

2. **OpenRouter Integration**
   - Integrate with OpenRouter's free models API endpoint
   - Support chat completion requests (`POST /v1/chat/completions`)
   - Proxy requests to OpenRouter with proper headers and API key
   - Map responses back to client in OpenAI-compatible format

3. **Rate Limiting**
   - Implement per-IP rate limiting (default: 10 requests/minute)
   - Implement per-API-key rate limiting if authentication is added
   - Return proper HTTP 429 when limit exceeded
   - Make rate limit configurable via environment variables

4. **Request Validation**
   - Validate incoming chat completion requests using Zod
   - Validate required fields: `model`, `messages`
   - Reject invalid requests with descriptive error messages

5. **Configuration Management**
   - Use environment variables for all configuration
   - Required env vars: `OPENROUTER_API_KEY`, `PORT`
   - Optional env vars: `RATE_LIMIT_WINDOW_MS`, `RATE_LIMIT_MAX_REQUESTS`, `LOG_LEVEL`

6. **Logging**
   - Implement structured logging with Winston
   - Log all incoming requests (method, path, IP)
   - Log all errors with stack traces
   - Support log levels: debug, info, warn, error

7. **Error Handling**
   - Implement global error handler middleware
   - Return proper HTTP status codes (400, 401, 429, 500, 502, 503)
   - Return error messages in JSON format: `{ error: string, message?: string }`

8. **Security**
   - Validate and sanitize all user inputs
   - Do not expose API keys in logs
   - Implement request timeout (default: 60 seconds)
   - Add security headers using `helmet` (optional enhancement)

---

## 4. File Structure

```
free-llm-gateway/
├── .env.example                # Environment variable template
├── .gitignore                   # Git ignore patterns
├── package.json                 # Dependencies and scripts
├── jest.config.js               # Jest configuration
├── jsconfig.json                # JavaScript project config
├── src/
│   ├── index.js                 # Entry point
│   ├── app.js                   # Express app setup
│   ├── config/
│   │   └── index.js             # Configuration loader
│   ├── routes/
│   │   ├── health.js             # Health check routes
│   │   └── chat.js               # Chat completion routes
│   ├── middleware/
│   │   ├── rateLimiter.js        # Rate limiting middleware
│   │   ├── errorHandler.js       # Global error handler
│   │   └── requestLogger.js      # Request logging middleware
│   ├── services/
│   │   └── openRouter.js         # OpenRouter API client
│   ├── validators/
│   │   └── chat.js               # Request validation schemas
│   └── utils/
│       └── logger.js             # Winston logger setup
├── tests/
│   ├── integration/
│   │   └── chat.test.js          # Integration tests
│   ├── unit/
│   │   ├── validators.test.js    # Validator tests
│   │   └── services.test.js      # Service tests
│   └── setup.js                  # Test setup/teardown
├── QUESTIONS.md                  # Clarifying questions for human
└── README.md                     # Project documentation
```

---

## 5. Testing Requirements

**Test Coverage Requirements:**
- Minimum 80% code coverage required
- All endpoints must have integration tests
- All validators must have unit tests
- All services must have unit tests

**Testing Guidelines:**
- Use `jest` as the test framework
- Use `supertest` for HTTP integration testing
- Mock external API calls (OpenRouter) in unit tests
- Use real HTTP calls in integration tests with proper mocking
- Run tests before every commit (pre-commit hook or CI)

**Test Types:**
1. **Unit Tests:** Test individual functions, validators, and services in isolation
2. **Integration Tests:** Test full API flow including routing and middleware
3. **Error Cases:** Test error handling for invalid inputs, rate limits, and external API failures

---

## 6. Git Protocol

**Commit Rules:**
- Follow conventional commits format: `type(scope): description`
- Types: `feat`, `fix`, `chore`, `docs`, `test`, `refactor`
- Example: `feat(rate-limit): add per-IP rate limiting`

**Branch Strategy:**
- Main branch: `main` (production-ready code)
- Feature branches: `feature/description` or `feat/description`
- Bugfix branches: `fix/description`
- Working branch: `develop` (optional, for larger features)

**Commit Sequence:**
1. Create feature branch from `main`
2. Write code and tests
3. Run full test suite locally
4. Commit with descriptive message
5. Push and create PR to `main`
6. Require at least one review (if team) or self-review

**CI/CD (GitHub Actions):**
- Run tests on every push
- Run linting on every push
- Build verification on every push

---

## 7. Completion Criteria

**Functional Criteria:**
- [ ] Express server starts without errors
- [ ] `GET /health` returns 200 with `{ status: "ok" }`
- [ ] `POST /v1/chat/completions` accepts valid requests and returns LLM responses
- [ ] Rate limiting returns 429 when limit exceeded
- [ ] Invalid requests return 400 with descriptive error
- [ ] Server errors return 500 with generic message (no stack in response)

**Non-Functional Criteria:**
- [ ] All environment variables documented in `.env.example`
- [ ] At least 80% test coverage
- [ ] All tests pass
- [ ] No linting errors
- [ ] README.md with setup and usage instructions

**Delivery:**
- Working code in `src/` directory
- Passing tests in `tests/` directory
- Updated documentation files
- No blocking issues in `QUESTIONS.md`

---

## Important Notes

1. **Free Resources Only:** Do not use paid services or API keys. Use OpenRouter's free tier exclusively.

2. **Independent Decisions:** Where specifications are ambiguous, make reasonable assumptions and document them in code comments or `QUESTIONS.md`.

3. **Self-Validation:** Before marking any task complete, verify:
   - Code compiles/runs without errors
   - All tests pass
   - No linting warnings

4. **Documentation:** Update `QUESTIONS.md` only when blocked and unable to proceed.

---

*Generated for Autonomous Coding Agent - Free LLM API Gateway Project*