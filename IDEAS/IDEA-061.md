# AGENTS.md - Intelligent Gateway Auto-Fallback System

## 1. Role/Mission

**Role**: Autonomous AI Coding Agent for Intelligent Gateway Auto-Fallback System

**Mission**: Build a gateway system that routes LLM requests through a configurable fallback chain of model providers. The system must automatically detect failures (rate limits, payment issues, provider outages, model unavailability), skip to the next available model in the fallback list, and temporarily blacklist models that are rate-limited until their limits reset. All decisions should be made autonomously using free available resources.

---

## 2. Technical Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Runtime** | Node.js (v18+) | Free, widely supported, runs on GitHub Actions |
| **HTTP Client** | Built-in `fetch` or `axios` | Simple HTTP calls to LLM providers |
| **State Storage** | JSON file-based (in `data/`) | No external dependencies, works offline |
| **Configuration** | YAML (`config/models.yaml`) | Human-readable, easy to modify |
| **Logging** | Console + file (`logs/gateway.log`) | Debug and audit trail |
| **Testing** | Jest (node test runner) | Free, well-supported |
| **CI/CD** | GitHub Actions | Already in use, free tier available |

**Providers to Support (Free Tiers)**:
- OpenAI (free tier API credits for new users)
- Anthropic (free tier available)
- Ollama (local, free - runs locally or via cloud)
- HuggingFace Inference API (free tier)
- Groq (free tier, fast)

---

## 3. Requirements

### Core Functionality

1. **Model Configuration Management**
   - Load model provider list from `config/models.yaml`
   - Each entry must include: name, provider, endpoint, API key env var, priority
   - Support priority ordering (lower = higher priority)

2. **Request Execution**
   - Accept a prompt/payload and execute against current model
   - Return first successful response
   - Track attempt count per request

3. **Automatic Fallback Chain**
   - On any failure, automatically try next model in priority list
   - Continue until success or all models exhausted
   - Return detailed error if all models fail

4. **Intelligent Rate Limit Blocking**
   - Detect 429 status codes from responses
   - Blacklist rate-limited model temporarily
   - Track reset time based on `Retry-After` header or default (60s)
   - Remove from blacklist when reset time passes

5. **Error Type Handling**
   - **429 (Rate Limit)**: Add to temporary blacklist
   - **402 (Payment Required)**: Add to persistent blacklist (requires manual intervention)
   - **503 (Service Unavailable)**: Add to temporary blacklist (5 min cooldown)
   - **400 (Bad Request)**: Log error, do NOT fallback (invalid prompt)
   - **401 (Unauthorized)**: Add to persistent blacklist (invalid API key)

6. **Blacklist Management**
   - Store blacklist in `data/blacklist.json`
   - Each entry: model name, reason, expiresAt timestamp
   - Auto-cleanup expired entries on startup and before each request

7. **State Persistence**
   - Save state to disk after each modification
   - Support graceful shutdown
   - Resume state on restart

### Logging & Monitoring

8. **Request Logging**
   - Log all attempts: timestamp, model, prompt (truncated), success/failure
   - Track latency per model
   - Store in `logs/requests.jsonl` format

9. **Metrics Collection**
   - Count successes/failures per model
   - Track fallback rates
   - Store in `data/metrics.json`

10. **Health Reporting**
    - Export summary of model availability
    - Report blacklist status

### CLI/UX

11. **Command-Line Interface**
    - `node index.js --prompt "your prompt here"` - Execute a single prompt
    - `node index.js --config check` - Validate configuration
    - `node index.js --status` - Show current model availability
    - `node index.js --blacklist clear` - Clear all blacklists (for testing)

12. **Environment Configuration**
    - All API keys via environment variables
    - Support `.env` file loading (gitignored)
    - Example: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.

---

## 4. File Structure

```
intelligent-gateway/
├── .github/
│   └── workflows/
│       └── test.yml              # GitHub Actions workflow
├── config/
│   ├── models.yaml              # Model provider configuration
│   └── defaults.yaml             # Default settings (timeouts, retries)
├── src/
│   ├── index.js                 # Main entry point & CLI
│   ├── gateway.js               # Core gateway logic
│   ├── providers/
│   │   ├── base.js              # Base provider class
│   │   ├── openai.js            # OpenAI provider
│   │   ├── anthropic.js         # Anthropic provider
│   │   ├── ollama.js            # Ollama provider
│   │   └── huggingface.js       # HuggingFace provider
│   ├── blacklist.js            # Blacklist management
│   ├── state.js                 # State persistence
│   ├── config-loader.js         # YAML config loading
│   └── logger.js                # Logging utilities
├── data/
│   ├── blacklist.json          # Current blacklisted models
│   ├── metrics.json            # Usage metrics
│   └── state.json               # Saved gateway state
├── logs/
│   ├── gateway.log              # General logs
│   └── requests.jsonl           # Request audit trail
├── tests/
│   ├── blacklist.test.js        # Blacklist logic tests
│   ├── gateway.test.js           # Gateway fallback tests
│   ├── providers.test.js        # Provider tests
│   └── fixtures/
│       ├── config.yaml          # Test config
│       └── mock-responses.js     # Mock HTTP responses
├── .env.example                 # Example environment vars
├── jest.config.js              # Jest configuration
├── package.json
├── README.md
└── AGENTS.md                    # This file
```

---

## 5. Testing Requirements

### Unit Tests (Mandatory)

| Test Suite | Coverage | Priority |
|------------|----------|----------|
| `blacklist.test.js` | Add, remove, check expiry, persistence | HIGH |
| `gateway.test.js` | Fallback chain logic, error handling | HIGH |
| `providers.test.js` | Provider instantiation, request formatting | MEDIUM |
| `state.test.js` | Save/load state, cleanup expired | MEDIUM |

### Test Scenarios

1. **Happy Path**: Single model succeeds → return response
2. **Fallback on Rate Limit**: First model returns 429 → fallback to second → success
3. **Fallback on Provider Down**: First returns 503 → fallback to second → success
4. **All Models Fail**: All return errors → return aggregated error
5. **Blacklist Expiry**: Model blacklisted → wait for expiry → model re-enabled
6. **Persistent Blacklist**: Payment error → model stays blacklisted
7. **Concurrent Requests**: Multiple requests in parallel → no state corruption

### Test Execution

```bash
# Run all tests
npm test

# Run with coverage
npm test -- --coverage

# Run specific test file
npm test -- blacklist.test.js
```

---

## 6. Git Protocol

### Branch Strategy

- **Main branch**: `main` - Production-ready code only
- **Working branch**: `dev` - Feature development
- **Feature branches**: `feat