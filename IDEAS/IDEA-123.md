# AGENTS.md

## Intelligent Multi-Provider LLM Gateway with Dynamic Fallback

---

## 1. Role/Mission

You are the autonomous coding agent responsible for building an **Intelligent Multi-Provider LLM Gateway**—a unified API layer that abstracts away the complexity of communicating with multiple LLM providers (OpenAI, Anthropic, Ollama, local/in-house models, and any other compatible APIs).

Your mission is to create a production-ready system that:

- **Intelligently routes** LLM requests across providers using dynamic fallback logic
- **Adapts** model selection based on real-time performance metrics (speed, cost, quality)
- **Handles failures gracefully** with configurable retry and fallback strategies
- **Persists context** across sessions with intelligent pruning
- **Tracks provider health** to distinguish temporary outages (rate limits) from long-term failures
- **Empowers users** with virtual model definitions (coding-elite, coding-fast, chat-best, etc.) that map to optimal real models
- **Operates within free resource constraints** (Oracle Cloud Free Tier VPS)

You MUST make independent architectural decisions where specifications are incomplete. Save any clarifying questions to `QUESTIONS.md` in the project root.

---

## 2. Technical Stack

### Runtime & Language
- **Node.js** (latest LTS, v20.x) — chosen for extensive async/await support, rich ecosystem for HTTP clients, and lightweight deployment
- **TypeScript** — for type safety and maintainability

### Core Dependencies
- **Express.js** (v4.x) — REST API gateway server
- **Axios** (v1.x) — HTTP client for provider API calls
- **ioredis** (v5.x) — Redis client for health tracking and caching (compatible with Redis Cloud free tier)
- **dotenv** — environment variable management
- **yaml** — config file parsing for fallback order files
- **uuid** — session/context ID generation

### CLI & Automation
- **Commander.js** (v11.x) — CLI for running sync scripts, health checks, config generation

### Storage (Free Tier Compatible)
- **Redis Cloud** (free tier) — health tracking, rate limit counters, session metadata
- **Filesystem (JSON/YAML)** — fallback order files, virtual model definitions, persisted context (for sessions without Redis)

### Client-Side Tools
- **GitHub API** (via Octokit) — for pushing regenerated fallback order files to repositories

### Testing
- **Vitest** (v1.x) — unit and integration testing
- **Supertest** — API endpoint testing
- **Mockery** — mocking external provider APIs

### DevOps (Minimal for Free Tier)
- **GitHub Actions** — CI/CD for automated testing and deployment to Oracle VPS
- **PM2** — process management on VPS

---

## 3. Requirements

### 3.1 Multi-Provider Abstraction Layer
1. Create a generic `LLMProvider` interface/abstract class that defines:
   - `name`: provider identifier
   - `models`: array of available model names
   - `complete(prompt, options)`: method for chat completion
   - `embed(text)`: method for embeddings (if supported)
   - `healthCheck()`: method for availability verification
2. Implement concrete adapters for:
   - **OpenAI** (ChatGPT family, including GPT-4, GPT-4 Turbo, GPT-3.5 Turbo)
   - **Anthropic** (Claude family: Claude 3 Opus, Sonnet, Haiku)
   - **Ollama** (local running models via HTTP)
   - **Generic OpenAI-compatible APIs** (allows adding local/in-house models)
3. All adapters MUST handle:
   - Request formatting (headers, payload)
   - Response parsing (normalizing to a unified format)
   - Error interpretation (rate limit, auth failure, timeout, server error)
   - Timeout configuration (configurable per-provider)

### 3.2 Virtual Model System
4. Define a **virtual model** concept — a logical name (e.g., `coding-elite`, `coding-fast`, `chat-best`) that maps to:
   - A **fallback order list** of real provider models (ordered best to worst)
   - A **cost权重** (cost weight: low/medium/high)
   - A **speed权重** (speed weight: low/medium/high)
   - A **quality权重** (quality weight: low/medium/high)
5. Create a config file `virtual-models.yaml` that users can edit to define virtual models
6. Implement a service that resolves a virtual model name to a concrete model + provider using the fallback order

### 3.3 Dynamic Fallback Logic
7. Implement a `FallbackRouter` that:
   - Takes a virtual model name
   - Iterates through its fallback order list
   - Skips providers marked as "long-term outage" in health tracking
   - Attempts the first available provider
   - On failure, automatically tries the next provider in the order
   - Collects diagnostic info (latency, error type) for health tracking
8. Support **manual override** — if a user explicitly specifies a provider/model, try ONLY that provider (or all providers offering that model) until success or complete failure
9. Configurable max retry attempts per provider (default: 2)

### 3.4 Context Management
10. Implement a `SessionManager` that:
    - Creates/retrieves sessions by ID
    - Stores conversation history (messages with roles: system, user, assistant)
    - Supports context transfer between sessions (copy history from session A to session B)
11. Implement a `ContextPruner` that:
    - Trims conversation history when token limits approach
    - Uses conservative truncation (remove oldest messages first)
    - Preserves system prompts
    - Configurable max tokens per session (default: 8192, configurable per virtual model)

### 3.5 Provider Health Tracking
12. Implement a `HealthTracker` using Redis (or filesystem fallback) that tracks:
    - **Provider status**: available, degraded, rate-limited, down
    - **Last success timestamp**
    - **Last failure timestamp**
    - **Failure reason category**: temporary (rate limit, timeout) vs persistent (auth, bad config, provider outage)
    - **Consecutive failure count**
13. Implement a decay algorithm:
    - Providers marked as "temporary failure" (rate limit) can be retried after a configurable cooldown (default: 60 seconds)
    - Providers marked as "persistent failure" require manual re-enable or longer cooldown (default: 10 minutes)
14. Implement a health check background job that pings each provider every 60 seconds and updates Redis

### 3.6 Prompt Analysis Module
15. Create a `PromptAnalyzer` that classifies incoming prompts to suggest optimal models:
    - **Task type**: coding, reasoning, creative writing, summarization, Q&A
    - **Complexity**: simple, moderate, complex
    - **Length**: short, medium, long
16. Use the classification to optionally suggest virtual models (e.g., complex coding → coding-elite, simple Q&A → chat-fast)
17. This module is advisory — the user's explicit virtual model choice takes precedence

### 3.7 Fallback Order File Generator
18. Create a CLI command that generates optimized fallback order files based on:
    - Historical health data from Redis (success rates, latency percentiles)
    - Cost data from config
    - User-defined priorities (speed vs cost vs quality weights)
19. Output a `fallback-order.yaml` file in the config directory
20. Provide a client-side script (`generate-fallback-order.js`) that runs locally, reads data from a local JSON export of Redis stats, and writes the optimized file

### 3.8 Client-Side GitHub Sync
21. Implement a `GitHubSync` module that:
    - Authenticates via GitHub Personal Access Token (user provides in env)
    - Reads the generated fallback order file
    - Commits and pushes changes to a designated repository
    - Supports configurable commit messages
22. Document the workflow:
    - User runs CLI command locally → generates optimized file → pushes to GitHub
    - VPS pulls changes on next deployment or via webhook

### 3.9 API Gateway Server
23. Build an Express.js server exposing:
    - `POST /v1/chat/completions` — main chat endpoint (OpenAI-compatible)
    - `GET /v1/models` — list available virtual and real models
    - `GET /health` — gateway health status
    - `POST /admin/session` — create/manage sessions
    - `GET /admin/session/:id` — retrieve session context
    - `POST /admin/session/:id/transfer` — transfer context to another session
24. Support query parameters:
    - `model` (required): virtual model name (e.g., `coding-elite`)
    - `provider` (optional): manual provider override
    - `session_id` (optional): continue existing session
    - `temperature`, `max_tokens`, etc. — passthrough options

### 3.10 Logging & Observability
25. Implement structured logging (JSON format) for:
    - Incoming requests (sanitized)
    - Provider calls (latency, success/failure, error details)
    - Fallback attempts
    - Health state changes
26. Log to stdout in JSON format for easy parsing by external tools

---

## 4. File Structure

```
llm-gateway/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy.yml
├── src/
│   ├── adapters/
│   │   ├── index.ts
│   │   ├── openai.ts
│   │   ├── anthropic.ts
│   │   ├── ollama.ts
│   │   └── openai-compatible.ts
│   ├── core/
│   │   ├── provider.interface.ts
│   │   ├── fallback-router.ts
│   │   ├── health-tracker.ts
│   │   └── session-manager.ts
│   ├── services/
│   │   ├── prompt-analyzer.ts
│   │   ├── context-pruner.ts
│   │   └── github-sync.ts
│   ├── cli/
│   │   ├── commands/
│   │   │   ├── health-check.ts
│   │   │   ├── generate-fallback.ts
│   │   │   └── github-push.ts
│   │   └── index.ts
│   ├── server/
│   │   ├── index.ts
│   │   ├── routes/
│   │   │   ├── chat.ts
│   │   │   ├── models.ts
│   │   │   ├── admin.ts
│   │   │   └── health.ts
│   │   └── middleware/
│   │       ├── error-handler.ts
│   │       └── logger.ts
│   ├── config/
│   │   ├── virtual-models.yaml
│   │   ├── providers.yaml
│   │   └── fallback-order.yaml
│   ├── types/
│   │   └── index.ts
│   ├── utils/
│   │   ├── logger.ts
│   │   └── config-loader.ts
│   └── main.ts
├── tests/
│   ├── unit/
│   │   ├── fallback-router.test.ts
│   │   ├── health-tracker.test.ts
│   │   ├── prompt-analyzer.test.ts
│   │   └── session-manager.test.ts
│   ├── integration/
│   │   ├── gateway.test.ts
│   │   └── providers.test.ts
│   └── fixtures/
│       └── mock-responses.ts
├── scripts/
│   ├── generate-fallback-order.js   # Client-side script
│   ├── export-redis-stats.js        # Export health data for client-side
│   └── setup.sh                      # VPS setup script
├── .env.example
├── .gitignore
├── package.json
├── tsconfig.json
├── vitest.config.ts
├── README.md
├── AGENTS.md
└── QUESTIONS.md
```

---

## 5. Testing Requirements

### 5.1 Unit Tests (Vitest)
- `fallback-router.test.ts`: Verify fallback iteration logic skips unhealthy providers
- `health-tracker.test.ts`: Verify decay algorithm correctly transitions status
- `prompt-analyzer.test.ts`: Verify classification accuracy for sample prompts
- `session-manager.test.ts`: Verify session CRUD and context pruning

### 5.2 Integration Tests
- `gateway.test.ts`: Test full request flow (virtual model → fallback → response)
- `providers.test.ts`: Test each provider adapter with mocked HTTP responses

### 5.3 Test Coverage
- Minimum 80% line coverage across `src/` modules
- Critical paths (fallback logic, health tracking decay) MUST have 100% coverage

### 5.4 Mocking Guidelines
- Use `nock` or `viest.fn()` to mock HTTP calls to provider APIs
- Mock Redis responses in unit tests
- Use test fixtures for provider response payloads

---

## 6. Git Protocol

### 6.1 Branch Strategy
- Work on feature branches: `feature