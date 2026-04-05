# AGENTS.md — Dynamic LLM Gateway & Fallback Orchestrator

## 1. Role/Mission

**Mission:** Design, implement, and maintain an intelligent API routing layer that dynamically manages LLM model fallback chains in real-time. This gateway must ensure maximum uptime and optimal performance for agentic coding workflows by continuously monitoring provider health, latency, rate limits, and cost efficiency, then automatically reordering fallback chains to use the best available model at any given moment.

**Core Responsibilities:**

- Build and operate a gateway service that sits between agentic coding agents and LLM providers
- Continuously poll `/v1/models` endpoints to discover available models and their capabilities
- Track real-time metrics: TTFT (Time To First Token), TPS (Tokens Per Second), error rates, and rate limit status
- Implement dynamic fallback chain reordering based on performance, cost, and availability
- Handle provider outages, API key invalidation, and rate limit exhaustion gracefully
- Maintain state in SQLite for provider metrics, model configurations, and historical performance data
- Provide health check endpoints and observability for deployed agents
- Auto-test providers and models on a configurable schedule (default: daily) to detect issues early

**Success Metrics:**

- Uptime: >99.9% for routing requests successfully to some available model
- Latency: Minimum average TTFT across selected primary models
- Fault Tolerance: Zero agentic coding workflow stalls due to gateway-level failures

---

## 2. Technical Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Language** | Python 3.11+ | Rich async ecosystem, excellent LLM/provider library support |
| **Database** | SQLite (via `aiosqlite`) | Lightweight, zero-config, no external dependencies, sufficient for state tracking |
| **HTTP Client** | `httpx` (async) | Modern async HTTP with retry/backoff built-in, connection pooling |
| **Task Scheduling** | `APScheduler` (async-safe) | In-process scheduler for periodic health checks and metric collection |
| **Configuration** | Environment variables + `.env` files | No external config server dependency, simple for free-tier deployment |
| **Logging** | `structlog` | Structured JSON logging for observability |
| **Metrics** | Prometheus-compatible counters/gauges via `prometheus_client` | Free metrics endpoint for GitHub Actions self-hosted runners |
| **Deployment Target** | Docker container (optional) | Portable, runs anywhere including free-tier cloud VMs |
| **Testing** | `pytest` + `pytest-asyncio` + `respx` (mocking) | Free, no external testing infrastructure |

**No External Paid Services Required:** All required dependencies are open-source. Deployment can use free-tier VMs or GitHub Actions self-hosted runners.

---

## 3. Requirements

### Core Infrastructure

1. **Gateway Service Entry Point**
   - Implement `GatewayServer` class that runs an async HTTP server (using `httpx` or `aiohttp`)
   - Expose endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics`
   - Accept OpenAI-compatible request payloads and transform to provider-specific formats

2. **Provider Configuration System**
   - Define configuration schema for providers: API base URL, API key source (env var name), model list, token multipliers
   - Support adding/removing providers via configuration reload without restart
   - Default configuration must include at least 3 providers (e.g., free tiers of OpenAI, Anthropic, Google)

3. **Dynamic Fallback Chain Manager**
   - `FallbackChainManager` class maintains ordered list of provider→model pairs for each virtual model
   - Implement `reorder_for_agent(agent_type)` method that returns optimal chain based on recent metrics
   - Default chains should prioritize: speed (>TPS), then reliability (>uptime), then cost (token multipliers)

4. **Real-Time Metrics Collection**
   - Implement `MetricsCollector` that tracks per-provider-model: TTFT, TPS, error count, rate limit hits, latency histogram
   - Use rolling window (default: 5 minutes) for metric aggregation
   - Store metrics in SQLite with table `provider_metrics(provider, model, timestamp, ttft_ms, tps, errors,RateLimitHits)`

5. **Health Check Scheduler**
   - Implement `ProviderHealthChecker` using APScheduler to run every N minutes (default: 1 minute)
   - For each configured provider: send lightweight request (e.g., `/v1/models` or chat completion with minimal prompt)
   - Record success/failure, latency, and extract rate limit headers
   - Update SQLite table `provider_status(provider, last_check, status, rate_limit_remaining)`

6. **Automatic Model Discovery**
   - Poll `/v1/models` endpoint for each provider to get list of available models
   - Cache model list with TTL (default: 5 minutes)
   - Compare discovered models against configuration to detect new models or missing models

7. **Smart Retry/Backoff Logic**
   - Implement `RetryPolicy` class with exponential backoff and jitter
   - Configurable retry count (default: 3), base delay (default: 1s), max delay (default: 30s)
   - On transient error (5xx, rate limit): retry next in fallback chain, not same provider
   - On permanent error (401, 403, invalid API key): mark provider as disabled, do not retry

8. **Token Quota Tracking**
   - Track estimated token usage per provider per day
   - Implement quota enforcement: if remaining quota