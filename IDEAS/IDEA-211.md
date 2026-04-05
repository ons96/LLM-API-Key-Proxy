# AGENTS.md

## Free AI Coding Model Router with Load Balancing

---

## 1. Role/Mission

**Role:** Senior Software Architect & Full-Stack Developer

**Mission:** Design and implement an autonomous AI coding tool that intelligently routes requests across multiple free AI API providers to achieve unlimited, rate-limit-free code generation. The system must function as a resilient load balancer, automatically rotating through providers and models to maximize availability and performance without incurring costs.

**Core Objectives:**
- Build a router system that distributes API calls across free providers (gpt4free, OpenRouter, new-api, direct API access)
- Implement intelligent rate limit management with automatic failover
- Select optimal coding models based on performance benchmarks (UC伯克利)
- Create a self-healing system that recovers from provider failures
- Enable high-throughput concurrent code generation for autonomous agents

---

## 2. Technical Stack

### Primary Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Router/LLM Abstraction** | `litellm` v1.x | Unified API for 100+ LLMs, built-in retries, fallbacks |
| **Free API Gateway** | `new-api` | Self-hosted free API server with multiple backends |
| **Alternative Gateway** | `OpenRouter.ai` | Aggregator with free tier, key rotation |
| **API Routing** | Custom Python | Rate limiting, key rotation, health monitoring |
| **HTTP Server** | `FastAPI` | Expose routing as REST API with streaming |
| **Retry Logic** | `tenacity` / `litellm` built-in | Exponential backoff, circuit breaker |
| **Configuration** | `pydantic-settings` + `.env` | Type-safe config management |
| **Rate Limiting** | `slowapi` + Redis | Per-client/request rate limiting |
| **Monitoring** | Prometheus + Grafana | Metrics, alerting, dashboard |

### Model Sources (Priority Order)

```
Tier 1 - High Performance Free:
├── OpenRouter (free tier credits)
├── new-api (self-hosted or public)
└── Together AI (free tier)

Tier 2 - Reliable Free:
├── Cohere (free tier)
├── Anthropic (free API - limited)
└── Google AI Studio (free tier)

Tier 3 - Supplementary:
├── Grok (x.ai free tier)
└── Perplexity (free tier)
```

### Benchmark Models (from UGI Leaderboard)

| Model | Type | Use Case | Source |
|-------|------|----------|--------|
| Qwen 2.5 72B | Coding/Reasoning | Primary code generation | OpenRouter |
| Llama 3.3 70B | General coding | Fallback code tasks | OpenRouter |
| Gemini Flash 2.0 | Fast inference | Quick code reviews | Google AI |
| Claude Haiku 3.5 | Fast coding | Lightweight tasks | Anthropic |
| Qwen 3 32B | Small efficient | Resource-constrained | new-api |
| GPT-4o Mini | Code completion | Microsoft AI |

---

## 3. Requirements

### Functional Requirements

1. **API Router Engine**
   - Implement a `Router` class that cycles through configured API keys
   - Track rate limits per provider and switch when approached
   - Support streaming responses for code generation
   - Handle both `/chat/completions` and `/completions` endpoints

2. **Rate Limit Management**
   - Monitor response headers for rate limit remaining
   - Implement sliding window rate limiting per provider
   - Preemptively rotate keys before hitting limits
   - Maintain per-model rate limit configuration

3. **Health Monitoring**
   - Track success/failure rates per provider
   - Implement circuit breaker pattern (3 failures → trip)
   - Auto-recover broken providers after cooldown period
   - Log all API calls with timing and status

4. **Model Selection**
   - Implement model routing based on task type (coding vs general)
   - Allow priority ranking of models in config
   - Support model fallbacks per request type

5. **Error Handling**
   - Graceful degradation when providers fail
   - Retry with exponential backoff (3 attempts default)
   - Return meaningful error messages with fallback suggestions

### Non-Functional Requirements

6. **Performance**
   - Target