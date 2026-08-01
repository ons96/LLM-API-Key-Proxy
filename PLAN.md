# LLM-API-Key-Proxy Development Plan

**Created:** 2026-02-24
**Status:** Active Development
**Based on:** VIRTUAL_MODELS_PLAN.md, NEXT_STEPS.md, BUGS.md

---

## Current State

The project is **VPS deployed and functional** with:
- 7 virtual models with automatic fallback
- 16+ provider integrations
- OpenCode integration working
- Free-only mode operational

---

## Phase 1: Hardening (Priority: Critical)

### 1.1 Testing Infrastructure
**Goal:** Comprehensive automated test suite

- [ ] Set up pytest with proper configuration
- [ ] Unit tests for router_core.py fallback logic
- [ ] Unit tests for RotatingClient retry/rotation
- [ ] Integration tests for all providers
- [ ] End-to-end tests for virtual models
- [ ] CI pipeline runs tests on every PR

**Impact:** Catches regressions, enables confident refactoring

### 1.2 Error Handling & Logging
**Goal:** Production-grade observability

- [ ] Structured JSON logging with request IDs
- [ ] Error alerting (email/webhook on critical failures)
- [ ] Request/response logging toggle per-endpoint
- [ ] Graceful degradation when all providers fail
- [ ] Health check endpoint with provider status

**Impact:** Faster debugging, better incident response

### 1.3 Security Hardening
**Goal:** Protect API keys and prevent abuse

- [ ] Rate limiting per API key
- [ ] API key rotation mechanism
- [ ] Audit logging for sensitive operations
- [ ] HTTPS enforcement (currently HTTP on VPS)
- [ ] Input validation for all endpoints

**Impact:** Prevents abuse, protects user data

---

## Phase 2: Dynamic Model Rankings (Priority: High)

### 2.1 Data Collection Pipeline
**Goal:** Automate benchmark data ingestion

From VIRTUAL_MODELS_PLAN.md:
- [ ] Artificial Analysis API integration (latency, quality scores)
- [ ] UGI Leaderboard scraping (roleplay quality)
- [ ] OpenRouter model statistics
- [ ] LiveCodeBench integration (coding benchmarks)
- [ ] Arena-Hard integration (chat benchmarks)
- [ ] GitHub Actions workflow for daily data refresh

**Impact:** Model rankings stay current automatically

### 2.2 Scoring Algorithm
**Goal:** Implement weighted scoring from VIRTUAL_MODELS_PLAN.md

Formulas to implement:
```
coding_score = 0.4 * LiveCodeBench + 0.3 * Arena-Hard + 0.2 * AA_Quality + 0.1 * AA_Speed
chat_score = 0.5 * Arena-Hard + 0.3 * AA_Quality + 0.2 * AA_Speed
rp_score = 0.6 * UGI + 0.2 * Arena-Hard + 0.2 * AA_Quality
```

- [ ] Implement scoring functions
- [ ] Create model ranking database
- [ ] Auto-update virtual model fallback chains
- [ ] A/B testing for ranking accuracy

**Impact:** Virtual models always use best available models

---

## Phase 3: Monitoring & Alerting (Priority: High)

### 3.1 Uptime Monitoring
**Goal:** Know when the gateway is down

- [ ] Uptime robot or similar integration
- [ ] Status page (public or private)
- [ ] Alert on provider failures (not just gateway)
- [ ] Response time tracking per provider

**Impact:** Proactive incident detection

### 3.2 Usage Analytics
**Goal:** Understand how the gateway is used

- [ ] Request volume tracking per virtual model
- [ ] Provider usage distribution
- [ ] Cost estimation (for non-free providers)
- [ ] User dashboard (if multi-tenant)

**Impact:** Data-driven optimization

---

## Phase 4: New Features (Priority: Medium)

### 4.1 Response Caching
**Goal:** Reduce latency and provider usage

- [ ] Redis integration for caching
- [ ] Cache key generation (model + messages hash)
- [ ] TTL-based cache invalidation
- [ ] Cache hit/miss metrics

**Impact:** Faster responses, lower costs

### 4.2 Streaming Improvements
**Goal:** Better real-time response handling

- [ ] Server-Sent Events (SSE) for streaming
- [ ] WebSocket support for bidirectional communication
- [ ] Stream interruption handling
- [ ] Partial response caching

**Impact:** Better UX for long responses

### 4.3 Additional Endpoints
**Goal:** OpenAI API parity

- [ ] `/v1/embeddings` endpoint
- [ ] `/v1/completions` (legacy) endpoint
- [ ] `/v1/moderations` endpoint
- [ ] `/v1/assistants` endpoint (if feasible)

**Impact:** Broader compatibility with OpenAI tools

---

## Phase 5: New Providers (Priority: Medium)

### 5.1 Free Tier Providers
**Goal:** Expand free options

- [ ] Cohere free tier
- [ ] Replicate free tier
- [ ] Hugging Face Inference API
- [ ] Cloudflare Workers AI (expand current)

**Impact:** More fallback options, better reliability

### 5.2 Paid Providers (Optional)
**Goal:** Premium options for users with API keys

- [ ] Anthropic Claude direct API
- [ ] OpenAI GPT-4 direct API
- [ ] Google Vertex AI
- [ ] AWS Bedrock

**Impact:** Access to best models when needed

---

## Phase 6: Infrastructure (Priority: Low)

### 6.1 Containerization
**Goal:** Easy deployment anywhere

- [ ] Optimize Dockerfile (multi-stage build)
- [ ] Kubernetes manifests
- [ ] Helm chart
- [ ] Docker Compose for local dev with Redis

**Impact:** Portable, scalable deployment

### 6.2 CI/CD Pipeline
**Goal:** Automated deployment

- [ ] GitHub Actions for testing
- [ ] Automated Docker builds
- [ ] Automated VPS deployment
- [ ] Rollback on failure

**Impact:** Faster, safer releases

---

## Phase 7: Documentation (Priority: Low)

### 7.1 User Documentation
- [ ] API reference (OpenAPI spec)
- [ ] Getting started guide
- [ ] Provider configuration guide
- [ ] Troubleshooting guide

### 7.2 Developer Documentation
- [ ] Architecture decision records (ADRs)
- [ ] Contributing guide
- [ ] Provider adapter development guide
- [ ] Testing guide

---

## Dependencies

| Phase | Depends On | Blocking Issues |
|-------|------------|-----------------|
| Phase 2.1 | None | API access to benchmark sources |
| Phase 3.1 | Phase 1.2 | None |
| Phase 4.1 | Phase 3.1 | Redis server |
| Phase 6.2 | Phase 1.1 | None |

---

## Open Questions

1. **Refresh interval for model rankings?** Daily? Hourly? On-demand?
2. **How to handle rate limits across multiple users?** Per-key or global?
3. **Should we support custom virtual models per user?**
4. **What's the upgrade path from HTTP to HTTPS on VPS?**
5. **Should we add a usage-based billing layer?**

---

## Success Metrics

| Metric | Current | Target | Deadline |
|--------|---------|--------|----------|
| Test coverage | ~0% | 80% | Q2 2026 |
| Uptime | Unknown | 99.9% | Q1 2026 |
| Response time (P50) | Unknown | <2s | Q1 2026 |
| Provider count | 16 | 25 | Q2 2026 |
| Virtual models | 7 | 10 | Q2 2026 |

---

## References

- [VIRTUAL_MODELS_PLAN.md](./VIRTUAL_MODELS_PLAN.md) - Detailed scoring formulas
- [BUGS.md](./BUGS.md) - Known issues to fix
- [CODEBASE.md](./CODEBASE.md) - Architecture reference
