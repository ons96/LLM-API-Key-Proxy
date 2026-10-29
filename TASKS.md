# LLM-API-Key-Proxy Tasks

**Last Updated:** 2026-02-24
**Source:** PLAN.md

---

## Phase 1: Hardening (Critical)

### 1.1 Testing Infrastructure
- [ ] Create `tests/` directory structure
- [ ] Add pytest configuration to `pyproject.toml` or `pytest.ini`
- [ ] Write unit tests for `router_core.py` fallback logic
- [ ] Write unit tests for `RotatingClient` retry/rotation
- [ ] Write integration tests for Groq provider
- [ ] Write integration tests for G4F provider
- [ ] Write integration tests for Gemini provider
- [ ] Write end-to-end tests for `coding-elite` virtual model
- [ ] Write end-to-end tests for `chat-fast` virtual model
- [ ] Add GitHub Actions workflow for test automation
- [ ] Configure test coverage reporting

### 1.2 Error Handling & Logging
- [ ] Implement structured JSON logging with request IDs
- [ ] Add error alerting (email/webhook) for critical failures
- [ ] Create request/response logging toggle per-endpoint
- [ ] Implement graceful degradation when all providers fail
- [ ] Create `/health` endpoint with provider status
- [ ] Document logging configuration options

### 1.3 Security Hardening
- [ ] Implement rate limiting per API key
- [ ] Create API key rotation mechanism
- [ ] Add audit logging for sensitive operations
- [ ] Configure HTTPS on VPS (certbot/letsencrypt)
- [ ] Add input validation for `/v1/chat/completions`
- [ ] Add input validation for `/v1/responses`
- [ ] Security audit of `.env` handling

---

## Phase 2: Dynamic Model Rankings (High)

### 2.1 Data Collection Pipeline
- [ ] Research Artificial Analysis API access
- [ ] Implement Artificial Analysis API client
- [ ] Research UGI Leaderboard scraping approach
- [ ] Implement UGI Leaderboard scraper
- [ ] Research OpenRouter API for model statistics
- [ ] Implement OpenRouter API client
- [ ] Research LiveCodeBench integration
- [ ] Implement LiveCodeBench client
- [ ] Research Arena-Hard integration
- [ ] Implement Arena-Hard client
- [ ] Create GitHub Actions workflow for daily data refresh
- [ ] Design data storage schema for benchmark data

### 2.2 Scoring Algorithm
- [ ] Implement `coding_score()` function
- [ ] Implement `chat_score()` function
- [ ] Implement `rp_score()` function
- [ ] Create SQLite/JSON model ranking database
- [ ] Build auto-update script for virtual model fallback chains
- [ ] Create A/B testing framework for ranking accuracy
- [ ] Document scoring formulas in code comments

---

## Phase 3: Monitoring & Alerting (High)

### 3.1 Uptime Monitoring
- [ ] Research uptime monitoring options (UptimeRobot, Pingdom, etc.)
- [ ] Set up uptime monitoring for gateway
- [ ] Create status page (public or private)
- [ ] Configure alerts for provider failures
- [ ] Implement response time tracking per provider
- [ ] Create provider health dashboard

### 3.2 Usage Analytics
- [ ] Design usage tracking schema
- [ ] Implement request volume tracking per virtual model
- [ ] Implement provider usage distribution tracking
- [ ] Build cost estimation for non-free providers
- [ ] Create usage analytics dashboard
- [ ] Export metrics to Prometheus/Grafana (optional)

---

## Phase 4: New Features (Medium)

### 4.1 Response Caching
- [ ] Set up Redis server (Docker or VPS)
- [ ] Design cache key generation strategy
- [ ] Implement caching middleware
- [ ] Add TTL-based cache invalidation
- [ ] Implement cache hit/miss metrics
- [ ] Document caching configuration

### 4.2 Streaming Improvements
- [ ] Research SSE implementation for FastAPI
- [ ] Implement SSE streaming endpoint
- [ ] Add WebSocket support for bidirectional communication
- [ ] Implement stream interruption handling
- [ ] Implement partial response caching
- [ ] Test streaming with all providers

### 4.3 Additional Endpoints
- [ ] Implement `/v1/embeddings` endpoint
- [ ] Implement `/v1/completions` (legacy) endpoint
- [ ] Implement `/v1/moderations` endpoint
- [ ] Research `/v1/assistants` feasibility
- [ ] Update OpenAPI documentation

---

## Phase 5: New Providers (Medium)

### 5.1 Free Tier Providers
- [ ] Research Cohere free tier API
- [ ] Implement Cohere provider adapter
- [ ] Research Replicate free tier API
- [ ] Implement Replicate provider adapter
- [ ] Research Hugging Face Inference API
- [ ] Implement Hugging Face provider adapter
- [ ] Expand Cloudflare Workers AI integration
- [ ] Test all new providers with fallback chain

### 5.2 Paid Providers (Optional)
- [ ] Research Anthropic Claude direct API integration
- [ ] Implement Anthropic provider adapter
- [ ] Research OpenAI direct API integration
- [ ] Implement OpenAI provider adapter
- [ ] Research Google Vertex AI integration
- [ ] Implement Vertex AI provider adapter
- [ ] Research AWS Bedrock integration
- [ ] Implement Bedrock provider adapter

---

## Phase 6: Infrastructure (Low)

### 6.1 Containerization
- [ ] Optimize Dockerfile with multi-stage build
- [ ] Create Kubernetes manifests
- [ ] Create Helm chart
- [ ] Update Docker Compose for local dev with Redis
- [ ] Test container deployment on VPS

### 6.2 CI/CD Pipeline
- [ ] Create GitHub Actions workflow for testing
- [ ] Create GitHub Actions workflow for Docker builds
- [ ] Create GitHub Actions workflow for VPS deployment
- [ ] Implement rollback mechanism on failure
- [ ] Document CI/CD process

---

## Phase 7: Documentation (Low)

### 7.1 User Documentation
- [ ] Generate OpenAPI spec from FastAPI
- [ ] Create API reference documentation
- [ ] Write getting started guide
- [ ] Write provider configuration guide
- [ ] Expand troubleshooting guide

### 7.2 Developer Documentation
- [ ] Create ADR template
- [ ] Write first ADR (architecture decisions)
- [ ] Write contributing guide
- [ ] Write provider adapter development guide
- [ ] Write testing guide

---

## Bugs to Fix

From BUGS.md:
- [ ] Investigate and resolve dormant router logic in main.py
- [ ] Document which G4F model IDs are known to work
- [ ] Add validation for G4F model ID complexity

---

## Quick Wins (Can do anytime)

- [ ] Add `.env.example` comments for each variable
- [ ] Add more provider status logging
- [ ] Improve error messages for provider failures
- [ ] Add request timing to logs
- [ ] Create a Makefile for common commands
- [ ] Add pre-commit hooks for linting

---

## Progress Tracking

| Phase | Total Tasks | Completed | Progress |
|-------|-------------|-----------|----------|
| 1.1 Testing | 11 | 0 | 0% |
| 1.2 Logging | 6 | 0 | 0% |
| 1.3 Security | 7 | 0 | 0% |
| 2.1 Data Collection | 12 | 0 | 0% |
| 2.2 Scoring | 7 | 0 | 0% |
| 3.1 Uptime | 6 | 0 | 0% |
| 3.2 Analytics | 6 | 0 | 0% |
| 4.1 Caching | 6 | 0 | 0% |
| 4.2 Streaming | 6 | 0 | 0% |
| 4.3 Endpoints | 5 | 0 | 0% |
| 5.1 Free Providers | 8 | 0 | 0% |
| 5.2 Paid Providers | 8 | 0 | 0% |
| 6.1 Containerization | 5 | 0 | 0% |
| 6.2 CI/CD | 5 | 0 | 0% |
| 7.1 User Docs | 5 | 0 | 0% |
| 7.2 Dev Docs | 5 | 0 | 0% |
| **Total** | **114** | **0** | **0%** |
