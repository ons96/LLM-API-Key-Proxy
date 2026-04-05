# AGENTS.md - Modular Gateway Architecture

**Project:** LLM-API-Key-Proxy Split
**Purpose:** Architecture design for resource-constrained deployment with separate gateway and analytics backend

---

## 1. Role/Mission

### Primary Mission
Design and implement a modular gateway architecture that splits the existing LLM-API-Key-Proxy into two independent services:

1. **Gateway Service (Lightweight)**
   - Minimal API proxy running on free VPS (Render.com free tier, Oracle Free Tier, or similar)
   - Handles API key rotation and request forwarding only
   - Zero local storage; stateless design
   - Must remain under 512MB RAM usage

2. **Analytics Backend (Heavyweight)**
   - Runs on a separate machine with more compute resources
   - Manages provider updates, model benchmarks, and leaderboards
   - Stores data about API providers, LLM performance metrics
   - Pushes configuration updates to the Gateway via Git or API

### Agent Responsibilities
- Plan and document the split architecture
- Design the communication protocol between services
- Create the initial project structure for both services
- Implement the core gateway with minimal dependencies
- Design the data sync mechanism for provider/leaderboard updates
- Use free CI/CD resources only (GitHub Actions free tier)

---

## 2. Technical Stack

### Gateway Service (VPS-Ready)
| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.11+ | Lightweight, minimal memory footprint |
| Web Framework | FastAPI | Async, low overhead, auto docs |
| Server | Uvicorn (single worker) | Minimal resource usage |
| Deployment | Render Free Tier / Oracle Free | Compatible with free limits |

### Analytics Backend (Full Power)
| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.11+ | Reuse gateway code, consistent stack |
| Web Framework | FastAPI | Unified API design |
| Database | SQLite (dev) / PostgreSQL (prod) | Simple setup, free tier ready |
| Task Queue | APScheduler or Celery | For benchmark scheduling |
| Charts | Chart.js or similar | Lightweight visualization |

### Shared Infrastructure
| Component | Choice | Rationale |
|-----------|--------|-----------|
| Config Format | YAML | Human-readable, versionable |
| API Communication | REST + Webhooks | Simple, no extra deps |
| Version Control | Git + GitHub | Free hosting, Actions ready |
| CI/CD | GitHub Actions | Free tier sufficient |

### Free Tier Constraints to Observe
- **Render.com Free:** 550 MB RAM, 0.1 CPU, sleeps after 15 min
- **Oracle Free:** AMPERE/A1 shape, 24 GB RAM total
- **GitHub Actions:** 2000 mins/month (Linux), 500 MB storage

---

## 3. Requirements

### 3.1 Gateway Service
1. Receive API requests and route to configured providers
2. Perform API key rotation based on configurable strategies
3. Cache provider list in-memory with TTL (no local database)
4. Expose health check endpoint at `/health`
5. Expose metrics endpoint at `/metrics` (request counts, errors)
6. Support hot-reload of provider config via environment variable or file watch
7. Proxy streaming responses from LLM providers
8. Handle auth header injection transparently
9. Log requests minimally (no PII) for debugging
10. Graceful shutdown within 5 seconds

### 3.2 Analytics Backend
1. Store and display provider metadata (name, base URL, auth docs)
2. Record benchmark results per model/provider combination
3. Generate and display simple leaderboard rankings
4. Provide API endpoint for Gateway to fetch latest config
5. Support scheduled benchmark runs (configurable intervals)
6. Export provider config as JSON for Gateway consumption
7. Track historical performance trends
8. Alert on provider failures (configurable threshold)
9. Manual trigger for immediate benchmark update
10. Web UI for viewing dashboards (simple HTML + JS)

### 3.3 Inter-Service Communication
1. Gateway polls Backend for config updates (configurable interval, default 5 min)
2. Backend exposes endpoint `/api/v1/config` returning full provider list
3. Backend exposes endpoint `/api/v1/providers` for provider metadata
4. Format: JSON with versioning for cache invalidation
5. Fallback: If Backend unreachable, Gateway uses last-cached config
6. Authentication: Simple API key header for Gateway-Backend communication

### 3.4 General Requirements
1. All secrets stored in environment variables only
2. No hardcoded API keys in source code
3. Configuration must be declarative (YAML-based)
4. Dockerfiles provided for both services
5. Both services must be runnable locally with one command
6. Documentation in markdown files (README.md for each service)

---

## 4. File Structure

```
llm-api-key-proxy/
├── .github/
│   └── workflows/
│       ├── gateway-ci.yml
│       ├── backend-ci.yml
│       └── deploy-gateway.yml
│
├── gateway/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── config.py           # Config loading
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── proxy.py         # Proxy endpoints
│   │   │   └── health.py       # Health + metrics
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── rotation.py     # Key rotation logic
│   │   │   └── backend.py     # Backend sync client
│   │   └── models/
│   │       ├── __init__.py
│   │       └── types.py        # Pydantic models
│   ├── config/
│   │   └── providers.example.yaml
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── .dockerignore
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_proxy.py
│   │   └── test_rotation.py
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── README.md
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── config.py           # Config loading
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── config.py       # Config export endpoints
│   │   │   ├── benchmarks.py  # Benchmark management
│   │   │   └── ui.py           # Simple dashboard
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── db.py           # Database operations
│   │   │   ├── benchmark.py    # Benchmark runner
│   │   │   └── scheduler.py    # Task scheduling
│   │   ├── models/
│   │       ├── __init__.py
│   │       └── types.py         # Pydantic models
│   │   └── db/
│   │       ├── __init__.py
│   │       └── schema.sql      # SQLite schema
│   ├── config/
│   │   └── settings.example.yaml
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── .dockerignore
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_benchmarks.py
│   │   └── test_routes.py
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── README.md
│
├── docs/
│   ├── architecture.md        # This file's source
│   ├── deployment.md          # VPS deployment guide
│   └── sync-protocol.md      # Gateway-Backend protocol
│
├── .gitignore
├── setup.py                   # Optional: single install
└── AGENTS.md                 # This file
```

---

## 5. Testing Requirements

### 5.1 Gateway Tests
| Test Type | Coverage | Tool |
|-----------|----------|-------|
| Unit | Rotation logic, config parsing | pytest + unittest.mock |
| Integration | Health endpoint, metrics | FastAPI TestClient |
| Contract | Provider response passthrough | requests + responses library |
| Performance | Response time