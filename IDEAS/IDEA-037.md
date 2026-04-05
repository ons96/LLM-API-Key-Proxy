# Dynamic Provider Response Time Reordering

## 1. Role/Mission

**Role:** Autonomous Coding Agent for Multi-Provider LLM Routing System

**Mission:** Create and maintain an intelligent system that dynamically measures, analyzes, and reorders AI model providers (such as OpenAI, Anthropic, Google, Azure OpenAI, etc.) based on their actual response times. The system must:

1. Continuously monitor response times across all configured providers
2. Apply weighted averaging to emphasize recent performance while retaining historical context
3. Detect anomalies (timeouts, latency spikes, degradation trends)
4. Automatically reorder providers from fastest to slowest expected response speed
5. Persist and cache provider rankings for efficient runtime routing

**Ultimate Goal:** Enable applications to always automatically select the fastest-responding provider for any given model, improving user experience by minimizing wait times while maintaining fallback capability for resilience.

---

## 2. Technical Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| **Language** | Python 3.10+ | Rich ecosystem for HTTP clients, async libraries, and data processing |
| **Async Runtime** | `asyncio` + `aiohttp` | Concurrent provider health checks without thread overhead |
| **HTTP Client** | `aiohttp` or `httpx` (async) | Non-blocking HTTP requests for response time measurement |
| **Data Storage** | SQLite (local file) or JSON file | Free, zero-resource persistence; no external database required |
| **Testing** | `pytest` + `pytest-asyncio` | Industry-standard async testing |
| **CI/CD** | GitHub Actions | Free for public/private repos; native integration |
| **Logging** | `logging` (stdlib) | No additional dependencies |
| **Caching** | In-memory + file persistence | Fast reads with durability |

**Key Dependencies (minimal):**
```txt
aiohttp>=3.9.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
httpx>=0.25.0  # optional, for sync fallback
pydantic>=2.0  # for data validation
```

---

## 3. Requirements (Numbered)

### 3.1 Core Functionality

1. **Provider Configuration Management**
   - Load provider configurations from a JSON/YAML file
   - Support at minimum: endpoint URL, API key (from env vars), model name, timeout settings
   - Allow easy addition/removal of providers without code changes

2. **Response Time Measurement**
   - Execute concurrent health-check requests to each provider
   - Measure time from request initiation to response receipt (excluding auth overhead where possible)
   - Use lightweight payloads (e.g., a short chat completion with minimal tokens) to minimize token-cost impact
   - Configure per-provider timeouts (default: 30 seconds)

3. **Weighted Historical Averaging**
   - Maintain rolling window of response time samples (configurable, default: last 10 measurements)
   - Apply exponential decay: newer samples weighted higher than older ones
   - Formula: `weighted_avg = Σ(sample_time * weight) / Σ(weight)` where `weight = e^(-k * age)` and `k` is decay constant
   - Default decay constant: 0.1 (adjustable via config)

4. **Anomaly Detection**
   - Flag responses exceeding 2x the moving average as anomalies
   - Track consecutive failures/timeouts separately from slow responses
   - Apply lower weight to anomalous samples in ranking calculation
   - Detect sustained degradation (3+ consecutive slow responses)

5. **Dynamic Reordering**
   - Rank providers by weighted average response time (fastest = rank 1)
   - Recalculate rankings after each measurement cycle
   - Persist rankings to local storage for runtime access
   - Provide a runtime API to get ordered provider list for any given model

6. **Caching and Persistence**
   - Cache rankings in memory for fast access
   - Persist to SQLite/JSON on each recalculation
   - Load persisted rankings on startup (warm boot)
   - Handle empty/corrupted persistence gracefully

### 3.2 Resilience Requirements

7. **Graceful Degradation**
   - If a provider fails to respond, mark as unavailable but retain in list for retry
   - Exclude unavailable providers from active ranking until restored
   - Do not block application startup if measurement fails

8. **Rate Limiting Compliance**
   - Respect provider rate limits (track request count per time window)
   - Space out health-check requests to avoid triggering limits
   - Use smallest possible models for measurement

### 3.3 Observability

9. **Logging**
   - Log all measurement cycles with timing details
   - Log ranking changes with before/after values
   - Log anomalies and provider state changes
   - Use appropriate log levels (DEBUG for measurements, INFO for rankings, WARNING for anomalies)

10. **Metrics Export (Optional)**
    - Expose basic Prometheus-style metrics if feasible
    - Include: current weighted avg per provider, last response time, availability status

---

## 4. File Structure

```
dynamic-provider-reranking/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI workflow
├── src/
│   └── provider_rerank/
│       ├── __init__.py
│       ├── config.py              # Configuration loading
│       ├── models.py              # Pydantic data models
│       ├── provider.py             # Provider client (HTTP)
│       ├── metrics.py              # Response time measurement
│       ├── weighter.py             # Weighted averaging logic
│       ├── anomaly.py              # Anomaly detection
│       ├── ranker.py               # Ranking orchestration
│       ├── storage.py               # Persistence (SQLite/JSON)
│       └── main.py                 # CLI entry point
├── tests/
│   ├── __init__.py
│   ├── test_metrics.py
│   ├── test_weighter.py
│   ├── test_anomaly.py
│   ├── test_ranker.py
│   └── test_integration.py
├── providers.json.example         # Example provider config
├── config.yaml.example            # Example system config
├── requirements.txt
├── pyproject.toml
├── README.md
├── AGENTS.md
└── QUESTIONS.md                   # Agent questions file
```

---

## 5. Testing Requirements

### 5.1 Unit Tests

| Module | Test Coverage Required |
|--------|------------------------|
| `weighter.py` | Weighted average calculation with various decay constants; edge cases (empty list, single sample, all same values) |
| `anomaly.py` | Flagging >2x average, detecting 3+ consecutive slow, handling mixed samples |
| `models.py` | Validation of provider configs; serialization/deserialization |
| `storage.py` | Save/load rankings, handle corruption, handle missing file |

### 5.2 Integration Tests

| Scenario | Expected Behavior |
|----------|-------------------|
| Full measurement cycle | All providers contacted, times recorded, rankings updated |
| Provider timeout | Provider marked unavailable, excluded from rankings |
| Startup with persistence | Rankings loaded from storage, system operational |
| Ranking change detected | Log shows old vs new ranking |

### 5.3 Test Execution

- Run all tests on every PR via GitHub Actions
- Target: 80%+ code coverage
- All tests must pass before merging
- Use `pytest` with async support via `pytest-asyncio`

---

## 6. Git Protocol

### 6.1 Branch Strategy

- **Main branch:** `main` - Production-ready code only
- **Development branch:** `dev` - Integration branch for features
- **Feature branches:** `feature