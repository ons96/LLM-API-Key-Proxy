# LLM Gateway with Auto-Discovery and Leaderboard

## 1. Role/Mission

**Mission:** Build and maintain an autonomous LLM gateway system that automatically discovers free LLM providers, aggregates benchmark data from multiple sources, generates ranked fallback lists for agentic coding and reasoning tasks, and provides resilient model routing with built-in failover capabilities.

**Core Objectives:**

- Automatically discover and integrate new free/public LLM API providers through web scraping and API research
- Maintain comprehensive leaderboard data by aggregating benchmarks from publicly available sources (e.g., OpenRouter, LMSYS, HuggingFace)
- Generate and maintain ordered fallback ranking lists (best-to-worst) for agentic coding and reasoning capabilities
- Provide multi-account support for providers to maximize parallel usage limits
- Enable resilient routing through virtual models with automatic failover based on fallback ranking files

**Operational Principles:**

- Only use free resources (no paid services, no API costs where possible)
- Run autonomously on GitHub Actions with scheduled workflows
- Make independent decisions within scope; escalate complex questions to QUESTIONS.md
- Prioritize reliability, transparency, and observability in all operations

---

## 2. Technical Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Language** | Python 3.10+ | Best ecosystem for web scraping, API integration, data processing |
| **HTTP Client** | `httpx` | Async-capable, supports retries, timeout handling |
| **Web Scraping** | `beautifulsoup4` + `playwright` | Lightweight scraping + JS-heavy site handling |
| **Data Storage** | JSON / YAML files | Version-controllable, human-readable ranking files |
| **Scheduling** | GitHub Actions (`schedule` event) | Free cron-like capability for periodic runs |
| **Data Processing** | `pandas` + `pyyaml` | Efficient benchmark aggregation and file handling |
| **Testing** | `pytest` + `pytest-asyncio` | Comprehensive testing framework |
| **Virtualization** | Custom router with fallback chains | Lightweight implementation without heavy infrastructure |

**Environment:** GitHub Actions (Ubuntu runner) with free tier allocation

---

## 3. Requirements (Numbered)

### 3.1 Auto-Discovery of Free LLM Providers

1. **Provider Discovery Module**
   - Implement a scheduled workflow (daily or weekly) that scans designated sources for new free LLM API providers
   - Primary sources: OpenRouter free models list, GitHub trending AI projects, Reddit r/LocalLLaMA, AI benchmarking hubs
   - Use web scraping (BeautifulSoup/Playwright) and API calls to detect new providers
   - Validate provider availability via connectivity tests before adding

2. **Provider Registry**
   - Maintain a `providers.json` registry containing:
     - Provider name, base URL, authentication method (API key, OAuth, etc.)
     - Available models and their capabilities
     - Rate limits, usage quotas, and tos/terms status
     - Health status (active, degraded, unavailable)

3. **Deduplication Logic**
   - Before adding new providers, check against existing registry to avoid duplicates
   - Use fuzzy matching on provider name, API endpoints, and model names

### 3.2 Multi-Account Management

4. **Account Manager**
   - For supported providers, facilitate creation of multiple free accounts/api keys
   - Store credentials securely using GitHub Secrets or encrypted files
   - Track account status (active, rate-limited, banned)

5. **Usage Distribution**
   - Implement round-robin or least-used distribution across accounts for the same provider
   - Include cooldown logic between account switches to avoid abuse detection

6. **Safety Constraints**
   - **DO NOT** attempt multi-account creation for providers known to enforce strict IP detection (Google/Gemini CLI, Antigravity, OpenAI)
   - Document provider-specific terms of service and respect usage policies
   - Implement circuit breakers when abuse signals are detected

### 3.3 Aggregated Leaderboard Creation

7. **Benchmark Aggregator**
   - Fetch benchmark data from publicly available sources:
     - OpenRouter rankings
     - LMSYS Chatbot Arena
     - HuggingFace Open LLM Leaderboard
     - AgentBench / SWE-Bench results
   - Normalize scores across different benchmarks to a common scale (0-100 or 0.0-1.0)

8. **Model Metadata Collection**
   - For each LLM, collect:
     - Model name, provider, version/family
     - Context window size
     - Capabilities (vision, function calling, streaming, etc.)
     - Performance metrics: agentic coding, reasoning, speed (TPS)

9. **Scheduled Updates**
   - Run leaderboard update workflow on schedule (daily or upon detecting new model releases)
   - Include buffer logic: wait N days after new model release before including in leaderboard (to allow benchmarks to appear)
   - Store versioned snapshots of leaderboard data for rollback capability

### 3.4 Fallback Model Ranking Lists

10. **Ranking File Generator**
    - Generate at least two ordered ranking files:
      - `ranking_agentic_coding.json`: Models ordered best-to-worst for coding tasks
      - `ranking_reasoning.json`: Models ordered best-to-worst for reasoning/chat tasks
    - Each entry includes: model ID, provider, aggregated score, confidence level

11. **Virtual Model Routing**
    - Implement virtual model aliases that map to underlying provider + model combinations
    - Primary + fallback chain: When primary fails, route to next model in the fallback chain
    - Generate fallback ranking list per virtual model (`fallbacks/{virtual_model}.json`)

12. **Ranking Persistence**
    - Store ranking files in version-controlled format (JSON/YAML)
    - Include timestamp, data source provenance, and score methodology
    - Support manual override capability (e.g., pinning preferred models)

### 3.5 Gateway Core Functionality

13. **LLM Gateway API**
    - Expose a simple API (e.g., FastAPI or Flask) for routing requests
    - Support: `chat.completions`, `embeddings` endpoints
    - Virtual model resolution: Map request to primary provider and fallback list
    - Automatic failover on timeout or 429/5xx errors

14. **Health Monitoring**
    - Periodic health checks for all registered providers
    - Alert on provider degradation or outage
    - Track latency and throughput metrics per provider

15. **Logging and Observability**
    - Log all requests, responses, errors, and routing decisions
    - Track fallback chain execution and success rates
    - Output structured logs for debugging and analytics

---

## 4. File Structure

```
llm-gateway/
├── .github/
│   └── workflows/
│       ├── auto-discover.yml      # Provider discovery scheduler
│       ├── update-leaderboard.yml # Benchmark aggregation scheduler
│       └── health-check.yml     # Provider health monitoring
├── src/
│   ├── __init__.py
│   ├── gateway/
│   │   ├── __init__.py
│   │   ├── router.py           # Virtual model routing and failover
│   │   ├── api.py             # FastAPI/Flask gateway endpoints
│   │   └── config.py          # Configuration management
│   ├── discover/
│   │   ├── __init__.py
│   │   ├── scraper.py         # Web scraping utilities
│   │   └── provider_finder.py  # Provider detection logic
│   ├── leaderboard/
│   │   ├── __init__.py
│   │   ├── aggregator.py      # Benchmark aggregation
│   │   ├── normalizer.py     # Score normalization
│   │   └── writer.py         # Ranking file generation
│   ├── accounts/
│   │   ├── __init__.py
│   │   ├── manager.py        # Multi-account tracking
│   │   └── rotator.py       # Account switching logic
│   └── utils/
│       ├── __init__.py
│       ├── http.py          # HTTP client with retries
│       └── logging.py       # Structured logging
├── configs/
│   ├── providers.json      # Provider registry
│   └── virtual_models.json # Virtual model definitions
├── rankings/
│   ├── ranking_agentic_coding.json
│   ├── ranking_reasoning.json
│   └── fallbacks/          # Per-virtual-model fallback lists
│       ├── model_alpha.json
│       └── model_beta.json
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_scraper.py
│   │   ├── test_aggregator.py
│   │   └── test_router.py
│   ├── integration/
│   │   ├── test_provider_discovery.py
│   │   └── test_leaderboard_update.py
│   └── fixtures/
│       └── sample_data.json
├── pyproject.toml
├── requirements.txt
├── AGENTS.md
├── QUESTIONS.md
└── README.md
```

---

## 5. Testing Requirements

### 5.1 Unit Tests

| Module | Test Coverage |
|--------|--------------|
| `scraper.py` | HTML parsing, selector logic, pagination handling |
| `aggregator.py` | Benchmark merging, weighted averaging, score normalization |
| `router.py` | Fallback chain traversal, error handling, failover logic |
| `manager.py` | Account state transitions, rate limiting logic |
| `normalizer.py` | Scale conversion, outlier handling, missing data handling |

### 5.2 Integration Tests

1. **Provider Discovery**
   - Test full discovery pipeline against mock sources
   - Validate registry update and deduplication logic

2. **Leaderboard Update**
   - Test benchmark fetching from mock API responses
   - Validate ranking file generation against expected output

3. **Gateway Routing**
   - Test virtual model resolution and failover with mocked provider responses
   - Simulate provider failures and verify fallback behavior

### 5.3 Validation Requirements

- All unit tests must pass before merging PRs
- Integration tests must pass in CI environment
- Ranking files must be valid JSON/YAML with required schema
- Provider registry must pass schema validation

### 5.4 Health Checks

- Scheduled health check workflow must run without errors
- Alerts triggered on provider status changes

---

## 6. Git Protocol

### 6.1 Branch Strategy

- **`main`**: Stable, deployable code only
- **`develop`**: Integration branch for features
- **`feature/*`**: Feature branches (e.g., `feature/auto-discover`, `feature/fallback-ranking`)
- **`hotfix/*`**: Urgent fixes

### 6.2 Commit Convention

```
