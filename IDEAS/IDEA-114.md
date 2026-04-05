# AGENTS.md — API Provider Live Response Time Tester

---

## 1. Role / Mission

**Mission:** Implement a performance testing module within the LLM gateway that programmatically probes multiple API providers (OpenAI, Anthropic, Ollama, local models, etc.), measures live latency metrics—**total response time**, **time-to-first-token (TTFT)**, and **tokens-per-second throughput**—and persists the results to a local SQLite database. This data will power an intelligent routing layer that selects the optimal provider for each request based on real-time performance.

**Key Responsibilities:**
- Design and implement HTTP timing instrumentation that intercepts both request start and streaming token arrival.
- Implement TTFT measurement by detecting the first non-empty chunk in a streaming response.
- Calculate throughput (tokens/sec) by counting delimiter-split tokens over the streaming duration.
- Build a persistence layer (SQLite) to store metrics with timestamps, provider IDs, model names, and status codes.
- Provide a query API to retrieve recent performance stats for routing decisions.

---

## 2. Technical Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Language** | Python 3.11+ | Required for async streaming, widely used in LLM gateways |
| **HTTP Client** | `httpx` (async) | Native async, supports streaming responses, precise timing |
| **Web Framework** | `FastAPI` | Lightweight, easy to embed as internal route |
| **Database** | SQLite (via `sqlalchemy` + `aiosqlite`) | Free, zero-config, supports async writes |
| **Data Model** | Pydantic v2 | Validation and schema definition |
| **Testing** | `pytest` + `pytest-asyncio` | Standard async testing |
| **Mocking** | `aioresponses` | Mock HTTP responses for unit tests |
| **Timing** | `time.perf_counter()` | Sub-microsecond resolution for latency |

**No paid services or external APIs required.** All testing uses locally configurable mock endpoints or self-hosted models.

---

## 3. Requirements (Numbered)

### 3.1 Core Metrics Collection
1. **Total Response Time**: Measure wall-clock time from HTTP request initiation (`httpx.AsyncClient.request()`) to complete response body receipt or stream termination.
2. **Time-to-First-Token (TTFT)**: Capture the elapsed time from request initiation to receipt of the first meaningful token chunk (non-empty, after any leading whitespace/buffer).
3. **Tokens Per Second (Throughput)**: Calculate as `total_tokens / total_streaming_duration`. For chunked responses, count tokens by splitting on provider-specific delimiters (e.g., `"data: "` parsing for SSE, or counting `\n`-split segments).
4. **Success/Failure Status**: Record HTTP status code and any error metadata (timeout, connection error, rate-limit).

### 3.2 Provider Configuration
5. Define a **Provider Registry** (YAML or JSON file) where each provider entry specifies:
   - `name`: Display name (e.g., `"openai-gpt-4"`)
   - `base_url`: Full endpoint URL
   - `auth_header`: API key or Bearer token
   - `model`: Model identifier to request
   - `stream`: Boolean indicating streaming mode preference
   - `timeout`: Request timeout in seconds
6. Support both **streaming** and **non-streaming** modes; default to streaming for throughput metrics.

### 3.3 Data Persistence
7. Use SQLite with schema:
   ```sql
   CREATE TABLE performance_logs (
       id INTEGER PRIMARY KEY,
       timestamp TEXT NOT NULL,
       provider TEXT NOT NULL,
       model TEXT NOT NULL,
       total_response_ms REAL,
       ttft_ms REAL,
       tokens_per_sec REAL,
       total_tokens INTEGER,
       status_code INTEGER,
       error_detail TEXT
   );
   ```
8. Implement async insert using `aiosqlite`.
9. Add an index on `(provider, timestamp)` for fast retrieval of recent logs.

### 3.4 Query API for Routing
10. Add a FastAPI endpoint `GET /routing/stats?provider