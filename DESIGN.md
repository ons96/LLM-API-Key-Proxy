# Dynamic Fallback System - Design Specification

## Overview

The LLM API Proxy uses a dynamic, data-driven fallback system that automatically routes requests through the best available providers based on quality, performance, and availability metrics.

## Core Principles

1. **Quality First for Virtual Models**: Prioritize best agentic coding models
2. **Performance Second**: Within quality tier, use fastest providers
3. **Availability Always**: Skip rate-limited/down providers automatically
4. **Data Driven**: All decisions based on real telemetry data
5. **Self Healing**: Providers become available again after reset

## Scoring Formulas

### Virtual Models (coding-elite, coding-smart, coding-fast, chat-smart, chat-fast)

Use when user wants "best overall" for a task (virtual models)

**Score Formula:**
```
Score = (AgenticScore × 0.60) + (TPS_Score × 0.30) + (AvailabilityScore × 0.10)
```

**Components:**

1. **AgenticScore (60% weight)**
   - Source: SWE-bench / HumanEval benchmark scores
   - File: `config/model_rankings.yaml`
   - Range: 0-100 (normalized to 0-1 for formula)
   - Example:
     - Claude Opus 4.5: 74.4 → 0.744
     - Gemini 3 Pro: 74.2 → 0.742
     - GPT-5.2: 71.8 → 0.718

2. **TPS_Score (30% weight)**
   - Source: Real telemetry measurement (tokens_per_second)
   - Calculation: measured_TPS / max_observed_TPS
   - Range: 0-1
   - Example:
     - Groq: 1050 TPS / 3000 max = 0.35
     - Cerebras: 2500 TPS / 3000 max = 0.83

3. **AvailabilityScore (10% weight)**
   - Source: Health monitoring + rate limit status
   - Calculation: (1 - failure_rate) × (1 - rate_limit_penalty)
   - Range: 0-1
   - Example:
     - Healthy, no rate limits: (1-0.02) × (1-0) = 0.98
     - Rate limited: (1-0.02) × (1-1) = 0.0

**Example Calculation:**
```
Groq/llama-3.3-70b-versatile:
  AgenticScore: 0.652 (SWE-bench 65.2)
  TPS_Score: 0.35 (1050/3000)
  AvailabilityScore: 0.98 (2% failure, no rate limits)

Score = (0.652 × 0.6) + (0.35 × 0.3) + (0.98 × 0.10)
     = 0.3912 + 0.105 + 0.098
     = 0.594

Cerebras/llama-3.3-70b:
  AgenticScore: 0.652
  TPS_Score: 0.83 (2500/3000)
  AvailabilityScore: 0.99

Score = (0.652 × 0.6) + (0.83 × 0.3) + (0.99 × 0.10)
     = 0.3912 + 0.249 + 0.099
     = 0.740 ✅ HIGHER - would be tried first
```

### Specific Models (gemini-3-pro, gpt-4o, gpt-4, etc.)

Use when user explicitly requests a specific model

**Score Formula:**
```
Speed = (TPS_Score × 0.70) + (AvailabilityScore × 0.30)
```

**Components:**

1. **TPS_Score (70% weight)**
   - Same as above
   - PRIMARY factor for specific models

2. **AvailabilityScore (30% weight)**
   - Same as above

**No AgenticScore** because model is fixed by user request

**Example Calculation (user requests gemini-3-pro):**
```
ZenMux/gemini-3-pro:
  TPS_Score: 0.90 (2700/3000)
  AvailabilityScore: 0.95

Score = (0.90 × 0.7) + (0.95 × 0.3)
     = 0.63 + 0.285
     = 0.915 ✅ First choice

Google/gemini-3-pro:
  TPS_Score: 0.60 (1800/3000)
  AvailabilityScore: 0.85

Score = (0.60 × 0.7) + (0.85 × 0.3)
     = 0.42 + 0.255
     = 0.675 ✅ Second choice
```

## Metrics Tracked

### Per Provider+Model Combination

| Metric | Type | Reset Frequency | Purpose |
|--------|------|-----------------|---------|
| Agentic Coding Score | Static | N/A | Model quality ranking |
| Tokens Per Second (TPS) | Dynamic | Hourly (moving avg) | Performance ranking |
| Latency P50/P95/P99 | Dynamic | Hourly | Performance profiling |
| Time To First Token (TTFT) | Dynamic | Hourly | Responsiveness |
| Success Rate | Dynamic | Daily | Reliability |
| Failure Rate | Dynamic | Daily | Unreliability penalty |
| RPM (Requests Per Minute) | Dynamic | Per minute | Rate limit detection |
| Daily Request Count | Dynamic | Daily (midnight) | Usage tracking |
| Rate Limit Status | Dynamic | Per minute | Immediate skip |
| Health Status | Dynamic | Per 5 min | Provider availability |
| Last Successful Request | Timestamp | Never | Health check reference |

### Auto-Skip Conditions

Provider+Model is **SKIPPED** if ANY of these are true:

1. **Rate Limited**
   - Detected 429 response
   - Active rate limit with known reset time
   - Reset: When rate limit window expires (usually 1 minute)

2. **Usage Limit Exceeded**
   - Daily quota exhausted
   - Monthly quota exhausted
   - Reset: Daily at midnight UTC, monthly at month start

3. **Unhealthy Provider**
   - Health check failed (timeout/connection error)
   - Recent failure rate > 50%
   - Reset: On successful health check or successful API call

4. **Down Provider**
   - Provider's API is unreachable
   - 5+ consecutive connection failures
   - Reset: On successful connection

## Fallback Decision Process

### When Virtual Model Requested (e.g., coding-elite)

```
1. Load model config from config/virtual_models.yaml
   - Get list of candidate models (static)
   - Example: [claude-opus-4.5, gemini-3-pro, gpt-5.2, gpt-4, llama-3.3-70b]

2. For each model, find all available providers
   - Query telemetry for provider+model combinations
   - Filter out: rate-limited, unhealthy, down providers

3. Calculate score for each valid combination
   - Use: Score = Agentic(60%) + TPS(30%) + Availability(10%)
   - Sort by score descending

4. Attempt requests in order
   - Try highest score
   - On failure/backoff: try next
   - On success: return result

5. Update metrics after each request
   - Record TPS, latency, success/failure
   - Update rate limit tracking
   - Trigger health check if needed
```

### When Specific Model Requested (e.g., gemini-3-pro)

```
1. Find all providers for requested model
   - Query provider configs
   - Example: [ZenMux, Google, iFlow, OpenRouter]

2. Filter out unavailable providers
   - Rate-limited, unhealthy, down

3. Calculate score for each valid provider
   - Use: Score = TPS(70%) + Availability(30%)
   - Sort by score descending (fastest first per user requirement)

4. Attempt requests in order
   - Same process as virtual model
```

## Telemetry Data Model

```sql
-- API call tracking (existing, to be extended)
CREATE TABLE api_calls (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    error_reason TEXT,
    response_time_ms INTEGER,
    time_to_first_token_ms INTEGER,
    tokens_per_second REAL,
    input_tokens INTEGER,
    output_tokens INTEGER
);

-- Rate limit tracking (NEW)
CREATE TABLE rate_limits (
    id INTEGER PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    limit_type TEXT NOT NULL, -- 'rpm', 'daily', 'monthly'
    current_count INTEGER DEFAULT 0,
    limit_limit INTEGER,
    reset_time DATETIME,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Provider health tracking (NEW)
CREATE TABLE provider_health (
    id INTEGER PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT,
    is_healthy BOOLEAN DEFAULT true,
    last_check_time DATETIME,
    failure_rate REAL DEFAULT 0,
    consecutive_failures INTEGER DEFAULT 0,
    last_success_time DATETIME
);

-- TPS aggregation (NEW)
CREATE TABLE tps_metrics (
    id INTEGER PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    tps REAL NOT NULL,
    window_minutes INTEGER NOT NULL, -- 1, 5, 15, 60
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## Scheduled Tasks

| Task | Frequency | Purpose |
|------|-----------|---------|
| Health Check | Every 5 min | Ping all providers, update health status |
| Rate Limit Reset | Per minute | Clear expired rate limits |
| Daily Reset | Midnight UTC | Reset daily counters, aggregate metrics |
| TPS Recalculation | Every hour | Update moving averages |
| Ranking Update | Every 5 min | Re-calculate provider scores |

## Configuration Examples

### Virtual Model Config (simplified, runtime ordering)

```yaml
# config/virtual_models.yaml
virtual_models:
  coding-elite:
    description: "Best agentic coding - dynamically ordered"
    model_candidates:
      - claude-opus-4-5      # Quality: 74.4 (best)
      - gemini-3-pro         # Quality: 74.2
      - gpt-5-2              # Quality: 71.8
      - claude-sonnet-4-5    # Quality: 70.6
      - gpt-4                # Quality: 68.5
      - gemini-1.5-pro       # Quality: 67.2
      - llama-3.3-70b        # Quality: 65.2
    # Runtime ordering: quality → (same Q) TPS → availability
```

### Scoring Weights Config

```yaml
# config/scoring_config.yaml
scoring:
  virtual_models:
    weights:
      agentic_score: 0.60
      tps_score: 0.30
      availability_score: 0.10

  specific_models:
    weights:
      tps_score: 0.70
      availability_score: 0.30

  availability:
    consecutive_failures_threshold: 5
    failure_rate_threshold: 0.50
    health_check_interval_seconds: 300
```

## Implementation Order

1. Extend telemetry schema with rate_limits, provider_health, tps_metrics
2. Implement health monitor (background task)
3. Implement rate limit tracker (middleware)
4. Create scoring engine (score_engine.py)
5. Integrate into router_core.py
6. Update virtual model fallback logic
7. Add telemetry endpoints for monitoring
8. Testing with simulated failures
9. Documentation and deployment

## Success Criteria

- All metrics tracked accurately
- Providers auto-skip when rate-limited/down
- Ranking updates dynamically based on performance
- No manual intervention needed
- Providers become available again after reset
- Performance overhead < 5ms per request
