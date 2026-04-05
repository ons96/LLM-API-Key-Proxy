# AGENTS.md: Dynamic Model Selection by Response Time

## 1. Role/Mission

**Mission:** Implement an intelligent model selection system that automatically chooses the optimal AI model for each task based on estimated response time performance. The system must balance speed vs. capability by considering token throughput, first-token latency, rate limits, and task complexity requirements.

**Autonomous Agent Directives:**
- Work independently without human intervention unless a blocking issue arises
- Use only free resources (no paid APIs, services, or infrastructure)
- Save any questions requiring human input to `QUESTIONS.md`
- Make architectural decisions using best practices
- Prioritize correctness over speed of implementation

---

## 2. Technical Stack

**Core Technologies:**
- **Language:** Python 3.10+
- **Runtime:** GitHub Actions (ubuntu-latest runner)
- **Package Manager:** pip with requirements.txt

**External Dependencies (Free Tier Compatible):**
- `aiohttp` - Async HTTP requests for API calls
- `asyncio` - Async/await support for concurrent operations
- `httpx` - Synchronous HTTP client
- `pydantic` - Data validation and settings management

**Mock/Testing Dependencies:**
- `pytest` - Testing framework
- `pytest-asyncio` - Async test support
- `pytest-mock` - Mocking utilities
- `freezegun` - Time mocking for rate limit tests

**API Compatibility:**
- OpenAI API format (compatible with open-source alternatives)
- Anthropic API format for Claude models
- LiteLLM unified API interface (for model abstraction)

---

## 3. Requirements (Numbered)

### 3.1 Latency Measurement Module

1. Implement `LatencyMonitor` class to measure:
   - Time to first token (TTFT)
   - Time per output token (TPOT)
   - Total response time (TTFT + tokens × TPOT)

2. Create `record_latency(response_data)` method that:
   - Accepts response metadata (timestamps, token counts)
   - Stores measurements in rolling window (configurable, default: 100 samples)
   - Calculates running averages (simple, weighted, exponential moving average)

3. Implement warm-up queries to establish baseline latency before active selection

### 3.2 Token Speed Estimation

4. Build `TokenSpeedEstimator` class with methods:
   - `estimate_tpots()` - Returns estimated tokens per second
   - `estimate_ttft()` - Returns estimated first token time in seconds
   - `estimate_total_time(expected_tokens)` - Combines for total estimate

5. Implement sliding window statistics:
   - Simple Moving Average (SMA)
   - Exponential Moving Average (EMA) for recency bias
   - Median for outlier resistance

6. Create ability to model speed degradation over context length

### 3.3 Usage/Rate Limit Tracking

7. Implement `RateLimitTracker` class to:
   - Track requests per minute (RPM) per model
   - Track tokens per minute (TPM) per model
   - Monitor proximity to rate limits
   - Calculate projected slowdown based on current usage

8. Create `get_adjusted_speed(model_id, expected_tokens)` method that:
   - Returns speed multiplier based on current usage vs. limits
   - Accounts for approaching limit warnings

9. Implement exponential backoff prediction for near-limit scenarios

### 3.4 Performance Model

10. Build `PerformanceModel` class that combines:
    - Latency measurements
    - Token speed estimates
    - Rate limit status
    - Historical performance data

11. Implement `predict_response_time(model_id, expected_tokens, task_complexity)` method returning:
    - Estimated total time in seconds
    - Confidence score
    - Risk level (low/medium/high)

12. Create task complexity estimator based on:
    - Estimated input token count
    - Task type classification (simple Q&A, coding, analysis, creative)
    - Historical task performance

### 3.5 Model Selection Engine

13. Implement `ModelSelector` class with selection criteria:
    - Minimum required capability (configurable per task)
    - Maximum acceptable response time
    - Balance between speed and quality

14. Create selection strategies:
    - `select_fastest()` - Choose fastest within capability threshold
    - `select_optimal()` - Balance speed/quality with confidence weighting
    - `select_smartest()` - Choose smartest within time budget

15. Implement fallback selection logic when primary choice unavailable

### 3.6 Configuration & Integration

16. Create `ModelSelectionConfig` using Pydantic for settings:
    - Model list with capability ratings
    - Default selection strategy
    - Latency window sizes
    - Rate limit thresholds
    - Time budgets by task type

17. Implement `SelectableModel` data class with:
    - Model ID, name, provider
    - Capability score (1-10)
    - Base pricing tier (for capability weighting)
    - Rate limits (RPM, TPM)

18. Create integration adapter for existing model-calling infrastructure

---

## 4. File Structure

```
model_selection/
├── __init__.py
├── config.py                 # Configuration classes
├── latency.py                # LatencyMonitor implementation
├── token_speed.py            # TokenSpeedEstimator implementation
├── rate_limits.py            # RateLimitTracker implementation
├── performance_model.py      # PerformanceModel implementation
├── selector.py               # ModelSelector implementation
├── models.py                # Model data classes
└── adapters.py              # Integration adapters

tests/
├── __init__.py
├── test_latency.py           # LatencyMonitor tests
├── test_token_speed.py       # TokenSpeedEstimator tests
├── test_rate_limits.py      # RateLimitTracker tests
├── test_performance_model.py # PerformanceModel tests
├── test_selector.py         # ModelSelector tests
├── test_integration.py      # End-to-end integration tests
├── fixtures/
│   └── __init__.py
└── mocks/
    ├── __init__.py
    └── mock_responses.py    # Mock API responses

docs/
├── ARCHITECTURE.md          # Architecture decision record
├── API_REFERENCE.md         # API documentation
└── USAGE.md                # Usage examples

scripts/
├── measure_baselines.py     # Script to establish baseline metrics
└── analyze_performance.py  # Performance analysis tool

.env.example                 # Environment variable template
requirements.txt            # Python dependencies
pytest.ini                  # Pytest configuration
.gitignore                  # Git ignore rules
QUESTIONS.md                # Questions for human input
```

---

## 5. Testing Requirements

### 5.1 Unit Test Requirements

**Coverage Targets:**
- Latency module: 90%+ coverage
- Token speed estimator: 85%+ coverage
- Rate limit tracker: 85%+ coverage
- Performance model: 80%+ coverage
- Model selector: 80%+ coverage

**Test Categories:**
1. Happy path tests for all public methods
2. Edge case tests (empty data, single sample, extreme values)
3. Error condition tests (invalid inputs, overflow, underflow)
4. Boundary condition tests (window boundaries, limit thresholds)

### 5.2 Integration Test Requirements

**Mock Testing:**
- Use mocked API responses for consistent testing
- Simulate various latency conditions
- Simulate rate limit scenarios

**Scenario Tests:**
- Selection under normal conditions
- Selection with approaching rate limits
- Selection with degraded model performance
- Fallback selection when primary unavailable

### 5.3 Performance Test Requirements

**Benchmarks:**
- Selection algorithm execution time