# AGENTS.md: Live API Provider Performance Tracking Database

## 1. Role/Mission

**Role**: You are an autonomous coding agent responsible for building a system that tracks LLM (Large Language Model) API provider performance metrics and implements a dynamic model fallback router.

**Mission**: 
- Continuously monitor and record performance metrics from various LLM API providers (response time, time-to-first-token, tokens-per-second, rate limits, usage limits)
- Build analytics to understand provider performance patterns by time-of-day
- Implement a dynamic router that selects the best available agentic coding model at any given moment based on historical trends and live data
- Operate autonomously using only free resources

## 2. Technical Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Database | SQLite | File-based, no external server needed, works on GitHub Actions |
| Language | Python 3.11+ | Wide library support, easy to run on most platforms |
| Scheduling | GitHub Actions (cron) | Free scheduling baked into GitHub |
| API Testing | ` requests` library | Lightweight HTTP calls |
| Data Storage | Local SQLite file in repo | Version controlled data files |
| Response Parsing | `json` + `re` | Standard library, no extra deps |

**Free Resource Constraints**:
- Only use free-tier LLM APIs (e.g., OpenRouter free credits, Ollama local models, Google's Gemini API free tier)
- No paid services or API keys requiring payment
- All persistence must be local file-based

## 3. Requirements

### 3.1 Performance Tracking

1. **Response Time Measurement**
   - Record time from request start to full response completion
   - Store in milliseconds with timestamp

2. **Time-to-First-Token (TTFT)**
   - Measure latency between request and first token received
   - Critical metric for interactive streaming applications

3. **Tokens-Per-Second (TPS)**
   - Calculate streaming speed: (total tokens / total streaming time)
   - Track both input and output token counts

4. **Rate Limit Monitoring**
   - Track free tier limits (requests/day, tokens/day)
   - Monitor remaining quota in real-time
   - Detect when provider limits are reached

5. **Usage Limit Tracking**
   - Record daily/monthly usage against provider limits
   - Calculate utilization percentage

### 3.2 Analytics

6. **Time-of-Day Analytics**
   - Aggregate performance metrics by hour of day
   - Build performance profiles (min/avg/max/stddev per time window)
   - Identify peak/off-peak performance patterns

7. **Historical Data Retention**
   - Store at least 30 days of granular data
   - Enable trend analysis

### 3.3 Dynamic Router

8. **Provider Health Scoring**
   - Calculate composite health score: `f(response_time, ttft, tps, availability)`
   - Weight factors appropriately (TTFT critical for agentic use)

9. **Fallback Chain Logic**
   - Implement priority-based provider selection
   - Auto-fallback when primary provider fails or rate-limited
   - Consider time-of-day performance patterns in selection

10. **Real-Time Selection**
    - On-demand selection of best available provider
    - Include retry logic with exponential backoff

### 3.4 System

11. **Automated Data Collection**
    - Run performance tests on scheduled interval (hourly minimum)
    - Use GitHub Actions cron for scheduling

12. **Data Persistence**
    - Store all metrics in SQLite database
    - Enable query interface for router decisions

## 4. File Structure

```
.
├── AGENTS.md                          # This file
├── README.md                         # Project overview
├── QUESTIONS.md                      # Questions for human clarification
├── .github/
│   └── workflows/
│       └── collect_metrics.yml      # Scheduled metric collection
├── src/
│   ├── __init__.py
│   ├── main.py                       # CLI entry point
│   ├── tracker/
│   │   ├── __init__.py
│   │   ├── collector.py              # Metric collection logic
│   │   └── providers.py              # Provider API interfaces
│   ├── database/
│   │   ├── __init__.py
│   │   ├── schema.py                # Database schema
│   │   └── manager.py                # DB operations
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── processor.py             # Time-of-day analytics
│   │   └── models.py                # Data models
│   └── router/
│       ├── __init__.py
│       ├── selector.py              # Provider selection logic
│       └── fallback.py              # Fallback chain implementation
├── tests/
│   ├── __init__.py
│   ├── test_tracker.py              # Tracker tests
│   ├── test_database.py             # Database tests
│   ├── test_router.py               # Router tests
│   └── test_integration.py          # Integration tests
├── data/
│   └── metrics.db                   # SQLite database (gitignored)
├── config/
│   └── providers.yaml               # Provider configuration
└── requirements.txt                 # Python dependencies
```

## 5. Testing Requirements

### 5.1 Unit Tests

- **Database Operations**: Test CRUD operations on metrics table
- **Analytics Calculation**: Test time-of-day aggregation logic
- **Router Selection**: Test provider selection with mock data

### 5.2 Integration Tests

- **Full Collection Flow**: Test complete metric collection pipeline
- **API Connectivity**: Test provider API calls (may need mocking for free tier limits)
- **Router Decision Flow**: Test end-to-end provider selection

### 5.3 Test Coverage

| Module | Minimum Coverage |
|--------|------------------|
| database/ | 80% |
| analytics/ | 70% |
| router/ | 75% |
| tracker/ | 60% (API limits may constrain full testing) |

### 5.4 Test Execution

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_router.py -v
```

## 6. Git Protocol

### 6.1 Branch Strategy

- **Main Branch**: `main` - Production-ready code only
- **Development**: `dev` - Integration branch for completed features
- **Feature Branches**: `feature