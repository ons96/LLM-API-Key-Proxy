# AGENTS.md - Rate and Usage Limit Tracker

## 1. Role/Mission

**Mission**: Build a Rate and Usage Limit Tracker system that tracks API rate limits and usage quotas for all LLM providers, enabling the gateway to intelligently skip requests to rate-limited or quota-exhausted providers without making unnecessary API calls.

The agent must implement a robust tracking mechanism that:
- Maintains current usage counts against defined limits for each provider
- Records timestamps for limit reset/expiry windows
- Provides a clean API for checking availability before attempting provider calls
- Automatically resets counters when limits expire based on time windows
- Supports multiple provider-specific limit configurations (per-minute, per-day, per-month)

**Decision Authority**: The agent is authorized to make all technical decisions independently, including database schema design, API interface choices, and implementation details. The agent should only save questions to QUESTIONS.md when blocked for more than 30 minutes or when a decision requires human context.

---

## 2. Technical Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Language** | Python 3.10+ | Industry standard, excellent library support |
| **Database** | SQLite (via SQLAlchemy) | Zero configuration, free, embedded |
| **ORM** | SQLAlchemy 2.0+ | Modern async support, type safety |
| **Async Framework** | asyncio | Native Python async support |
| **Testing** | pytest + pytest-asyncio | Free, comprehensive testing |
| **Type Checking** | mypy | Free static type checking |
| **Formatting** | Ruff | Fast, free Python formatter |

**Key Dependencies** (requirements.txt):
```
sqlalchemy[asyncio]>=2.0.0
aiosqlite>=0.19.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
mypy>=1.5.0
ruff>=0.1.0
python-dateutil>=2.8.0
```

---

## 3. Requirements

### 3.1 Database Schema

1. **Providers Table**
   - `id`: Primary key (integer, auto-increment)
   - `provider_name`: Unique identifier (string, e.g., "openai", "anthropic", "cohere")
   - `display_name`: Human-readable name for logging
   - `rate_limit_requests_per_minute`: Default rate limit
   - `rate_limit_tokens_per_minute`: Token limit if applicable
   - `rate_limit_requests_per_day`: Daily quota
   - `created_at`: Timestamp of provider registration
   - `updated_at`: Last modification timestamp

2. **Rate Limit Status Table**
   - `id`: Primary key
   - `provider_id`: Foreign key to Providers
   - `limit_type`: Enum ("per_minute", "per_day", "per_month")
   - `current_count`: Current usage within window
   - `limit_value`: Maximum allowed in window
   - `window_start`: Timestamp when current window began
   - `window_duration_seconds`: Length of limit window
   - `expires_at`: When this limit record becomes invalid
   - `last_updated`: Last modification timestamp

3. **Usage Logs Table** (for debugging/analytics)
   - `id`: Primary key
   - `provider_id`: Foreign key
   - `request_timestamp`: When request was attempted
   - `was_rate_limited`: Boolean flag
   - `tokens_used`: Token count (if available)
   - `retry_after_seconds`: Value from 429 response if present

### 3.2 Core Functionality

4. **Provider Registration**
   - Ability to register new providers with custom limit configurations
   - Support for registering multiple limit types per provider
   - Default configurations for common providers (OpenAI, Anthropic, etc.)

5. **Limit Checking API**
   - `is_available(provider_name: str) -> bool`: Check if provider can accept requests
   - `check_limits(provider_name: str) -> dict`: Detailed limit status
   - `get_retry_after(provider_name: str) -> int | None`: Seconds to wait if rate limited

6. **Limit Tracking**
   - `record_request(provider_name: str, tokens: int = 0) -> bool`: Record usage, returns False if rate limited
   - `record_rate_limit_response(provider_name: str, retry_after: int)`: Update based on 429 response
   - `increment_usage(provider_name: str, limit_type: str, amount: int)`: Manually increment usage

7. **Automatic Reset**
   - Window-based reset: When `expires_at` passes, counter resets to 0
   - Background task or check-on-use to handle window expiration
   - Support for sliding windows vs fixed windows

8. **Provider Configuration**
   - Pre-built configurations for common providers:
     - OpenAI: 60 req/min, 90k tokens/min, 120k req/day
     - Anthropic: 50 req/min (varies by tier)
     - Google AI: 60 req/min
     - Cohere: Various based on tier

### 3.3 Helper Features

9. **Configuration Helper**
   - Load/save provider configurations from JSON file
   - Validate configurations before applying
   - Support environment variable overrides

10. **Logging and Observability**
    - Structured logging for all limit checks and updates
    - Warning logs when approaching limits (80% threshold)
    - Alert logs when rate limited

---

## 4. File Structure

```
rate_limit_tracker/
├── .github/
│   └── workflows/
│       └── test.yml
├── .gitignore
├── .ruff.toml
├── mypy.ini
├── pyproject.toml
├── requirements.txt
├── README.md
├── agents/
│   ├── __init__.py
│   ├── models.py              # SQLAlchemy models
│   ├── repository.py          # Database operations
│   ├── service.py             # Business logic
│   ├── config.py              # Configuration loader
│   └── types.py               # Custom types/enums
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # pytest fixtures
│   ├── test_models.py
│   ├── test_repository.py
│   ├── test_service.py
│   └── test_integration.py
├── providers/
│   └── default_config.json    # Default provider configs
└── main.py                    # CLI entry point for testing
```

**Key Files**:

| File | Purpose |
|------|---------|
| `agents/models.py` | SQLAlchemy table definitions |
| `agents/repository.py` | CRUD operations, data access layer |
| `agents/service.py` | Business logic, limit checking algorithms |
| `agents/config.py` | Configuration loading and validation |
| `agents/types.py` | Enums and custom type definitions |
| `tests/conftest.py` | Shared pytest fixtures with in-memory DB |
| `providers/default_config.json` | JSON config for default providers |

---

## 5. Testing Requirements

### 5.1 Unit Tests (Required)

- **test_models.py**: Model initialization and relationship tests
- **test_repository.py**: CRUD operation tests
  - Provider create/read/update
  - Limit record upsert
  - Usage record logging
- **test_service.py**: Core business logic tests
  - `is_available()` returns correct status
  - `record_request()` increments correctly and enforces limits
  - Window expiration triggers reset
  - Retry-after handling from 429 responses

### 5.2 Integration Tests (Required)

- **test_integration.py**: Full workflow tests
  - Register provider → check available → record request → verify count
  - Rate limiting: Record until limit exceeded, verify rejection
  - Window reset: Advance time, verify counter resets

### 5.3 Test Coverage Requirements

- Minimum **80% code coverage** required
- All public API methods must have tests
- Edge cases: Limit boundaries, window transitions, concurrent access

### 5.4 Test Fixtures

- Use in-memory SQLite database for tests
- Provide sample provider configurations
- Include helper to mock time windows for expiration testing

---

## 6. Git Protocol

### 6.1 Branch Strategy

- **Main branch**: `main` (production-ready code)
- **Working branch**: Agent creates feature branches as needed
- **Format**: `feature