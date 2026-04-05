# AGENTS.md - Intelligent Rate Limit Prediction & Avoidance System

## 1. Role/Mission

**Mission:** Design and implement an intelligent system that tracks API rate limits and usage limits across multiple providers, uses historical call data to predict when limits will expire, and automatically avoids making API calls to providers that are currently rate-limited or approaching their limits.

**Autonomous Agent Responsibilities:**
- Create a persistent data store for tracking provider limits and API call history
- Implement prediction algorithms to estimate when rate limits will expire based on historical patterns
- Build an intelligent routing mechanism that skips providers showing signs of rate limiting
- Generate warnings and notifications when usage thresholds are approached
- Make independent decisions about which provider to use based on current limit status
- Continuously learn from past API call outcomes to improve prediction accuracy

## 2. Technical Stack

**Programming Language:** Python 3.9+

**Core Dependencies:**
- `sqlite3` - Built-in SQLite for local database storage
- `dataclasses` - Built-in for data modeling
- `datetime` - Built-in for timestamp handling
- `threading` - Built-in for concurrent operations
- `json` - Built-in for configuration management
- `logging` - Built-in for logging functionality
- `hashlib` - Built-in for data integrity

**Optional Enhancements (Free Tier):**
- `apscheduler` - For scheduled limit reset checks (free)
- `pandas` - For statistical analysis of call patterns (free)

**Storage:**
- SQLite database (local file-based, no cost)
- JSON configuration files

**Runtime Environment:**
- Python 3.9+ standard library
- No external API keys required for the tracking system itself

**Version Control:**
- Git
- GitHub Actions for CI/CD

## 3. Requirements (Numbered)

### 3.1 Data Storage Requirements

1. **Provider Limit Database**
   - Create SQLite database file: `provider_limits.db`
   - Table: `providers` with columns:
     - `provider_id` (PRIMARY KEY) - Unique provider identifier
     - `provider_name` - Human-readable provider name
     - `rate_limit_max` - Maximum requests allowed in a window
     - `rate_limit_window_seconds` - Time window for rate limit
     - `usage_limit_max` - Maximum usage units (if applicable)
     - `current_rate_limit_count` - Current requests in window
     - `current_usage_count` - Current usage units
     - `rate_limit_reset_timestamp` - When rate limit resets (ISO 8601)
     - `usage_limit_reset_timestamp` - When usage limit resets (ISO 8601)
     - `last_updated` - Last time data was updated
   - Auto-create database on first run

2. **API Call History Table**
   - Table: `api_call_history` with columns:
     - `call_id` (PRIMARY KEY, AUTOINCREMENT)
     - `provider_id` - Which provider was called
     - `timestamp` - When the call was made
     - `endpoint` - Which endpoint was called
     - `success` - Whether call succeeded (boolean)
     - `response_status` - HTTP status code or error code
     - `rate_limit_hit` - Whether rate limit was encountered (boolean)
     - `retry_after_seconds` - If rate limited, how long to wait

### 3.2 Core Functionality Requirements

3. **Rate Limit Checker**
   - Function: `is_provider_available(provider_id)` returns Boolean
   - Check if current time is past `rate_limit_reset_timestamp`
   - Return `False` if null timestamp or if reset time is in future
   - Log check results for audit trail

4. **Usage Limit Checker**
   - Function: `has_usage_available(provider_id, units_needed)` returns Tuple(Boolean, Int)
   - Check if `current_usage_count + units_needed