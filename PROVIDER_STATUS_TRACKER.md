# Provider Status Tracker Documentation

## Overview

The Provider Status Tracker is a real-time monitoring system that tracks the health, uptime, latency, rate limits, and response times of all configured LLM API providers in the LLM-API-Key-Proxy. It enables intelligent provider routing and fallback decisions by providing up-to-date health information to the proxy's routing logic.

## Features

### 1. Multi-Provider Health Monitoring

The tracker monitors all configured free/non-paid providers including:

- **Groq**
- **OpenRouter**
- **Together AI**
- **G4F endpoints** (main, groq, grok, gemini, nvidia)
- **Gemini CLI**
- **Nvidia**
- **Mistral**
- **HuggingFace**
- **Gemini**
- **Google**

### 2. Health Metrics Tracked

For each provider, the tracker monitors:

- **Uptime %**: Percentage of successful health checks over time window
- **Response latency**: Min, max, avg, p95 response times in milliseconds
- **Rate limit status**: Current usage percentage and limits if available
- **Last check timestamp**: When the last health check was performed
- **Status**: healthy/degraded/down

### 3. Health Check Implementation

- **Async concurrent health checks** using aiohttp for efficient monitoring
- **Lightweight requests** to verify connectivity (GET /models or equivalent)
- **10-second timeout** per provider to prevent hanging
- **Configurable interval** (default: 5 minutes)
- **Concurrent execution** (all providers checked in parallel)
- **Graceful error handling** (provider down ≠ service crash)

### 4. Data Storage

- **SQLite database**: `provider_status.db`
- **Schema**: Comprehensive health check history with timestamps
- **7-day rolling window**: Automatic cleanup of old records
- **Efficient queries**: Indexed for fast current status and historical data retrieval

### 5. REST API Endpoints

#### GET `/api/providers/status`
Returns current status snapshot for all providers:

```json
{
  "timestamp": "2026-01-07T12:34:56Z",
  "providers": {
    "groq": {
      "status": "healthy",
      "response_time_ms": 245,
      "uptime_percent": 99.8,
      "rate_limit_percent": 45,
      "last_check": "2026-01-07T12:34:00Z",
      "error_message": "",
      "consecutive_failures": 0
    },
    "openrouter": {
      "status": "degraded",
      "response_time_ms": 1200,
      "uptime_percent": 92.3,
      "rate_limit_percent": 88,
      "last_check": "2026-01-07T12:34:05Z",
      "error_message": "High latency detected",
      "consecutive_failures": 2
    }
  }
}
```

#### GET `/api/providers/status/{provider_name}`
Returns detailed status for a single provider.

#### GET `/api/providers/best`
Returns the healthiest/fastest provider recommendation:

```json
{
  "best_provider": "groq",
  "reason": "lowest latency (245ms) + healthy status",
  "alternatives": ["together", "nvidia"]
}
```

#### GET `/api/providers/history?hours=24`
Returns historical data for all providers (time-series data for graphs).

#### GET `/api/providers/export/csv`
Exports all provider status data in CSV format for auditing and reporting.

### 6. Integration with Proxy Routing

The tracker provides a `get_healthiest_provider()` function that can be used by the router logic:

```python
from proxy_app.status_api import get_healthiest_provider

# Get the healthiest provider for routing
best_provider = get_healthiest_provider(app.state.provider_status_tracker)
```

The router can use this information to:
- Prefer healthy providers over down/degraded ones
- Among healthy providers, prefer lower latency options
- Implement intelligent fallback when primary providers are unavailable

## Configuration

### Environment Variables

The tracker automatically discovers providers from environment variables:

```bash
# Example configuration
GROQ_API_KEY="your_groq_api_key"
OPENROUTER_API_KEY="your_openrouter_api_key"
GEMINI_CLI_PROJECT_ID="your_gemini_cli_project_id"
# etc.
```

### Tracker Parameters

The tracker can be configured with these parameters:

```python
ProviderStatusTracker(
    check_interval_minutes=5,        # Health check interval (default: 5)
    max_consecutive_failures=3,     # Failures before marking as down (default: 3)
    degraded_latency_threshold_ms=1000  # Latency threshold for degraded status (default: 1000)
)
```

## Usage Examples

### 1. Getting Current Status

```python
from proxy_app.status_api import get_status_tracker

tracker = get_status_tracker(request)
status = tracker.get_current_status()
print(f"Current status: {status}")
```

### 2. Getting Best Provider for Routing

```python
from proxy_app.status_api import get_healthiest_provider

best_provider = get_healthiest_provider(app.state.provider_status_tracker)
if best_provider:
    print(f"Routing to: {best_provider}")
else:
    print("No healthy providers available")
```

### 3. Accessing Historical Data

```python
# Get 24-hour history for all providers
history = tracker.get_all_history(hours=24)

# Get 7-day history for a specific provider
provider_history = tracker.get_provider_history("groq", hours=168)
```

### 4. Using API Endpoints

```bash
# Get current status
curl http://localhost:8000/api/providers/status

# Get best provider
curl http://localhost:8000/api/providers/best

# Get historical data
curl http://localhost:8000/api/providers/history?hours=24

# Export to CSV
curl http://localhost:8000/api/providers/export/csv > provider_status.csv
```

## Database Schema

The `provider_status.db` SQLite database contains:

```sql
CREATE TABLE provider_health_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_name TEXT NOT NULL,
    status TEXT NOT NULL,
    response_time_ms REAL,
    uptime_percent REAL,
    rate_limit_percent REAL,
    last_check_timestamp DATETIME NOT NULL,
    error_message TEXT,
    consecutive_failures INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_provider_timestamp 
ON provider_health_checks(provider_name, last_check_timestamp);
```

## Error Handling

The tracker includes robust error handling:

- **Database unavailable**: Caches results in memory, syncs when available
- **Provider unreachable**: Marks as "down" after N consecutive failures
- **All providers down**: Returns graceful error response
- **Timeouts**: Don't crash service, just mark provider as failed check
- **Duplicate checks**: Avoids overlapping scheduled runs

## Logging

The tracker logs key events:

- Health check start/end
- Individual provider check results
- Failures and error details
- Database operations
- Schema updates

## Integration with Existing Proxy

The tracker integrates seamlessly with the existing proxy:

1. **Automatic initialization**: Starts with the proxy in the lifespan function
2. **Background operation**: Runs health checks independently without affecting proxy performance
3. **API endpoints**: Mounted under `/api/providers/*` namespace
4. **No breaking changes**: Existing proxy functionality remains unchanged

## Testing

Run the test script to verify functionality:

```bash
python test_provider_status.py
```

## Future Enhancements

Potential improvements for future versions:

1. **Rate limit tracking**: Parse response headers for accurate rate limit data
2. **Advanced analytics**: Predictive failure detection using ML
3. **Alerting**: Notifications when providers go down or degrade
4. **Custom thresholds**: Per-provider configuration for degraded/down status
5. **Geographic monitoring**: Check providers from multiple regions
6. **Performance trends**: Long-term performance analysis and reporting

## Support

For issues or questions, please refer to the main project documentation or open an issue on GitHub.