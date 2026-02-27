# Data Pipeline Integration

This document describes how the data collection components integrate with the broader router system.

## Component Interaction

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Data Sources  │────▶│  Collection      │────▶│   Storage       │
│                 │     │  Layer           │     │   (CSV/YAML)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                        │
┌─────────────────┐     ┌──────────────────┐           │
│   Router Core   │◀────│  Service Layer   │◀──────────┘
│   (Proxy)       │     │  (Model Info)    │
└─────────────────┘     └──────────────────┘
        │
        ▼
┌─────────────────┐
│  Provider       │
│  Selection      │
└─────────────────┘
```

## Integration Points

### 1. Model Ranker Integration

**File**: `src/proxy_app/model_ranker.py`

The ranker consumes processed data to make routing decisions:

```python
from rotator_library.model_info_service import ModelInfoService

class ModelRanker:
    def __init__(self):
        self.info_service = ModelInfoService()
        
    def get_best_model(self, capability: str) -> str:
        # Uses intelligence index from CSV + provider limits from YAML
        candidates = self.info_service.get_models_by_capability(capability)
        return self.score_candidates(candidates)
```

**Data Dependencies**:
- `artificial_analysis_models.csv`: Intelligence scores, speed metrics
- `config/providers_database.yaml`: Rate limits, context windows

### 2. Provider Adapter Integration

**File**: `src/rotator_library/provider_adapter.py`

Adapters validate against collected data:

```python
def validate_model_availability(self, model_id: str) -> bool:
    # Check if model exists in latest benchmark data
    if not self.model_info.exists(model_id):
        logger.warning(f"Model {model_id} not in benchmark data")
    return self.check_provider_limits(model_id)
```

### 3. Health Checker Integration

**File**: `src/proxy_app/health_checker.py`

Monitors data freshness:

```python
def check_data_freshness(self) -> HealthStatus:
    timestamp_file = "last_successful_fetch.txt"
    if not os.path.exists(timestamp_file):
        return HealthStatus.WARNING, "No data collection timestamp found"
    
    # Parse timestamp and check age
    # Alert if > 48 hours old
```

### 4. Background Refresher

**File**: `src/rotator_library/background_refresher.py`

Automated refresh without blocking:

```python
async def refresh_benchmark_data(self):
    """Background task to update AI Analysis data."""
    from fetch_artificial_analysis_data import run_collection_pipeline
    
    while True:
        success = run_collection_pipeline(force_refresh=False)
        if not success:
            self.failure_logger.log("benchmark_refresh_failed")
        await asyncio.sleep(3600)  # Check every hour
```

## Data Contracts

### Model Record Schema (Post-Processing)

```json
{
  "id": "uuid-string",
  "name": "GPT-4 Turbo",
  "slug": "gpt-4-turbo",
  "model_creator_name": "OpenAI",
  "model_creator_slug": "openai",
  "eval_artificial_analysis_intelligence_index": 85.4,
  "eval_mmlu": 86.4,
  "price_input_per_1m": 10.00,
  "price_output_per_1m": 30.00,
  "median_output_tokens_per_second": 45.2
}
```

### Provider Record Schema

```yaml
id: openrouter
name: "OpenRouter"
free_models:
  - id: "google/gemini-2.0-flash-exp:free"
    context: 1000000
    rpm: 10
capabilities: [chat, code, vision]
```

## Configuration Integration

### Router Config

**File**: `config/router_config.yaml`

```yaml
data_sources:
  artificial_analysis:
    enabled: true
    cache_hours: 24
    priority: 1
  provider_database:
    enabled: true
    path: "config/providers_database.yaml"
    priority: 2

scoring:
  weights:
    intelligence_index: 0.4
    speed: 0.3
    cost: 0.2
    reliability: 0.1
```

### Scoring Config

**File**: `config/scoring_config.yaml`

Uses collected data fields:

```yaml
metrics:
  intelligence:
    source: "eval_artificial_analysis_intelligence_index"
    normalize: true
  speed:
    source: "median_output_tokens_per_second"
    weight: 0.25
  cost_efficiency:
    source: "price_input_per_1m"
    inverse: true  # Lower is better
```

## Error Handling Strategy

### Data Unavailable Fallbacks

1. **API Down**: Use cached CSV (indefinite fallback)
2. **CSV Corrupted**: Use provider database only (basic routing)
3. **Partial Data**: Weight available scores, ignore missing

### Circuit Breaker Pattern

```python
class DataSourceCircuitBreaker:
    def __init__(self):
        self.failure_count = 0
        self.threshold = 5
        
    def call(self, func):
        if self.failure_count >= self.threshold:
            return self.use_cached_data()
        try:
            result = func()
            self.failure_count = 0
            return result
        except Exception:
            self.failure_count += 1
            raise
```

## Performance Considerations

### Memory Management
- CSV loaded once at startup
- Service layer maintains in-memory cache
- Background refresh updates cache atomically

### Disk I/O
- Write to temp file, then atomic move
- Prevents partial reads during update
- Timestamp updated only after successful write

### Startup Sequence
1. Load provider database (fast, static)
2. Load benchmark CSV (fallback to empty if missing)
3. Trigger background refresh if cache stale
4. Mark router ready

## Monitoring & Observability

### Metrics to Track

| Metric | Source | Alert Threshold |
|--------|--------|----------------|
| data_age_hours | `last_successful_fetch.txt` | > 48 hours |
| api_success_rate | fetch script | < 95% over 24h |
| records_processed | process_api_data | < 100 (suspicious) |
| parse_errors | exception handler | > 5 per hour |

### Logging Integration

All data collection components use structured logging:

```python
logger.info("data_collection.completed", extra={
    "records_processed": len(records),
    "source": "artificial_analysis_api",
    "duration_ms": elapsed_time
})
```

## Testing Integration

### Mock Data for Tests

```python
# conftest.py
@pytest.fixture
def mock_benchmark_data():
    return pd.DataFrame([
        {"name": "test-model", "eval_intelligence": 80.0}
    ])

@pytest.fixture  
def mock_provider_db():
    return load_yaml("tests/fixtures/test_providers.yaml")
```

### Integration Tests

Test the full pipeline:
1. Start with empty directory
2. Run fetch with mock API
3. Verify CSV output
4. Verify service layer can read data
5. Verify router uses data for selection
