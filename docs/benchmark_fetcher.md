# Benchmark Fetcher Module

## Overview

The Benchmark Fetcher Module is part of Phase 2.1 Data Collection Pipeline. It provides a modular, extensible system for fetching model performance data from various external benchmark sources and integrating it into the proxy's routing decisions.

## Features

- **Multiple Source Support**: Fetch from Artificial Analysis, LMSYS Chatbot Arena, and custom static files
- **Normalized Data Model**: Consistent `BenchmarkEntry` format across all sources
- **Concurrent Fetching**: Asyncio-based concurrent fetching from all sources
- **Intelligent Caching**: Disk-based caching with configurable TTL
- **Error Resilience**: Retry logic with exponential backoff for each source
- **Easy Integration**: Simple integration with existing `ModelRanker` and routing system

## Configuration

Create `config/benchmark_sources.yaml`:

```yaml
sources:
  - name: artificial_analysis
    enabled: true
    url: https://api.artificialanalysis.ai/v1/models
    refresh_interval_minutes: 60
    priority: 100

  - name: lmsys_arena
    enabled: true
    url: https://chat.lmsys.org/api/leaderboard
    refresh_interval_minutes: 120
    priority: 90
```

## Usage

### Basic Usage

```python
from rotator_library import fetch_latest_benchmarks

# One-shot fetch
cache = await fetch_latest_benchmarks()

# Query results
entries = cache.get_by_model("gpt_4")
elo_scores = cache.get_by_source("lmsys_arena")
```

### Advanced Usage

```python
from rotator_library.benchmark_fetcher import BenchmarkFetcher

async with BenchmarkFetcher() as fetcher:
    # Initialize from config
    await fetcher.initialize()
    
    # Fetch all sources
    cache = await fetcher.fetch_all()
    
    # Get leaderboard
    leaderboard = fetcher.get_leaderboard(metric_name="elo_rating")
    
    # Get specific model scores
    scores = fetcher.get_model_scores("claude_3_opus")
```

### Integration with Proxy

```python
from proxy_app.benchmark_integration import setup_benchmark_integration

# In main.py startup
benchmark_integration = await setup_benchmark_integration(
    model_ranker=ranker,
    auto_refresh=True
)

# Get performance data for routing decisions
summary = benchmark_integration.get_model_performance_summary("gpt_4")
```

## Extending

To add a new benchmark source:

1. Create a fetcher class inheriting from `Fetcher`
2. Implement `fetch()` and `normalize_model_id()`
3. Register in `BenchmarkFetcher.FETCHER_REGISTRY`

```python
from rotator_library.benchmark_fetcher import Fetcher, BenchmarkEntry

class CustomFetcher(Fetcher):
    async def fetch(self, session):
        # Implementation
        return [BenchmarkEntry(...)]
    
    def normalize_model_id(self, raw_id):
        return raw_id.lower()

# Register
fetcher = BenchmarkFetcher()
fetcher.register_fetcher("custom", CustomFetcher)
```

## Data Model

### BenchmarkEntry
- `model_id`: Normalized model identifier
- `source`: Source of the benchmark (e.g., "artificial_analysis")
- `metric_name`: Type of metric (e.g., "throughput", "elo_rating")
- `metric_value`: Numeric value
- `unit`: Optional unit (e.g., "tokens/sec")
- `timestamp`: When the data was fetched
- `context`: Additional metadata

## Cache

Benchmark data is cached in `cache/benchmark_cache.json` with the following structure:
- `entries`: List of BenchmarkEntry objects
- `last_updated`: ISO timestamp of last successful fetch
- `source_status`: Status of each source ("ok" or error message)
