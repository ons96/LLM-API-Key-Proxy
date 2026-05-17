# Streaming Usage Tracking

## Overview

Phase 4.2 introduces comprehensive streaming usage tracking to the rotator library. This feature enables accurate load balancing and monitoring of long-running streaming connections to LLM providers.

## Features

- **Stream Lifecycle Tracking**: Record start and end of streaming sessions
- **Token Accounting**: Track total tokens processed through streams
- **Duration Metrics**: Measure cumulative streaming duration
- **Active Stream Management**: In-memory tracking of concurrent streams per credential
- **Stale Stream Cleanup**: Automatic cleanup of orphaned stream entries

## Usage

### Recording Stream Usage

```python
from rotator_library.usage_manager import UsageManager

usage_mgr = UsageManager()

# When starting a stream
await usage_mgr.record_stream_start(credential="path/to/cred.json")

# When ending a stream
await usage_mgr.record_stream_end(
    credential="path/to/cred.json", 
    tokens=150,  # Optional: tokens processed
    duration_ms=2500  # Optional: stream duration
)
```

### Retrieving Statistics

```python
# Get stats for specific credential
stats = await usage_mgr.get_streaming_stats(credential="path/to/cred.json")
# Returns:
# {
#     "stream_starts": 42,
#     "stream_tokens": 15000,
#     "stream_duration_ms": 45000,
#     "active_streams": 2,
#     "last_stream_start": "2024-01-15T10:30:00+00:00",
#     "last_stream_end": "2024-01-15T10:35:00+00:00"
# }

# Get aggregate stats
all_stats = await usage_mgr.get_streaming_stats()
```

### Load Balancing Integration

The UsageManager automatically incorporates active stream counts into the credential selection weight calculation:

- **Deterministic Mode** (`rotation_tolerance=0`): Active streams count as additional usage
- **Random Mode** (`rotation_tolerance>0`): Active streams are scaled by tolerance factor

This ensures credentials with many active streams are less likely to receive new requests, preventing overload.

### Data Persistence

Streaming statistics are persisted to the usage JSON file with the following schema:

```json
{
  "credentials": {
    "cred_id_hash": {
      "usage_count": 100,
      "streaming": {
        "stream_starts": 50,
        "stream_tokens": 10000,
        "stream_duration_ms": 300000,
        "last_stream_start": "2024-01-15T10:00:00+00:00",
        "last_stream_end": "2024-01-15T11:00:00+00:00"
      }
    }
  }
}
```

Note: Active stream counts are maintained in memory only and reset on restart. Stale stream cleanup is performed automatically to handle crashes.

## Configuration

No additional configuration is required. Streaming tracking is automatically available when using UsageManager.

## Migration

Existing usage data files are automatically migrated to version 2.0 format with streaming support on first load.
