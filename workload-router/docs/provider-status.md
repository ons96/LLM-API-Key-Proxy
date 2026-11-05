# Read-only provider status compatibility

Date: 2026-11-03. No deployment, probe, or provider-database write was performed.

The sidecar can expose the gateway's existing `provider_status.db` health rows at
the legacy `/api/providers/*` paths. It does **not** run probes and does not
write the database. The gateway remains the only writer.

## Configuration

```bash
export SIDECAR_PROVIDER_STATUS_DB=/home/ubuntu/LLM-API-Key-Proxy/provider_status.db
export SIDECAR_PROVIDER_STATUS_STALE_SECONDS=900
```

`SIDECAR_PROVIDER_STATUS_DB` is optional. If it is unset, the status routes return
HTTP 503 with `provider status database not configured`. If it is set to a
missing file, startup fails rather than silently serving empty health data.
`SIDECAR_PROVIDER_STATUS_STALE_SECONDS` defaults to 900 seconds (15 minutes).

All `/api/providers/*` routes require the normal sidecar Bearer token. This is a
deliberate narrowing from the legacy gateway: provider status is operational
data, so it is not exposed unauthenticated.

## Routes

| Route | Behaviour |
| --- | --- |
| `GET /api/providers/status` | Latest row per provider, keyed by `provider_name`; includes `source`, `read_only`, `stale_after_seconds`, and a top-level `stale` flag |
| `GET /api/providers/status/{provider_name}` | Latest row for one provider, or 404 |
| `GET /api/providers/best` | Prefers a fresh healthy provider, then a stale healthy provider, then a degraded one; returns `best_provider`, `alternatives`, `source`, `stale`, and `as_of` |
| `GET /api/providers/history?hours=24` | Per-provider history for 1–168 hours |
| `GET /api/providers/history/{provider_name}?hours=24` | One provider's history for 1–168 hours, or 404 |
| `GET /api/providers/export/csv` | CSV with the legacy health columns |
| `GET /api/providers/health` | Snapshot health: `healthy`, `degraded`, or `unknown`; always `running:false`, `read_only:true` |

Each provider row retains the gateway fields:

```json
{
  "provider_name": "groq-free",
  "status": "healthy",
  "response_time_ms": 123.4,
  "uptime_percent": 99.2,
  "rate_limit_percent": 0.0,
  "last_check": "2026-11-03T12:34:56",
  "error_message": "",
  "consecutive_failures": 0,
  "age_seconds": 42.1,
  "stale": false
}
```

History rows use `provider_name`, `status`, `response_time_ms`, and `timestamp`.
CSV columns are `provider_name,status,response_time_ms,uptime_percent,
rate_limit_percent,last_check_timestamp,consecutive_failures`.

## Staleness semantics

- `stale` is based on `last_check_timestamp` and the configured threshold, never
  on the current wall-clock time alone.
- If the gateway stops writing the database, rows keep being served but become
  `stale:true`; `/api/providers/health` changes to `degraded`.
- `best_provider` can still name a stale provider when no fresh candidate exists,
  but its response carries `stale:true` and the reason says `stale snapshot`.
- `running:false` means the sidecar is not probing; it does not mean the gateway
  is stopped.
- No latency or uptime number is invented. Missing SQLite values remain `null`.

## Limitations

- No live probe, cooldown, circuit-breaker, or provider-switch logic is added by
  these endpoints.
- Provider names are whatever the gateway wrote; no alias translation is done.
- The compatibility inventory treats these routes as partial because their data
  is a read-only snapshot and they require sidecar auth.
- Reading is safe while the gateway is writing SQLite, but a long-running
  snapshot becomes stale by design.
