# Uptime Monitoring Setup Guide

This guide explains how to configure external uptime monitoring for the Mirro-Proxy service.

## Health Check Endpoint

The proxy exposes a health check endpoint at:

```
GET /health
```

### Response Format

**Healthy (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime_seconds": 3600,
  "version": "1.0.0",
  "checks": {
    "rotating_client": "ok",
    "credential_manager": "ok",
    "background_refresher": "ok"
  }
}
```

**Unhealthy (503 Service Unavailable):**
```json
{
  "status": "unhealthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime_seconds": 3600,
  "version": "1.0.0",
  "checks": {
    "rotating_client": "error",
    "credential_manager": "ok",
    "background_refresher": "error"
  },
  "error_details": "Connection timeout to provider API"
}
```

## Quick Start

### 1. Verify Endpoint Availability

Start the proxy server and test the health endpoint:

```bash
curl http://localhost:8000/health
```

### 2. Configure Monitoring Service

#### UptimeRobot
1. Add new monitor → HTTP(s)
2. URL: `http://your-host:8000/health`
3. Monitoring Interval: 1-5 minutes
4. Alert When: Status ≠ 200

#### Pingdom
1. Add new check → HTTP
2. Host: `your-host`
3. Port: `8000`
4. Path: `/health`
5. Encryption: No (unless using HTTPS)

#### Datadog Synthetic Monitor
```json
{
  "type": "api",
  "config": {
    "request": {
      "method": "GET",
      "url": "http://your-host:8000/health"
    },
    "assertions": [
      {"type": "statusCode", "operator": "is", "target": 200},
      {"type": "body", "operator": "validatesJSONPath", "target": {"jsonPath": "$.status", "operator": "is", "expectedValue": "healthy"}}
    ]
  }
}
```

## Kubernetes Probes

Add to your deployment manifest:

```yaml
livenessProbe:
  httpGet:
    path: /live
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
  failureThreshold: 3
```

## Docker Health Check

Add to your `docker-compose.yml`:

```yaml
services:
  proxy:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

## Configuration

Modify `config/router_config.yaml` to customize health check behavior:

```yaml
uptime_monitoring:
  enabled: true
  
  endpoint:
    path: "/health"
    # Set to true if using monitoring service that supports custom headers
    require_auth: false
    
  health_checks:
    check_providers: true
    check_credential_manager: true
    check_background_refresher: true
    
  thresholds:
    max_response_time_ms: 5000
    max_error_rate: 0.1
```

## Endpoint Reference

| Endpoint | Method | Auth Required | Purpose |
|----------|--------|---------------|---------|
| `/health` | GET | No* | Comprehensive health check (returns 503 if unhealthy) |
| `/live` | GET | No | Liveness probe - basic process check |
| `/ready` | GET | No | Readiness probe - checks if ready to accept traffic |
| `/status` | GET | Yes** | Detailed system status dashboard |

\* Configure `require_auth: true` in config to enforce PROXY_API_KEY header
\*\* Requires PROXY_API_KEY for detailed operational data

## Troubleshooting

**Health check returns 503:**
```bash
# Check logs
docker logs <container_id>

# Verify provider connectivity
curl -v http://localhost:8000/health

# Check specific component
curl http://localhost:8000/status -H "Authorization: Bearer $PROXY_API_KEY"
```

**High response latency:**
- Increase `thresholds.max_response_time_ms` in config
- Check provider API latency in logs
- Verify rate limiting isn't throttling health checks

## Security Considerations

By default, `/health`, `/live`, and `/ready` endpoints are publicly accessible to support external monitoring services. To secure these endpoints:

1. Set `uptime_monitoring.endpoint.require_auth: true` in config
2. Configure your monitoring service to send the header:
   ```
   Authorization: Bearer <PROXY_API_KEY>
   ```
3. For Kubernetes probes, use the `PROXY_API_KEY` in the HTTP headers configuration.

## Alerting Best Practices

1. **Page on-call only for 503 errors** on `/health` (not for `/live` or `/ready`)
2. **Set degradation thresholds**: Alert if response time > 2s for 5 consecutive checks
3. **Provider-specific monitoring**: Use `/status` endpoint to alert on individual provider failures
4. **Uptime correlation**: Track uptime in monitoring service against `uptime_seconds` field to detect restarts
