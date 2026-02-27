# Dockerfile Healthcheck Configuration

To add the healthcheck to your Dockerfile, include the following instruction:

```dockerfile
# Install Python if using slim/alpine base images
# FROM python:3.11-slim

# Copy the application
COPY src/ /app/src/

# Add healthcheck using the provided script
HEALTHCHECK --interval=30s \
            --timeout=10s \
            --start-period=40s \
            --retries=3 \
            CMD python /app/src/proxy_app/healthcheck.py || exit 1

# Or using curl if available in your image:
# HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
#   CMD curl -f http://localhost:8000/health || exit 1

# Environment variables for healthcheck script
ENV PORT=8000
ENV HOST=localhost

# Start the application
CMD ["python", "-m", "proxy_app.main"]
```

## Alternative: Inline Healthcheck

If you prefer not to use the Python script, you can use curl directly in the Dockerfile:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8000/health | grep -q "healthy" || exit 1
```

## Health Endpoint Details

The health endpoint is available at `GET /health` and returns:

```json
{
  "status": "healthy",
  "timestamp": 1234567890.123,
  "version": "2.0.0",
  "uptime": 123.45
}
```

HTTP Status codes:
- `200 OK`: Service is healthy and ready to accept requests
- `503 Service Unavailable`: Service is unhealthy or still initializing
