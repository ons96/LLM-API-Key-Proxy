# API Authentication Guide

This document describes the authentication system for the LLM API Proxy.

## Overview

The proxy uses Bearer token authentication via the `Authorization` header. All API endpoints (except health check) require a valid API key.

## Authentication Method

### Header Format

```bash
# Standard Bearer token
Authorization: Bearer YOUR_API_KEY
```

### Example Request

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer your-secret-key"
```

## Configuration

### Setting the API Key

Add to your `.env` file:

```env
# Required for production use
PROXY_API_KEY=your-secure-random-key-here
```

### API Key Validation

The proxy validates the API key as follows:

1. If `PROXY_API_KEY` is set in environment, it validates against that key
2. If not set, authentication is bypassed (development mode)

## Endpoints and Authentication

| Endpoint | Auth Required | Description |
|----------|--------------|-------------|
| `GET /` | No | Root info |
| `GET /stats` | No | Server statistics |
| `GET /health` | No | Health check |
| `GET /v1/models` | Yes | List available models |
| `POST /v1/chat/completions` | Yes | Chat completions |
| `POST /v1/responses` | Yes | Responses API |
| `GET /v1/providers` | Yes | List providers |
| `POST /v1/cost-estimate` | Yes | Cost estimation |

## Implementation Details

### FastAPI Dependency

The authentication is implemented as a FastAPI dependency:

```python
from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def verify_api_key(auth: str = Depends(api_key_header)) -> str:
    """Verify the API key from the Authorization header."""
    # Get expected key from settings
    expected_key = get_settings().proxy_api_key
    
    # If no key configured, allow all requests (dev mode)
    if not expected_key:
        return "dev-mode"
    
    # Validate the key
    if auth and auth.startswith("Bearer "):
        provided_key = auth[7:]  # Remove "Bearer " prefix
        if provided_key == expected_key:
            return provided_key
    
    raise HTTPException(
        status_code=401,
        detail="Invalid API key"
    )
```

### Applying to Endpoints

All protected endpoints use the dependency:

```python
@app.get("/v1/models")
async def list_models(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),  # Authentication required
    enriched: bool = True,
) -> dict[str, Any]:
    # Endpoint logic
    pass
```

## Security Best Practices

### For Production

1. **Set a strong API key**:
   ```env
   # Generate a secure random key
   PROXY_API_KEY=$(openssl rand -hex 32)
   ```

2. **Use HTTPS** in production to protect the API key in transit

3. **Rotate keys regularly**: Update `PROXY_API_KEY` periodically

4. **Restrict by IP** (optional): Use a reverse proxy like Nginx

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `PROXY_API_KEY` | No* | API key for proxy authentication |
| | | *Required for production, optional for development |

## Integration Examples

### Python

```python
import requests

headers = {
    "Authorization": "Bearer your-api-key",
    "Content-Type": "application/json"
}

response = requests.get(
    "http://localhost:8000/v1/models",
    headers=headers
)
```

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"  # Any value, validated by proxy
)

response = client.chat.completions.create(
    model="coding-fast",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### cURL

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer your-api-key"
```

## Troubleshooting

### 401 Unauthorized

**Cause**: Invalid or missing API key

**Solution**:
1. Check `PROXY_API_KEY` is set in `.env`
2. Ensure header format is correct: `Authorization: Bearer YOUR_KEY`
3. Verify key matches exactly (no extra spaces)

### Authentication Bypass

**Cause**: `PROXY_API_KEY` not set

**Solution**: Set `PROXY_API_KEY` in `.env` for production:

```env
PROXY_API_KEY=your-secure-key
```

## Related Documentation

- [Developer Setup Guide](./DEVELOPER_SETUP.md)
- [Deployment Guide](./DEPLOYMENT.md)
- [Settings Configuration](./SETTINGS.md)
