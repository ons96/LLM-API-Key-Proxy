# Phase 6.2: Automated Deployment Workflow

This document describes the automated deployment workflow implementation for CI/CD pipeline integration.

## Overview

The deployment workflow provides endpoints and configuration for automated deployment platforms including GitHub Actions, GitLab CI, Kubernetes, and Docker Compose.

## Configuration File

`config/deployment_config.yaml` contains environment-specific settings:

- **Health check parameters**: Timeouts, retries, and paths for readiness/liveness probes
- **Required environment variables**: Validated on startup and deployment verification
- **Graceful shutdown**: Timeout configuration for zero-downtime deployments
- **Verification endpoints**: Automated post-deployment API checks

## Integration Points

### 1. Health Check Endpoints

- `GET /deployment/health/ready` - Returns 200 when app is initialized and ready for traffic
- `GET /deployment/health/live` - Returns 200 if application is running (basic liveness)
- `GET /deployment/status` - Detailed status including version and environment variable validation

### 2. Deployment Verification

- `POST /deployment/verify` - Triggers verification sequence, returns required checks for CI/CD
- `GET /deployment/config` - Returns safe configuration subset for deployment scripts

## Usage in Application

Add to `main.py` after FastAPI app creation:

```python
from proxy_app.deployment_hooks import init_deployment_hooks

app = FastAPI(...)
init_deployment_hooks(app)
```

## CI/CD Examples

### GitHub Actions Workflow

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy
        run: |
          # Your deployment commands here
          docker-compose up -d
      
      - name: Wait for rollout
        run: sleep 10
      
      - name: Health Check
        run: |
          curl -f http://localhost:8000/deployment/health/ready || exit 1
      
      - name: Verify Deployment
        run: |
          curl -f http://localhost:8000/deployment/verify || exit 1
          curl -f http://localhost:8000/models || exit 1
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mirro-proxy
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    spec:
      containers:
      - name: proxy
        image: mirro-proxy:latest
        env:
        - name: DEPLOYMENT_ENV
          value: "production"
        - name: APP_VERSION
          value: "${GITHUB_SHA}"
        readinessProbe:
          httpGet:
            path: /deployment/health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /deployment/health/live
            port: 8000
          periodSeconds: 30
```

## Environment Variables

- `DEPLOYMENT_ENV`: Specifies environment section to load from config (default: `production`)
- `APP_VERSION`: Application version reported in status endpoints
- `PROXY_API_KEY`: Required by default configuration (can be customized in config)

## Graceful Shutdown

The application handles SIGTERM for graceful shutdown:
1. Stops accepting new connections
2. Waits for existing requests to complete (configurable timeout)
3. Exits cleanly

Configure timeout in `config/deployment_config.yaml` under `shutdown.graceful_timeout_seconds`.
