# CI/CD Setup Guide

This document outlines the Continuous Integration and Continuous Deployment (CI/CD) setup for the Mirro-Proxy application.

## Overview

The CI/CD pipeline automates testing, building, and deployment of the proxy application and rotator library. The pipeline ensures code quality through automated testing and provides consistent deployment procedures across environments.

## GitHub Actions Workflow

Create `.github/workflows/ci-cd.yml` in your repository root:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.11"
  PROXY_API_KEY: ${{ secrets.PROXY_API_KEY_TEST }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
          
      - name: Lint with ruff
        run: |
          ruff check src/
          ruff format --check src/
          
      - name: Type check with mypy
        run: |
          mypy src/proxy_app/ src/rotator_library/ --ignore-missing-imports
          
      - name: Run tests
        run: |
          pytest tests/ -v --cov=src --cov-report=xml
          
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  integration-test:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          
      - name: Start proxy server
        run: |
          python -m proxy_app.main --host 127.0.0.1 --port 8000 &
          sleep 5
          
      - name: Run health check
        run: |
          python -m proxy_app.ci_health_check --url http://127.0.0.1:8000 --max-retries 5

  build-docker:
    runs-on: ubuntu-latest
    needs: [test, integration-test]
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        
      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
          
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:latest
            ghcr.io/${{ github.repository }}:${{ github.sha }}

  deploy-staging:
    runs-on: ubuntu-latest
    needs: build-docker
    environment: staging
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to staging
        run: |
          # Add your staging deployment commands here
          echo "Deploying to staging..."
          
      - name: Verify staging deployment
        run: |
          python -m proxy_app.ci_health_check --url ${{ secrets.STAGING_URL }} --api-key ${{ secrets.STAGING_API_KEY }}

  deploy-production:
    runs-on: ubuntu-latest
    needs: deploy-staging
    environment: production
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          # Add your production deployment commands here
          echo "Deploying to production..."
          
      - name: Verify production deployment
        run: |
          python -m proxy_app.ci_health_check --url ${{ secrets.PROD_URL }} --api-key ${{ secrets.PROD_API_KEY }}
```

## Required Secrets

Configure these secrets in your GitHub repository settings:

| Secret | Description |
|--------|-------------|
| `PROXY_API_KEY_TEST` | API key for testing in CI environment |
| `STAGING_URL` | Staging environment URL |
| `STAGING_API_KEY` | Staging API key for verification |
| `PROD_URL` | Production environment URL |
| `PROD_API_KEY` | Production API key for verification |
| `GITHUB_TOKEN` | Auto-provided, used for package registry |

## Local CI Simulation

Test the CI pipeline locally using [act](https://github.com/nektos/act):

```bash
# Install act
brew install act

# Run test job locally
act -j test

# Run integration test (requires Docker)
act -j integration-test
```

## Environment Configuration

### Test Configuration

For CI environments, create a minimal configuration to avoid loading heavy provider plugins:

```yaml
# config/ci_test_config.yaml
environment: ci
log_level: ERROR
features:
  enable_request_logging: false
  enable_tui: false
  enable_credential_tool: false
providers:
  # Use mock/test providers only
  test_mode: true
```

### Environment Variables

Set these environment variables in your CI environment:

```bash
# Required
export PROXY_API_KEY=test-key-ci-only

# Optional - disable heavy features for faster CI
export DISABLE_PROVIDER_DISCOVERY=1
export LITELLM_LOG=ERROR
```

## Health Check Integration

Use the built-in CI health check script to verify deployments:

```bash
# Basic health check
python -m proxy_app.ci_health_check --url http://localhost:8000

# With authentication
python -m proxy_app.ci_health_check --url https://api.example.com --api-key $API_KEY

# With retries for startup delays
python -m proxy_app.ci_health_check --url http://localhost:8000 --max-retries 10 --interval 5
```

## Database and State Management

For CI environments:

1. **Credential Management**: Use ephemeral credentials or mock providers
2. **Rate Limiting**: Disable or use in-memory stores
3. **Background Refresh**: Disable in CI to prevent unnecessary API calls

Example CI-specific environment variables:
```bash
export CREDENTIAL_STORE=memory
export DISABLE_BACKGROUND_REFRESH=1
export RATE_LIMITER_MODE=disabled
```

## Troubleshooting

### Common Issues

**Issue**: Health check fails in CI but works locally
- **Solution**: Ensure `--host 0.0.0.0` is used when starting the server in CI containers
- Check firewall rules in GitHub Actions runners

**Issue**: Provider discovery takes too long
- **Solution**: Set `DISABLE_PROVIDER_DISCOVERY=1` or use `config/ci_test_config.yaml`

**Issue**: LiteLLM timeout errors
- **Solution**: Increase timeout or use mock providers in CI

**Issue**: Port conflicts in integration tests
- **Solution**: Use dynamic port allocation or ensure cleanup between runs

## Best Practices

1. **Parallel Testing**: Run unit tests in parallel to speed up CI
2. **Caching**: Cache pip dependencies and Docker layers
3. **Security**: Never commit real API keys; use GitHub Secrets
4. **Artifacts**: Store test results and coverage reports as artifacts
5. **Notifications**: Configure Slack/Discord notifications for deployment status

## Maintenance

Keep this documentation updated when:
- Adding new environment variables
- Changing provider configurations
- Modifying the health check API
- Adding new deployment environments
