# Docker Deployment Guide

This guide covers containerizing and deploying the Mirro-Proxy application using Docker and Docker Compose.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose v2.0+ (recommended)
- Existing `requirements.txt` in project root with all Python dependencies

## Quick Start

1. Copy the example Docker files to the project root:
   ```bash
   cp docs/examples/Dockerfile .
   cp docs/examples/docker-compose.yml .
   ```

2. Create a `requirements.txt` if you haven't already:
   ```txt
   fastapi>=0.104.0
   uvicorn[standard]>=0.24.0
   litellm>=1.0.0
   rich>=13.0.0
   pydantic>=2.0.0
   colorlog>=6.0.0
   python-dotenv>=1.0.0
   pyyaml>=6.0
   ```

3. Build and run:
   ```bash
   docker-compose up --build -d
   ```

## Dockerfile Configuration

The Dockerfile uses a multi-stage approach for optimization:

- **Base Image**: `python:3.11-slim` for balance of size and compatibility
- **Port**: Exposes 8000
- **Non-root user**: Runs as `appuser` for security (optional, see Production section)
- **Volumes**: Expects `config/` and `.env` to be mounted

### Key Features

1. **Layer Caching**: Copies `requirements.txt` before source code to leverage Docker layer caching
2. **PYTHONPATH**: Set to `/app/src` to ensure imports work correctly
3. **Health Checks**: Built-in healthcheck using the application's status endpoint

## Docker Compose Setup

The provided `docker-compose.yml` includes:

- **Environment Management**: Loads variables from `.env` file
- **Configuration Mounting**: Read-only mount of `config/` directory
- **Port Mapping**: Maps host port 8000 to container port 8000
- **Restart Policy**: `unless-stopped` for production stability
- **Health Checks**: Monitors application availability

### Environment Variables

Required environment variables (via `.env` file or `environment` section):

- `PROXY_API_KEY`: Authentication key for the proxy (required)
- Provider-specific keys (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.) depending on your configured providers

### Volume Mounts

1. **config/**: Mount your configuration directory (read-only recommended)
   - Contains: `aliases.yaml`, `model_rankings.yaml`, `providers_database.yaml`, etc.
   
2. **.env**: Mount your environment file (read-only recommended)
   - Contains API keys and sensitive configuration

## Building the Image

### Development Build
```bash
docker build -t mirro-proxy:latest .
```

### Production Build
```bash
docker build -t mirro-proxy:latest --no-cache .
```

## Running the Container

### Using Docker Run (Simple)
```bash
docker run -d \
  --name mirro-proxy \
  -p 8000:8000 \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/.env:/app/.env:ro \
  -e PROXY_API_KEY=your-secret-key \
  mirro-proxy:latest
```

### Using Docker Compose (Recommended)
```bash
# Start in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Production Deployment

### Security Hardening

1. **Non-root User**: Uncomment the user creation in Dockerfile:
   ```dockerfile
   RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
   USER appuser
   ```

2. **Read-only Filesystem**: Add to docker-compose:
   ```yaml
   services:
     proxy:
       read_only: true
       tmpfs:
         - /tmp
   ```

3. **Resource Limits**:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 2G
       reservations:
         cpus: '1'
         memory: 512M
   ```

### Health Monitoring

The container includes a healthcheck that pings the `/health` endpoint every 30 seconds. Monitor container health with:
```bash
docker ps
# or
docker-compose ps
```

### Scaling

For high availability, use Docker Swarm or Kubernetes:

```bash
# Docker Swarm example
docker stack deploy -c docker-compose.yml mirro-proxy
```

## Configuration Management

### External Configuration

Mount configuration files as volumes rather than copying into the image:

```yaml
volumes:
  - ./config:/app/config:ro
  - ./custom_config.yaml:/app/config/custom_config.yaml:ro
```

### Environment-specific Settings

Use Docker Compose overrides for different environments:

**docker-compose.prod.yml**:
```yaml
version: '3.8'
services:
  proxy:
    restart: always
    deploy:
      replicas: 2
```

Run with:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Troubleshooting

### Container exits immediately

Check logs:
```bash
docker-compose logs proxy
```

Common causes:
- Missing `PROXY_API_KEY` environment variable
- Missing config files in mounted volume
- Port 8000 already in use on host

### Permission Denied on Config Files

Ensure the config files are readable:
```bash
chmod -R 755 config/
```

### Module Not Found Errors

Ensure `PYTHONPATH` is set correctly in Dockerfile and `src/` directory is properly copied.

### Memory Issues

For large model operations, increase container memory:
```yaml
deploy:
  resources:
    limits:
      memory: 4G
```

## Updating

To update the application:

1. Pull latest code changes
2. Rebuild image:
   ```bash
   docker-compose up --build -d
   ```
3. Or for zero-downtime (Swarm mode):
   ```bash
   docker service update --force mirro-proxy_proxy
   ```

## Networking

The container exposes port 8000. To use with reverse proxy (nginx/traefik):

```yaml
# docker-compose with nginx
version: '3.8'
services:
  proxy:
    build: .
    expose:
      - "8000"
    networks:
      - backend
      
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    networks:
      - backend
    depends_on:
      - proxy

networks:
  backend:
    driver: bridge
```
