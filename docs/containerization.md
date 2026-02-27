# Containerization Guide

## Building the Image

The Dockerfile uses a multi-stage build to minimize the final image size:

```bash
docker build -t proxy-app:latest .
```

## Running the Container

### Basic Usage

```bash
docker run -p 8000:8000 \
  -e PROXY_API_KEY=your_secret_key \
  proxy-app:latest
```

### With Environment Files

Mount your environment files instead of baking them into the image:

```bash
docker run -p 8000:8000 \
  -v $(pwd)/.env:/app/.env:ro \
  -v $(pwd)/config:/app/config:ro \
  proxy-app:latest
```

### Using Docker Compose

```yaml
version: '3.8'

services:
  proxy:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Multi-Stage Build Benefits

1. **Smaller Image Size**: Only runtime dependencies are included in the final image
2. **Security**: Build tools (gcc, etc.) are not present in the runtime image
3. **Non-root User**: Application runs as `appuser` (UID 1000) for security
4. **Layer Caching**: Dependencies are cached separately from application code
5. **BuildKit Cache**: Pip cache is mounted during build for faster rebuilds
