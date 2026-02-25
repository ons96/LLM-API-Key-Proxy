# LLM API Proxy with Virtual Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT/LGPL-3.0](https://img.shields.io/badge/license-MIT%20%7C%20LGPL--3.0-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![GitHub issues](https://img.shields.io/github/issues/ons96/LLM-API-Key-Proxy.svg)](https://github.com/ons96/LLM-API-Key-Proxy/issues)

**A FastAPI-based LLM API proxy with virtual models and automatic fallback. Route requests through 10+ free LLM providers with intelligent fallback chains.**

## Quick Start

```bash
# Clone and install
git clone https://github.com/ons96/LLM-API-Key-Proxy.git
cd LLM-API-Key-Proxy
pip install -r requirements.txt

# Start the server (no API keys required!)
python src/proxy_app/main.py --host 0.0.0.0 --port 8000

# Test it works
curl http://localhost:8000/v1/models
```

### One-Minute Example

```bash
# Send a chat completion request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "coding-elite",
    "messages": [{"role": "user", "content": "Write a Python function to reverse a string"}]
  }'
```

**That's it!** No API keys needed - works with free providers out of the box.

## Features

- **Virtual Models**: Pre-configured model aliases with automatic fallback chains
- **Free Tier Support**: Works with Groq, G4F, Gemini, and more - no paid keys required
- **Automatic Fallback**: Seamlessly switches providers on rate limits or errors
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API
- **Multiple Providers**: Groq, Gemini, G4F, GitHub Models, Cloudflare Workers AI
- **Web Search**: DuckDuckGo, Brave, Tavily integration for AI tools
- **Telemetry**: Built-in tracking of requests, tokens, and costs

## Virtual Models

Virtual models are pre-configured aliases that automatically select the best available provider:

| Model | Use Case | Fallback Chain |
|-------|----------|----------------|
| `coding-elite` | Agentic coding, complex code | Groq → Gemini → G4F |
| `coding-fast` | Quick edits, simple tasks | Groq → Gemini → G4F |
| `chat-smart` | Complex reasoning, analysis | Groq → Gemini |
| `chat-fast` | Quick responses | Groq → Gemini → G4F |
| `chat-rp` | Roleplay | Specialized models |

### How Fallback Works

```
Request → Groq (primary) → Rate limited? → Gemini (backup) → Error? → G4F (fallback)
```

1. Request goes to priority 1 provider (e.g., Groq)
2. If rate limited or down → automatically try next provider
3. If still failing → try fallback provider
4. No manual intervention needed - seamless failover

## API Examples

### Chat Completions

```bash
# Using virtual model (recommended)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "coding-elite",
    "messages": [{"role": "user", "content": "Explain async/await in Python"}]
  }'

# Using specific provider
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "groq/llama-3.3-70b-versatile",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# With streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "coding-fast",
    "messages": [{"role": "user", "content": "Write a haiku about code"}],
    "stream": true
  }'
```

### List Available Models

```bash
curl http://localhost:8000/v1/models
```

### Health Check

```bash
curl http://localhost:8000/stats
```

### Responses API (OpenCode Compatible)

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "coding-elite",
    "input": [{"type": "message", "role": "user", "content": [{"type": "text", "text": "Hello"}]}]
  }'
```

## Integration Examples

### With OpenCode

Add to your OpenCode config (`~/.config/opencode/opencode.json`):

```json
{
  "model": "openai/coding-elite",
  "provider": {
    "openai": {
      "name": "LLM Gateway",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "any-value"
      }
    }
  }
}
```

### With Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="any-value"  # Not validated in default config
)

response = client.chat.completions.create(
    model="coding-elite",
    messages=[{"role": "user", "content": "Write a function"}]
)
print(response.choices[0].message.content)
```

### With LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model_name="coding-elite",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="any-value"
)

response = llm.invoke("What is the meaning of life?")
print(response.content)
```

## Configuration

### Environment Variables

Create a `.env` file (copy from `.env.example`):

```bash
# Optional: Add API keys for more providers
GEMINI_API_KEY=your_gemini_key        # Google Gemini
GITHUB_MODELS_API_KEY=ghp_xxx         # GitHub Models
CLOUDFLARE_ACCOUNT_ID=xxx             # Cloudflare Workers AI
CLOUDFLARE_API_TOKEN=xxx
BRAVE_API_KEY=xxx                     # Brave Search
TAVILY_API_KEY=xxx                    # Tavily Search
```

### Virtual Model Configuration

Edit `config/virtual_models.yaml` to customize fallback chains:

```yaml
virtual_models:
  coding-elite:
    description: "Best agentic coding performance"
    candidates:
      - provider: groq
        model: llama-3.3-70b-versatile
        priority: 1
      - provider: gemini
        model: gemini-2.5-pro
        priority: 2
      - provider: g4f
        model: gpt-4
        priority: 3
```

### Router Configuration

Edit `config/router_config.yaml` to enable/disable providers:

```yaml
providers:
  groq:
    enabled: true
    free: true
  gemini:
    enabled: true
    free: true  # Has free tier
  openai:
    enabled: false  # Requires paid API key
```

## Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run in foreground
python src/proxy_app/main.py --host 0.0.0.0 --port 8000

# Run with request logging
python src/proxy_app/main.py --host 0.0.0.0 --port 8000 --enable-request-logging
```

### Production (VPS)

```bash
# SSH into your server
ssh user@your-server

# Clone and setup
git clone https://github.com/ons96/LLM-API-Key-Proxy.git
cd LLM-API-Key-Proxy
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run in background
nohup python src/proxy_app/main.py --host 0.0.0.0 --port 8000 > llm_proxy.log 2>&1 &

# View logs
tail -f llm_proxy.log

# Stop server
pkill -f "src/proxy_app/main.py"
```

### Docker

```bash
# Using docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Troubleshooting

### Gateway not starting

```bash
# Check if port is in use
lsof -i :8000

# Verify dependencies
pip install -r requirements.txt
```

### Model requests failing

```bash
# Check server logs
tail -f /tmp/llm_proxy.log

# Check provider health
curl http://localhost:8000/stats

# Try a specific provider
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "groq/llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "test"}]}'
```

### Rate limiting

The proxy automatically falls back to other providers. No manual intervention needed.

## Virtual Model Status

| Model | Status | Primary Provider |
|-------|--------|------------------|
| `coding-elite` | ✅ Working | Groq |
| `coding-fast` | ✅ Working | Groq |
| `chat-smart` | ✅ Working | Groq |
| `chat-fast` | ✅ Working | Groq |

## Telemetry

The gateway tracks requests via SQLite (`/tmp/llm_proxy_telemetry.db`):

- API calls with timing (response time, time-to-first-token)
- Success/failure rates per provider
- Token usage (input/output) and tokens-per-second
- Cost estimates per request
- Provider health status and failure rates

## Project Structure

```
LLM-API-Key-Proxy/
├── config/              # YAML configs (router, virtual models)
├── src/
│   ├── proxy_app/       # Main gateway application
│   │   ├── main.py           # FastAPI app entry point
│   │   ├── router_core.py    # Router logic
│   │   └── launcher_tui.py   # Terminal UI launcher
│   └── rotator_library/ # Core library
│       ├── client.py         # RotatingClient
│       ├── credential_tool.py # OAuth management
│       └── providers/        # Provider adapters
├── tests/               # Test fixtures
├── docs/                # Documentation
└── .env.example         # Environment template
```

## License

This project has dual licensing:

- **`src/rotator_library/`**: [LGPL-3.0](src/rotator_library/COPYING)
- **`src/proxy_app/`**: [MIT](src/proxy_app/LICENSE)

## Contributing

Issues and pull requests are welcome! See [GitHub Issues](https://github.com/ons96/LLM-API-Key-Proxy/issues) for open tasks.

## Verification

Run the self-verification script to ensure all components are wired correctly:

```bash
python verify_installation.py
```
