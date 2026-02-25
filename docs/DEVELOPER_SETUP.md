# Developer Setup Guide

This guide will help you set up the LLM API Proxy for local development.

## Prerequisites

- **Python 3.10+** (3.10, 3.11, or 3.12 recommended)
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- **Virtual environment** (venv or conda)

### Verify Prerequisites

```bash
# Check Python version (must be 3.10+)
python --version
# or
python3 --version

# Check pip
pip --version

# Check Git
git --version
```

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/ons96/LLM-API-Key-Proxy.git
cd LLM-API-Key-Proxy

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Copy environment template
cp .env.example .env
# Edit .env with your API keys if needed

# 5. Run the server
python src/proxy_app/main.py --host 0.0.0.0 --port 8000

# 6. Test it works
curl http://localhost:8000/v1/models
```

## Detailed Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/ons96/LLM-API-Key-Proxy.git
cd LLM-API-Key-Proxy
```

### Step 2: Set Up Virtual Environment

Using venv (recommended):

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate   # Windows PowerShell
# or
venv\Scripts\activate.bat # Windows CMD
```

Using conda:

```bash
# Create environment
conda create -n llm-proxy python=3.11

# Activate it
conda activate llm-proxy
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **LiteLLM** - LLM API abstraction
- **G4F** - Free LLM fallback
- **Rich** - Terminal UI
- **HTTPX/AIOHTTP** - Async HTTP clients

### Step 4: Configure Environment (Optional)

The proxy works in **free-only mode** by default without any API keys. To add providers:

```bash
# Copy the example file
cp .env.example .env

# Edit with your API keys
nano .env  # or use your preferred editor
```

#### Minimal Configuration (Free Tier)

No configuration needed! The proxy works out of the box with free providers:

```bash
# Just run it - no .env file required
python src/proxy_app/main.py
```

#### With API Keys (Enhanced Access)

Create `.env` with your keys:

```env
# Proxy authentication (recommended for production)
PROXY_API_KEY=your-secret-key-here

# Optional: Add provider keys for more models
GROQ_API_KEY_1=gsk_xxx           # Groq (free tier available)
GEMINI_API_KEY_1=xxx             # Google Gemini (free tier available)
OPENAI_API_KEY_1=sk-xxx          # OpenAI (paid)
```

### Step 5: Run the Server

#### Basic Run

```bash
python src/proxy_app/main.py
```

#### With Options

```bash
# Custom host/port
python src/proxy_app/main.py --host 127.0.0.1 --port 8080

# With request logging (for debugging)
python src/proxy_app/main.py --enable-request-logging

# Add OAuth credentials interactively
python src/proxy_app/main.py --add-credential
```

#### Interactive Launcher (TUI)

Running without arguments launches the interactive terminal UI:

```bash
python src/proxy_app/main.py
```

This provides a menu to:
- Configure settings
- Add credentials
- Start the server
- View logs

### Step 6: Verify Installation

```bash
# Check the health endpoint
curl http://localhost:8000/stats

# List available models
curl http://localhost:8000/v1/models

# Test a chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "coding-fast",
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```

## Project Structure

```
LLM-API-Key-Proxy/
├── config/                    # Configuration files
│   ├── router_config.yaml     # Provider settings
│   └── virtual_models.yaml    # Virtual model definitions
├── src/
│   ├── proxy_app/             # Main gateway application
│   │   ├── main.py            # Entry point
│   │   ├── router_core.py     # Request routing logic
│   │   ├── settings_tool.py   # Settings management
│   │   └── launcher_tui.py    # Terminal UI
│   └── rotator_library/       # Core library
│       ├── client.py          # RotatingClient
│       ├── credential_tool.py # OAuth management
│       ├── usage_manager.py   # Usage tracking
│       └── providers/         # Provider implementations
├── oauth_creds/               # OAuth credentials (auto-created)
├── tests/                     # Test files
├── docs/                      # Documentation
├── .env.example               # Environment template
└── requirements.txt           # Python dependencies
```

## Configuration Files

### config/router_config.yaml

Enables/disables providers and sets defaults:

```yaml
providers:
  groq:
    enabled: true
    free: true
  gemini:
    enabled: true
    free: true
  openai:
    enabled: false  # Requires paid API key
```

### config/virtual_models.yaml

Defines virtual model aliases:

```yaml
virtual_models:
  coding-elite:
    description: "Best for agentic coding"
    candidates:
      - provider: groq
        model: llama-3.3-70b-versatile
        priority: 1
      - provider: gemini
        model: gemini-2.5-pro
        priority: 2
```

### .env File

See `.env.example` for all available options. Key variables:

| Variable | Purpose | Required |
|----------|---------|----------|
| `PROXY_API_KEY` | Authenticate requests to proxy | No (but recommended) |
| `GROQ_API_KEY_1` | Groq API access | No |
| `GEMINI_API_KEY_1` | Gemini API access | No |
| `OPENAI_API_KEY_1` | OpenAI API access | No |
| `LOG_LEVEL` | Logging verbosity | No (default: INFO) |

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_router.py

# With coverage
pytest --cov=src tests/
```

### Code Style

This project uses:
- **Black** for formatting (88-char line limit)
- Type hints for function signatures

```bash
# Format code
black src/

# Check types
mypy src/
```

### Logging

Logs are written to stdout. For debugging:

```bash
# Enable request logging
python src/proxy_app/main.py --enable-request-logging

# Or set log level in .env
LOG_LEVEL=DEBUG
```

## OAuth Credential Setup

For providers that use OAuth (Gemini CLI, Qwen Code, etc.):

### Interactive Setup

```bash
python src/proxy_app/main.py --add-credential
```

### Manual Setup

1. Place credential JSON files in `oauth_creds/` directory
2. Files are automatically detected and loaded

### Supported OAuth Providers

| Provider | Credential Source | File Name Pattern |
|----------|------------------|-------------------|
| Gemini CLI | `~/.gemini/credentials.json` | `gemini_cli_oauth_*.json` |
| Qwen Code | `~/.qwen/oauth_creds.json` | `qwen_code_oauth_*.json` |
| iFlow | `~/.iflow/oauth_creds.json` | `iflow_oauth_*.json` |

## Troubleshooting

### Import Errors

```bash
# Ensure virtual environment is active
which python  # Should point to venv/bin/python

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Use a different port
python src/proxy_app/main.py --port 8080
```

### No Models Available

1. Check your internet connection
2. Verify provider status: `curl http://localhost:8000/stats`
3. Check logs for errors

### OAuth Token Expired

```bash
# Re-add the credential
python src/proxy_app/main.py --add-credential

# Or delete and re-authenticate
rm oauth_creds/gemini_cli_oauth_1.json
```

## IDE Setup

### VS Code

Recommended extensions:
- Python (Microsoft)
- Pylance
- Python Debugger

Settings (`.vscode/settings.json`):

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "editor.formatOnSave": true
}
```

### PyCharm

1. Open project directory
2. Set Python interpreter to `venv/bin/python`
3. Enable Black formatter in settings

## Next Steps

- Read the [API Documentation](./API.md) for endpoint details
- See [DEPLOYMENT.md](./DEPLOYMENT.md) for production setup
- Check [VIRTUAL_MODELS_PLAN.md](../VIRTUAL_MODELS_PLAN.md) for advanced configuration

## Getting Help

- **GitHub Issues**: https://github.com/ons96/LLM-API-Key-Proxy/issues
- **Documentation**: See `docs/` directory
