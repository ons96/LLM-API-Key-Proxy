# Barebones Proxy Deployment

This package contains the essential components to run the LLM API Key Proxy on a minimal VPS.

## Prerequisites
- Python 3.9+
- A `.env` file with your API keys. Use `simple-env-template.env` as a reference.

## Installation
1. Extract the package: `tar -xzf barebones_proxy.tar.gz`
2. Install dependencies: `pip install -r requirements.txt`

## Running the Proxy
For minimal resource usage, run uvicorn directly:

```bash
python -m uvicorn src.proxy_app.main:app --host 0.0.0.0 --port 8000
```

Or using the launcher if you want the TUI:
```bash
python src/proxy_app/launcher_tui.py
```

## Configuration
Essential configs are in `config/router_config.yaml` and `config/model_rankings.yaml`.
