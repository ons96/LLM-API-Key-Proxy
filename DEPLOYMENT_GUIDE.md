# Complete Deployment Guide - LLM API Gateway

**Last Updated:** 2026-02-07  
**Your Use Case:** Personal coding assistant on laptop, running free on Oracle VPS

---

## Table of Contents
1. [Quick Answer: Where to Run](#quick-answer-where-to-run)
2. [Architecture Overview](#architecture-overview)
3. [Option 1: Run on Oracle VPS (Recommended)](#option-1-run-on-oracle-vps-recommended)
4. [Option 2: Run Locally in WSL](#option-2-run-locally-in-wsl)
5. [Option 3: Hybrid Setup](#option-3-hybrid-setup)
6. [API Keys Already Configured](#api-keys-already-configured)
7. [Updating Provider Models](#updating-provider-models)
8. [Web Search Configuration](#web-search-configuration)
9. [Using the Gateway](#using-the-gateway)
10. [Troubleshooting](#troubleshooting)

---

## Quick Answer: Where to Run

**RECOMMENDED: Run on Oracle VPS** ✅

**Why?**
- ✅ **Free forever** - Oracle Free Tier never expires
- ✅ **Always available** - 24/7 uptime even when laptop is off
- ✅ **Low latency** - Direct internet connection, no WSL overhead
- ✅ **No battery drain** - Laptop doesn't run the gateway
- ✅ **Access from anywhere** - Use from laptop, phone, or other devices

**Your Setup:**
- **VPS:** `ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233`
- **Gateway runs on VPS** → Listen on `0.0.0.0:8000`
- **Laptop connects via** → `http://40.233.101.233:8000/v1`
- **OpenCode, chatbots, Kobold, etc.** → All point to VPS

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Laptop (WSL/Windows)                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  OpenCode    │  │ Kobold Lite  │  │  SillyTavern │     │
│  └───────┬──────┘  └──────┬───────┘  └──────┬───────┘     │
│          │                │                  │             │
│          └────────────────┴──────────────────┘             │
│                           │                                │
│          All connect to: http://40.233.101.233:8000/v1     │
│                           │                                │
└───────────────────────────┼────────────────────────────────┘
                            │
                            │ Internet
                            │
┌───────────────────────────▼────────────────────────────────┐
│              Oracle Free VPS (40.233.101.233)              │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │     LLM API Gateway (running on port 8000)           │ │
│  │                                                       │ │
│  │  • Auto-fallback routing                             │ │
│  │  • Virtual models (coding-elite, chat-smart, etc.)   │ │
│  │  • Web search (Brave, Tavily, DuckDuckGo)            │ │
│  │  • Rate limit handling                               │ │
│  └──────────────────────────────────────────────────────┘ │
│                           │                                │
└───────────────────────────┼────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │   Groq   │      │ Cerebras │      │   G4F    │
    │  (Free)  │      │  (Free)  │      │  (Free)  │
    └──────────┘      └──────────┘      └──────────┘
```

---

## Option 1: Run on Oracle VPS (Recommended)

### Step 1: SSH into Your VPS

```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233
```

### Step 2: Navigate to Project Directory

```bash
cd ~/LLM-API-Key-Proxy
```

### Step 3: Pull Latest Changes (if you pushed from local)

```bash
git pull origin main
```

### Step 4: Create/Update .env File

Your `.env` file should already have these keys configured:

```bash
# Check existing keys
grep -E "^(GROQ_API_KEY|CEREBRAS_API_KEY|BRAVE_API_KEY|TAVILY_API_KEY)" .env

# If missing, add them:
nano .env
```

Required keys (you already have these):
```env
GROQ_API_KEY=your_groq_key
CEREBRAS_API_KEY=your_cerebras_key
BRAVE_API_KEY=your_brave_key
TAVILY_API_KEY=your_tavily_key

# Optional: Disable proxy authentication for personal use
# PROXY_API_KEY=  # Leave empty to disable auth
```

### Step 5: Install Dependencies (if not already installed)

```bash
# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
pip install -r requirements.txt
```

### Step 6: Start Gateway in Production Mode

**Option A: Run in background with nohup (simple)**

```bash
nohup python src/proxy_app/main.py --host 0.0.0.0 --port 8000 > ~/llm_proxy.log 2>&1 &
```

**Option B: Run with systemd (persistent across reboots)**

Create service file:
```bash
sudo nano /etc/systemd/system/llm-gateway.service
```

Paste this content:
```ini
[Unit]
Description=LLM API Gateway
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/LLM-API-Key-Proxy
Environment="PATH=/home/ubuntu/LLM-API-Key-Proxy/venv/bin"
ExecStart=/home/ubuntu/LLM-API-Key-Proxy/venv/bin/python src/proxy_app/main.py --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable llm-gateway
sudo systemctl start llm-gateway
sudo systemctl status llm-gateway
```

### Step 7: Verify Gateway is Running

```bash
# Check if port 8000 is listening
sudo netstat -tulpn | grep :8000

# Test gateway
curl http://localhost:8000/v1/models
```

### Step 8: Open Firewall (if needed)

```bash
# Allow port 8000
sudo ufw allow 8000/tcp
sudo ufw status
```

### Step 9: Test from Your Laptop

From WSL or Windows terminal:

```bash
curl http://40.233.101.233:8000/v1/models
```

---

## Option 2: Run Locally in WSL

**Use this if:** You want to test locally before deploying to VPS

### Step 1: Open WSL Terminal

```bash
cd ~/CodingProjects/LLM-API-Key-Proxy
```

### Step 2: Start Gateway (Local Development)

```bash
# Listen on localhost only (more secure for local testing)
python src/proxy_app/main.py --host 127.0.0.1 --port 8000

# OR listen on all interfaces (if you want to access from Windows)
python src/proxy_app/main.py --host 0.0.0.0 --port 8000
```

### Step 3: Configure OpenCode to Use Local Gateway

In OpenCode settings:
```json
{
  "model": "coding-elite",
  "provider": {
    "openai": {
      "name": "Local LLM Gateway",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "any-value-or-disable-auth"
      }
    }
  }
}
```

---

## Option 3: Hybrid Setup

**Best of both worlds:**

1. **Development:** Run locally in WSL for testing changes
2. **Production:** Push to GitHub → Pull on VPS → Restart gateway

**Workflow:**

```bash
# On laptop (WSL):
git add .
git commit -m "Update virtual models"
git push origin main

# On VPS:
cd ~/LLM-API-Key-Proxy
git pull origin main
sudo systemctl restart llm-gateway  # if using systemd
# OR
pkill -f "main.py" && nohup python src/proxy_app/main.py --host 0.0.0.0 --port 8000 > ~/llm_proxy.log 2>&1 &
```

---

## API Keys Already Configured

✅ **You already have these keys in your .env:**

- **GROQ_API_KEY** - Free tier with many models
- **CEREBRAS_API_KEY** - 1M tokens/day free
- **BRAVE_API_KEY** - 2,000 searches/month
- **TAVILY_API_KEY** - 1,000 searches/month

**Status:** All configured! No action needed.

---

## Updating Provider Models

### Problem: Static Model Lists Are Outdated

**Current Issue:**
- `config/router_config.yaml` has hardcoded `free_tier_models` lists
- Groq has 100+ models but only 4 are listed
- Cerebras has 6 models but only 2 are listed

### Good News: Providers Have Auto-Discovery! ✅

Both Groq and Cerebras providers **already support dynamic model discovery**:

**Groq Provider** (`src/rotator_library/providers/groq_provider.py`):
```python
async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
    response = await client.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    return [f"groq/{model['id']}" for model in response.json().get("data", [])]
```

**Cerebras Provider** (`src/rotator_library/providers/cerebras_provider.py`):
```python
async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
    response = await client.get(
        "https://api.cerebras.ai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    return [f"cerebras/{model['id']}" for model in response.json().get("data", [])]
```

### Discovered Models from Live APIs:

**Groq Models (100+ available):**
```
✅ groq/compound-mini
✅ groq/llama-4-maverick-17b-128e-instruct (NEW!)
✅ groq/llama-3.3-70b-versatile
✅ groq/llama-guard-4-12b
✅ groq/allam-2-7b
✅ groq/whisper-large-v3
✅ groq/gpt-oss-safeguard-20b (NEW!)
✅ groq/kimi-k2-instruct-0905 (NEW!)
... and 90+ more!
```

**Cerebras Models (6 available):**
```
✅ cerebras/qwen-3-235b-a22b-instruct-2507 (NEW! - Best for coding)
✅ cerebras/qwen-3-32b (NEW!)
✅ cerebras/llama-3.3-70b
✅ cerebras/llama3.1-8b
✅ cerebras/zai-glm-4.7 (NEW!)
✅ cerebras/gpt-oss-120b (NEW!)
```

### Action Item: Update router_config.yaml

**Replace the static lists with comprehensive lists:**

```yaml
  groq:
    enabled: true
    env_var: GROQ_API_KEY
    free_tier_models:
    # Keep existing
    - llama-3.1-8b-instant
    - llama-3.3-70b-versatile
    - mixtral-8x7b-32768
    - gemma2-9b-it
    # ADD THESE (discovered from API):
    - compound-mini
    - llama-4-maverick-17b-128e-instruct  # Llama 4 preview!
    - llama-guard-4-12b
    - allam-2-7b
    - whisper-large-v3  # Audio transcription
    - gpt-oss-safeguard-20b
    - kimi-k2-instruct-0905
    - llama-prompt-guard-2-86m
    # (Add more as needed - run curl command to see full list)

  cerebras:
    enabled: true
    env_var: CEREBRAS_API_KEY
    free_tier_models:
    # Keep existing
    - llama-3.3-70b
    - llama-3.1-70b
    # ADD THESE:
    - qwen-3-235b-a22b-instruct-2507  # Best for coding!
    - qwen-3-32b
    - zai-glm-4.7
    - gpt-oss-120b
    - llama3.1-8b
```

### How to Get Full List:

Run these commands on VPS or locally:

```bash
# Groq models
curl -s "https://api.groq.com/openai/v1/models" \
  -H "Authorization: Bearer $GROQ_API_KEY" \
  | jq -r '.data[].id'

# Cerebras models  
curl -s "https://api.cerebras.ai/v1/models" \
  -H "Authorization: Bearer $CEREBRAS_API_KEY" \
  | jq -r '.data[].id'
```

---

## Web Search Configuration

### Status: ✅ Already Configured!

Your `config/router_config.yaml` already has web search enabled:

```yaml
  brave_search:
    enabled: true
    env_var: BRAVE_API_KEY
    free_tier: true
  
  tavily:
    enabled: true
    env_var: TAVILY_API_KEY
    free_tier: true
    paid_tier: true
  
  duckduckgo:
    enabled: true
    free_tier: true
    no_api_key_required: true
```

**And your .env has the API keys!** ✅

### How Web Search Works

**Search provider implementations are in `src/proxy_app/router_core.py`:**

1. **BraveSearchProvider** (lines 183-265)
2. **TavilySearchProvider** (lines 268-350+)
3. **DuckDuckGoSearchProvider** (built-in, no API key needed)

**Automatic fallback order:**
- DuckDuckGo (free, no key) → Brave (2k/month) → Tavily (1k/month)

**When does search trigger?**
- When LLM requests a tool/function call for web search
- When virtual model has `search_enabled: true` (e.g., `router/best-research`)

**No configuration needed** - it's already working!

---

## Using the Gateway

### For OpenCode (Agentic Coding)

**Configuration:**
```json
{
  "model": "coding-elite",
  "provider": {
    "openai": {
      "name": "My VPS LLM Gateway",
      "options": {
        "baseURL": "http://40.233.101.233:8000/v1",
        "apiKey": "any-value"
      }
    }
  }
}
```

**Or if running locally:**
```json
{
  "baseURL": "http://127.0.0.1:8000/v1"
}
```

### For Kobold Lite / SillyTavern

```
API Type: OpenAI
API URL: http://40.233.101.233:8000/v1
Model: chat-smart (or chat-fast, coding-elite, etc.)
API Key: any-value (or leave empty if auth disabled)
```

### For Custom Python Scripts

```python
import openai

client = openai.OpenAI(
    base_url="http://40.233.101.233:8000/v1",
    api_key="any-value"
)

response = client.chat.completions.create(
    model="coding-elite",  # Auto-fallback across 10+ providers!
    messages=[{"role": "user", "content": "Write a Python function"}]
)
```

### Available Virtual Models

| Model | Best For | Fallback Chain |
|-------|----------|----------------|
| `coding-elite` | Agentic coding, complex tasks | Claude Opus 4.5 → Gemini 3 Pro → GPT-5.2 → ... (10+ providers) |
| `coding-fast` | Quick edits, simple tasks | Fast providers, speed-optimized |
| `chat-smart` | Intelligent conversation | High intelligence models |
| `chat-fast` | Quick responses | Low latency models |

---

## Troubleshooting

### Gateway Not Starting

```bash
# Check if already running
ps aux | grep main.py

# Kill existing process
pkill -f "main.py"

# Check logs
tail -f ~/llm_proxy.log

# Test manually
python src/proxy_app/main.py --host 127.0.0.1 --port 8000
```

### Port 8000 Already in Use

```bash
# Find process using port 8000
sudo lsof -i :8000

# Kill it
sudo kill -9 <PID>
```

### Can't Connect from Laptop

```bash
# On VPS: Check firewall
sudo ufw status
sudo ufw allow 8000/tcp

# On VPS: Check if listening on 0.0.0.0
sudo netstat -tulpn | grep :8000

# Should show: 0.0.0.0:8000 (not 127.0.0.1:8000)
```

### Models Not Working

```bash
# Test specific provider
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "groq/llama-3.3-70b-versatile",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
  }'

# Check logs for errors
tail -f ~/llm_proxy.log
```

### Update Models After Editing Config

```bash
# Restart gateway
sudo systemctl restart llm-gateway

# Or if using nohup:
pkill -f "main.py"
nohup python src/proxy_app/main.py --host 0.0.0.0 --port 8000 > ~/llm_proxy.log 2>&1 &
```

---

## Difference: Local Dev vs Production

| Aspect | Local Development | Production (VPS) |
|--------|-------------------|------------------|
| **Host** | `127.0.0.1` (localhost only) | `0.0.0.0` (all interfaces) |
| **Port** | 8000 (any port) | 8000 (standard) |
| **Process** | `python src/proxy_app/main.py` | `nohup ... &` or systemd service |
| **Logs** | Terminal output | `~/llm_proxy.log` or journalctl |
| **Restart** | Ctrl+C and re-run | `systemctl restart` or `pkill + nohup` |
| **Auto-start** | Manual | systemd enables on boot |
| **Access** | Only from same machine | From internet (via VPS IP) |
| **Use Case** | Testing changes | Always-on production use |

---

## Summary: Your Personal Setup

**Recommended Configuration:**

```
┌──────────────────────────────────────────────────┐
│  1. Gateway runs on Oracle VPS (free forever)    │
│     ssh ubuntu@40.233.101.233                    │
│     cd ~/LLM-API-Key-Proxy                       │
│     systemctl start llm-gateway                  │
│                                                  │
│  2. Laptop connects to VPS gateway              │
│     http://40.233.101.233:8000/v1               │
│                                                  │
│  3. All tools point to VPS:                     │
│     - OpenCode                                   │
│     - Kobold Lite                                │
│     - SillyTavern                                │
│     - Custom scripts                             │
│                                                  │
│  4. API keys in .env on VPS:                    │
│     ✅ GROQ_API_KEY                              │
│     ✅ CEREBRAS_API_KEY                          │
│     ✅ BRAVE_API_KEY                             │
│     ✅ TAVILY_API_KEY                            │
│                                                  │
│  5. Auto-fallback works automatically:          │
│     Groq → Cerebras → G4F → ... (10+ providers) │
└──────────────────────────────────────────────────┘
```

**Next Steps:**
1. ✅ SSH into VPS
2. ✅ Pull latest code (`git pull`)
3. ✅ Update `config/router_config.yaml` with new models
4. ✅ Start gateway with systemd or nohup
5. ✅ Point OpenCode to `http://40.233.101.233:8000/v1`
6. ✅ Start coding with free, auto-fallback LLMs!
