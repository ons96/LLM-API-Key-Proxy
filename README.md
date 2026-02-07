# LLM API Proxy with Virtual Models

## Core Feature: Agentic Coding with Automatic Fallback

**WORKING VIRTUAL MODEL:** Use `coding-elite` for agentic coding tasks with automatic fallback between free providers.

### Available Virtual Models for OpenCode:

#### ✅ coding-elite (Recommended for Agentic Coding)
- **Primary Use:** Agentic coding, complex code generation, debugging
- **Fallback Chain:**
  1. Groq (llama-3.3-70b-versatile) - Free, fast, high quality
  2. Gemini (gemini-3-pro) - Google free tier (optional, requires API key)
  3. G4F (gpt-4) - Free fallback model

#### coding-fast (Quick coding tasks)
- **Primary Use:** Fast code completion, simple refactoring
- **Fallback Chain:**
  1. Groq (llama-3.1-8b-instant) - Free, ultra fast
  2. Gemini (gemini-2-5-flash) - Google free tier (optional)
  3. G4F (gpt-4) - Free fallback

#### chat-smart (High intelligence chat)
- **Primary Use:** Complex reasoning, analysis, research
- **Fallback Chain:**
  1. Groq (llama-3.3-70b-versatile) - Free
  2. Gemini (gemini-3-pro) - Google free tier (optional)

#### chat-fast (Quick chat)
- **Primary Use:** Quick responses, simple queries
- **Fallback Chain:**
  1. Groq (llama-3.1-8b-instant) - Free, fast
  2. Gemini (gemini-2-5-flash) - Google free tier (optional)
  3. G4F (gpt-3.5-turbo) - Free fallback

### How Fallback Works

When using virtual models like `coding-elite`:
1. Request goes to priority 1 provider (Groq)
2. If rate limited/down → automatically try next provider (Gemini)
3. If still down → try fallback provider (G4F)
4. No manual intervention needed - seamless failover

## Configuration

### Quick Start (No API Keys Required)

The proxy works out-of-the-box with free providers:
- Groq (pre-configured, no key needed for free tier)
- G4F (pre-configured, no key needed)

Simply start the server and use `coding-elite`:
```bash
python src/proxy_app/main.py --host 0.0.0.0 --port 8000
```

### Optional: Add More Providers

Add these to your `.env` file for additional fallback options:

#### GitHub Models (Free but rate-limited)
```bash
GITHUB_MODELS_API_BASE=https://models.github.ai/inference
GITHUB_MODELS_API_KEY=ghp_your_token_here
# Then set enabled: true in config/router_config.yaml for github-models
```

#### Cloudflare Workers AI (10,000 Neurons/day free)
```bash
CLOUDFLARE_ACCOUNT_ID=your-account-id
CLOUDFLARE_API_TOKEN=your-api-token
# Then set enabled: true in config/router_config.yaml for cloudflare
```

#### Google Gemini (API key required)
```bash
GEMINI_API_KEY=your_gemini_key_here
```

### Web Search Providers (Optional)

The gateway supports web search functionality for AI tools. Three free providers are available:

#### DuckDuckGo (FREE, No API Key Required)
DuckDuckGo search works out of the box - no configuration needed. Automatically enabled in `config/router_config.yaml`.

#### Brave Search (2,000 free requests/month)
```bash
BRAVE_API_KEY=your_brave_api_key_here
# Get your key at: https://api.search.brave.com/app/keys
```

#### Tavily Search (1,000 free searches/month)
```bash
TAVILY_API_KEY=your_tavily_api_key_here
# Get your key at: https://tavily.com/
```

**Web search fallback order:** DuckDuckGo → Brave → Tavily (falls back automatically if one fails)

## Deployment

### Local Development
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure (Optional):**
   - Copy `.env.example` to `.env` and add your API keys
   - Basic usage works with free providers (Groq, G4F, DuckDuckGo)
   - Add Brave/Tavily/Gemini keys for more options
3. **Run:**
   ```bash
   python src/proxy_app/main.py --host 0.0.0.0 --port 8000
   ```

### With OpenCode
To use this gateway in OpenCode (AI coding assistant):

1. Start the proxy server:
   ```bash
   python src/proxy_app/main.py --host 127.0.0.1 --port 8000
   ```

2. Configure OpenCode settings:
   ```yaml
   base_url: http://127.0.0.1:8000/v1
   model: coding-elite  # for agentic coding
   api_key: any-value  # not used when PROXY_API_KEY is disabled
   ```

3. Server URL format: `http://127.0.0.1:8000/v1`

4. Available models for OpenCode:
   - `coding-elite` - Best for agentic coding (Groq → Gemini → G4F fallback)
   - `coding-fast` - Quick coding tasks
   - `chat-smart` - High intelligence reasoning
   - `chat-fast` - Low latency chat
   - `chat-rp` - NSFW roleplay mode

**OpenCode will see all virtual models in the model dropdown.**

### Virtual Models Explained

The gateway provides **intelligent virtual models** that automatically select the best available provider:

**coding-elite** - Best agentic coding performance
- Minimum 70.0 SWE-bench score required
- Scoring: 80% agentic + 15% TPS + 5% availability
- Fallback order: Claude Opus 4.5 → Gemini 3 Pro → GPT-5.2 → GPT-4o → ...

**coding-smart** - Smart coding with balanced performance  
- Minimum 65.0 SWE-bench score required
- Same scoring formula as coding-elite
- More fallback options than coding-elite

**coding-fast** - Speed-focused coding
- No minimum score (pure speed optimization)
- Prioritizes TPS over coding performance
- Best for quick edits and simple tasks

**chat-smart** - Highest intelligence for reasoning
- Based on Artificial Analysis rankings
- Intelligence score priority (Chatbot Arena, MMLU, etc.)
- Best for complex reasoning and analysis

**chat-fast** - Efficient chat (intelligence ÷ response time)
- Efficiency ratio: Intelligence / Response Time
- Best models that are both smart AND fast
- Gemini 2 Flash, Mistral Medium 3.1, Llama 4 Scout

**chat-rp** - Roleplay optimized
- UGI leaderboard models for NSFW RP
- MN-Violet-Lotus, Violet-Twilight, etc.

### Production Deployment
To run in background (persistent across sessions):
```bash
cd /home/owens/CodingProjects/LLM-API-Key-Proxy
source venv/bin/activate
nohup python src/proxy_app/main.py --host 127.0.0.1 --port 8000 > /tmp/llm_proxy.log 2>&1 &
```

View logs:
```bash
tail -f /tmp/llm_proxy.log
```

Stop server:
```bash
pkill -f "src/proxy_app/main.py"
```

### Docker Deployment
```bash
docker-compose up -d
```

## API Endpoints

### Chat Completions
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "coding-elite",
    "messages": [{"role": "user", "content": "Write a Python function"}]
  }'
```

### Responses API (OpenCode Compatible)
The gateway supports OpenAI Responses API for OpenCode integration:
```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "coding-elite",
    "input": [{"type": "message", "role": "user", "content": [{"type": "text", "text": "Hello"}]}]
  }'
```

### List Models (includes virtual models)
```bash
curl http://localhost:8000/v1/models
```

### Health Status
```bash
curl http://localhost:8000/stats
```

## Troubleshooting

### Gateway not starting
- Check port 8000 is not in use
- Verify dependencies are installed: `pip install -r requirements.txt`

### Model requests failing
- Check server logs for error details
- Verify provider health: `GET /stats` endpoint (if available)
- try using a specific provider directly: `groq/llama-3.3-70b-versatile`

### Rate limiting
- The proxy automatically falls back to other providers
- No manual intervention needed
- Adjust virtual model priorities in `config/router_config.yaml` if needed

## Virtual Model Status

| Model | Status | Primary Provider |
|-------|--------|------------------|
| coding-elite | ✅ Working | Groq |
| coding-fast | ✅ Working | Groq |
| chat-smart | ⚠️ Known Issue | Groq (config issue) |
| chat-fast | ✅ Working | Groq |
| router/best-coding | ⚠️ Known Issue | Multiple providers |

**For agentic coding in OpenCode:** Use `coding-elite`

## Telemetry System

The gateway includes comprehensive telemetry tracking via SQLite database (`/tmp/llm_proxy_telemetry.db`):

**Tracked Metrics:**
- ✅ API calls with timing (response time, time-to-first-token)
- ✅ Success/failure rates per provider
- ✅ Token usage (input/output) and tokens-per-second
- ✅ Cost estimates per request
- ✅ Provider health status and failure rates
- ✅ Rate limit tracking per provider

**Database Location:** `/tmp/llm_proxy_telemetry.db`

**Note:** Telemetry infrastructure exists and is ready for use. Integration with request pipeline is planned for automatic recording of all metrics.

**Usage:**
Request `model: "coding-smart"` in your OpenAI client.

### 2. Auto-Ranking (Intelligent Model Ordering)
Enabled via `auto_order: true` in virtual model config. The router dynamically re-orders candidates based on benchmark scores (SWE-bench, HumanEval) appropriate for the task (coding, chat, etc.).

**Rankings Data:** `config/model_rankings.yaml`

### 3. Rate Limit Tracking
A new `RateLimitTracker` monitors usage (RPM, Daily) and enforces limits defined in `virtual_models.yaml`. It prevents hitting provider 429s by proactively rotating or rejecting requests.

### 4. Health Checks
A background `HealthChecker` periodically pings configured providers.
**Status Endpoint:** `GET /stats` returns real-time health and usage data.

## Deployment

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure:**
    -   Edit `.env` with your API keys.
    -   Edit `config/virtual_models.yaml` to customize routing.
3.  **Run:**
    ```bash
    python src/proxy_app/main.py
    ```

## Verification
Run the self-verification script to ensure all components are wired correctly:
```bash
python verify_installation.py
```
