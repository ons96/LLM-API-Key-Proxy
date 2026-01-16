# LLM API Proxy with Virtual Models

## New Features (Phase 2-4)

### 1. Virtual Models & Intelligent Routing
The proxy now supports "Virtual Models" which are aliases that route to a chain of provider candidates. This allows for intelligent fallback and auto-ranking.

**Configuration:** `config/virtual_models.yaml`

```yaml
virtual_models:
  coding-smart:
    description: "Best models for complex coding tasks"
    fallback_chain:
      - provider: gemini
        model: gemini-1.5-pro
        priority: 1
      - provider: groq
        model: llama-3.3-70b-versatile
        priority: 2
```

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
