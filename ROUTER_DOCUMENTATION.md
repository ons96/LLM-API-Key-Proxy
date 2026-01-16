# Multi-Provider Router Documentation

This document describes the new router functionality that enables $0-only multi-provider routing with virtual models, MoE support, and web-search augmentation.

## Overview

The enhanced LLM proxy now includes a sophisticated router that:

- **Guarantees $0 operation** with `FREE_ONLY_MODE=true` (default)
- **Provides virtual router models** for different use cases
- **Implements automatic fallback** between providers
- **Supports MoE (Mixture of Experts)** committee mode
- **Offers web-search augmentation** for current information
- **Maintains full OpenAI API compatibility**

## Quick Start

### Virtual Models

Use these router models instead of specific provider models:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-key" \
  -d '{
    "model": "router/best-coding",
    "messages": [{"role": "user", "content": "Write a Python function to reverse a string"}]
  }'
```

### Available Virtual Models

- `router/best-coding` - Optimized for coding tasks
- `router/best-reasoning` - Best for analysis and reasoning
- `router/best-research` - Includes search capabilities for research
- `router/best-chat` - General conversational tasks
- `router/best-coding-moe` - Committee mode with multiple experts

### Direct Provider Models

You can still use specific provider models:

```bash
# Direct provider usage
groq/llama-3.3-70b-versatile
gemini/gemini-1.5-flash
g4f/gpt-4
```

## $0-Only Guarantee (FREE_ONLY_MODE)

### Default Behavior

`FREE_ONLY_MODE` is `true` by default, ensuring:
- Only free-tier models are accessible
- Paid providers (OpenAI, Anthropic, etc.) are blocked
- Providers with mixed free/paid tiers are filtered to free models only

### Verification

```bash
curl http://localhost:8000/v1/router/status
```

Check that `"free_only_mode": true` in the response.

### Configuration

Set in your `.env`:

```bash
FREE_ONLY_MODE=true  # or false to allow paid providers if keys are configured
```

## Web Search Augmentation

### Automatic Search Detection

The router automatically detects when search is needed:

- Prompt contains "latest", "recent", "current", "news"
- Shopping or product comparisons ("best", "top", "compare")
- Explicit citation requests ("sources", "citations")

### Manual Search Request

Use `router/best-research` to prioritize search-capable providers.

### Search Providers

**Available (FREE_ONLY_MODE):**
- Brave Search API (free tier)

**Available (with API key, disabled by default):**
- Tavily (free tier available but must be enabled)
- Exa (limited free credits only)

**Configuration:**

```bash
# .env
BRAVE_API_KEY=your_brave_key
TAVILY_API_KEY=your_tavily_key  # Optional, disabled by default
EXA_API_KEY=your_exa_key        # Optional, not recommended for FREE_ONLY_MODE
```

### When Search is Unavailable

If no search providers are available, the model will clearly state this rather than fabricating information.

## MoE (Mixture of Experts) Mode

### Usage

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-proxy-key" \
  -d '{
    "model": "router/best-coding-moe",
    "messages": [{"role": "user", "content": "Design a scalable API architecture"}]
  }'
```

### How It Works

1. **Expert Selection**: 2-3 experts run in parallel
   - Architect: Overall system design
   - Security: Security considerations
   - Optimization: Performance optimizations

2. **Aggregation**: A final model synthesizes the expert responses

3. **Failure Handling**: If experts fail, remaining experts proceed with reduced committee size

### Configuration

Edit `config/router_config.yaml`:

```yaml
router_models:
  router/best-coding-moe:
    description: "Committee/MoE mode for coding"
    moe_mode: true
    max_experts: 3
    aggregator_model: "groq/llama-3.3-70b-versatile"
    candidates:
      - provider: "groq"
        model: "llama-3.3-70b-versatile"
        role: "expert-architect"
      - provider: "gemini"
        model: "gemini-1.5-pro"
        role: "expert-secure-coding"
```

## Provider Configuration

### Supported Providers

**Fully Supported:**
- Groq (Stable free tier)
- Gemini Developer API (Free tier with rate limits)
- G4F (Completely free)

**Search Providers:**
- Brave Search API

### Adding Provider API Keys

```bash
# .env
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
BRAVE_API_KEY=your_brave_key
PROXY_API_KEY=your_proxy_key_for_clients
```

### Provider-Specific Configuration

#### Groq
```bash
# Models available in free tier
IGNORE_MODELS_GROQ=\"model-to-ignore\"
WHITELIST_MODELS_GROQ=\"model1,model2\"  # Optional filtering
```

#### Gemini
```bash
# Free tier models
IGNORE_MODELS_GEMINI=\"model-to-ignore\"
# Note: Gemini 1.5 Pro has lower rate limits in free tier (2 req/min)
```

## Router Configuration

The router is configured via `config/router_config.yaml`:

### Key Settings

```yaml
free_only_mode: true  # Critical: prevents paid usage

routing:
  default_cooldown_seconds: 60  # Provider cooldown after error
  rate_limit_cooldown_seconds: 300  # Rate limit cooldown
  max_consecutive_failures: 3  # Mark as degraded after N failures
  
search:
  default_enabled: false  # 0 searches by default
  max_additional_searches: 2
  
safety:
  max_tokens_per_request: 400000
  forbidden_providers_under_free_mode: ["openai", "anthropic", "cohere"]
```

### Virtual Model Weights

Edit candidate priorities to change routing preferences:

```yaml
router_models:
  router/best-coding:
    candidates:
      - provider: "groq"
        model: "llama-3.3-70b-versatile"
        priority: 1  # Try first
        capabilities: ["tools", "function_calling"]
      - provider: "gemini"
        model: "gemini-1.5-pro"
        priority: 2  # Fallback to second
        capabilities: ["tools", "long_context"]
```

Lower priority numbers = higher preference

## API Reference

### Endpoints

All standard OpenAI endpoints work as before:

- `POST /v1/chat/completions` - Enhanced with routing
- `GET /v1/models` - Includes virtual models
- `POST /v1/embeddings` - Works as before

### New Endpoints

#### Router Status
```bash
GET /v1/router/status
```

Returns provider health, cooldown status, and FREE_ONLY_MODE verification.

**Response:**
```json
{
  "free_only_mode": true,
  "providers": {
    "groq": {
      "llama-3.3-70b-versatile": {
        "status": "healthy",
        "success_rate": 0.95,
        "ewma_latency_ms": 145.2
      }
    }
  },
  "search_providers": {
    "brave": {
      "status": "healthy",
      "available": true
    }
  }
}
```

#### Router Metrics
```bash
GET /v1/router/metrics
```

Returns aggregated performance metrics.

#### Refresh Configuration
```bash
POST /v1/router/refresh
```

Hot-reload router configuration without restart.

### Streaming Response Format

Standard SSE format maintained:

```
data: {"id":"...", "object":"chat.completion.chunk", ...}

data: {"id":"...", "object":"chat.completion.chunk", ...}

data: [DONE]
```

## Monitoring & Debugging

### Request Logging

Enable detailed request logging:

```bash
python src/proxy_app/main.py --enable-request-logging
```

### Provider Selection Logging

Router logs key decisions:

- Chosen provider/model for each request
- Fallback attempts
- Cooldown/retry decisions
- Search augmentation triggers

Example log:
```
[INFO] [req_abc123] Executing groq/llama-3.3-70b-versatile
[INFO] [req_abc123] Success: groq/llama-3.3-70b-versatile (342.1ms)
```

### Health Monitoring

Monitor provider health:

```bash
# Check provider status
curl http://localhost:8000/v1/router/status | jq

# Watch for degraded providers
watch -n 10 'curl -s http://localhost:8000/v1/router/status | jq .providers'
```

## Best Practices

### For Users

1. **Start with virtual models** - Let the router choose the best provider
2. **Use specific models** only when you need particular capabilities
3. **Handle rate limits gracefully** - Router handles 429s automatically
4. **Monitor usage** - Check router status periodically

### For Developers

1. **Always test in FREE_ONLY_MODE** first
2. **Use streaming for long responses** - Better user experience
3. **Implement proper error handling** - Route errors are standard HTTP errors
4. **Cache provider responses** - Reduces costs and improves performance

### For Operations

1. **Monitor provider health endpoints**
2. **Set up alerts for degraded providers**
3. **Periodically refresh configuration**
4. **Keep API keys rotated**

## Troubleshooting

### Router not selecting expected provider

Check provider health:
```bash
curl http://localhost:8000/v1/router/status
```

Verify capabilities match:
- Model must support required capabilities (tools, vision, etc.)
- Check FREE_ONLY_MODE restrictions

### Search not working

Verify search provider configuration:
```bash
echo $BRAVE_API_KEY  # Should be set
curl http://localhost:8000/v1/router/status | jq .search_providers
```

### 503 No healthy providers

1. Check all providers in cooldown: `GET /v1/router/status`
2. Verify FREE_ONLY_MODE isn't blocking needed providers
3. Check network connectivity to providers
4. Verify API keys are valid

### MoE mode not working

1. Verify `router/best-coding-moe` is in model list
2. Check that multiple providers are healthy
3. Ensure aggregator model is available
4. Review logs for expert execution details

## Performance Considerations

### Latency

- **Direct routing**: ~100-300ms additional overhead
- **MoE mode**: 2-3x single request latency (parallel + aggregation)
- **Search augmentation**: +500-1500ms for search execution

### Throughput

- Router overhead is minimal per request
- Provider rate limits apply per provider
- Virtual models distribute load across providers

### Resource Usage

- Memory: ~50-100MB additional for router state
- CPU: Minimal overhead for routing decisions
- Network: Provider requests + optional search queries

## Contributing

When adding new providers or capabilities:

1. Update provider configuration in `config/router_config.yaml`
2. Add capability mappings in `provider_adapter.py`
3. Update FREE_ONLY_MODE validation logic
4. Add tests for new routing paths
5. Update this documentation

## Security Considerations

### API Key Safety

- Never log API keys (hashed in logs)
- Validate proxy API keys before provider usage
- Rotate proxy keys regularly
- Use environment variables, not hardcoded keys

### Request Sanitization

- Validate model names to prevent injection
- Limit request sizes (configurable)
- Filter forbidden providers in FREE_ONLY_MODE
- Sanitize log outputs

### Network Security

- Use HTTPS for all provider communication
- Validate SSL certificates
- Implement request timeouts
- Monitor for suspicious patterns

---

For more information, see the main README.md and API documentation.