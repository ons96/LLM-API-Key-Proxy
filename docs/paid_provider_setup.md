# Paid Provider Setup Guide

This document describes how to configure paid LLM API providers for use with the proxy rotator system.

## Overview

Paid providers offer higher reliability, better rate limits, and access to premium models compared to free-tier providers. The system supports any OpenAI-compatible API endpoint through the generic `OpenAICompatibleProvider` class.

## Supported Paid Providers

The following paid providers are supported out of the box:

- **OpenAI** (`openai`) - GPT-4, GPT-3.5-turbo, and other OpenAI models
- **Anthropic** (`anthropic`) - Claude 3.5 Sonnet, Claude 3 Opus, etc.
- **Azure OpenAI** (`azure`) - Enterprise OpenAI deployment
- **Groq** (`groq`) - High-performance inference for open source models
- **Mistral AI** (`mistral`) - Mistral Large, Medium, and Small models
- **Cohere** (`cohere`) - Command R and Embed models
- **Together AI** (`together`) - Fast inference for open source models
- **OpenRouter** (`openrouter`) - Unified API for multiple providers
- **NVIDIA** (`nvidia`) - NVIDIA NIM inference microservices

## Configuration

### Environment Variables

Each paid provider requires specific environment variables to be set:

#### OpenAI
```bash
export OPENAI_API_KEY="sk-..."
# Optional: Custom base URL (defaults to https://api.openai.com/v1)
export OPENAI_API_BASE="https://api.openai.com/v1"
```

#### Anthropic
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# Optional: Custom base URL (defaults to https://api.anthropic.com/v1)
export ANTHROPIC_API_BASE="https://api.anthropic.com/v1"
```

#### Groq
```bash
export GROQ_API_KEY="gsk_..."
# Optional: Custom base URL (defaults to https://api.groq.com/openai/v1)
export GROQ_API_BASE="https://api.groq.com/openai/v1"
```

#### Mistral AI
```bash
export MISTRAL_API_KEY="..."
# Optional: Custom base URL (defaults to https://api.mistral.ai/v1)
export MISTRAL_API_BASE="https://api.mistral.ai/v1"
```

#### Cohere
```bash
export COHERE_API_KEY="..."
# Optional: Custom base URL (defaults to https://api.cohere.ai/v1)
export COHERE_API_BASE="https://api.cohere.ai/v1"
```

#### Together AI
```bash
export TOGETHER_API_KEY="..."
# Optional: Custom base URL (defaults to https://api.together.xyz/v1)
export TOGETHER_API_BASE="https://api.together.xyz/v1"
```

#### OpenRouter
```bash
export OPENROUTER_API_KEY="sk-or-..."
# Optional: Custom base URL (defaults to https://openrouter.ai/api/v1)
export OPENROUTER_API_BASE="https://openrouter.ai/api/v1"
```

#### NVIDIA
```bash
export NVIDIA_API_KEY="nvapi-..."
# Optional: Custom base URL (defaults to https://integrate.api.nvidia.com/v1)
export NVIDIA_API_BASE="https://integrate.api.nvidia.com/v1"
```

#### Azure OpenAI
```bash
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://your-resource.openai.azure.com/openai/deployments/your-deployment"
# Note: Azure requires the full deployment URL including deployment name
```

## Usage

Once configured, paid providers are automatically discovered and prioritized by the system. They receive a default tier priority of `1` (highest priority) compared to free providers.

### Provider Selection

The router automatically prefers paid providers when:
1. The provider is healthy (passing health checks)
2. Rate limits have not been exceeded
3. The requested model is available on the provider

### Health Monitoring

Paid providers are monitored by the `ProviderStatusTracker` for:
- Response latency
- Error rates
- Rate limit exhaustion
- Uptime percentage

### Rate Limiting

Paid providers typically offer higher rate limits. Configure custom rate limits in `config/router_config.yaml`:

```yaml
rate_limits:
  openai:
    requests_per_minute: 500
    tokens_per_minute: 150000
  anthropic:
    requests_per_minute: 400
    tokens_per_minute: 100000
```

## Adding Custom Paid Providers

To add a new OpenAI-compatible paid provider:

1. Set the environment variables following the pattern:
   ```bash
   export PROVIDERNAME_API_KEY="..."
   export PROVIDERNAME_API_BASE="https://api.provider.com/v1"
   ```

2. Update `PAID_PROVIDER_MAP` in `src/rotator_library/provider_factory.py`:
   ```python
   PAID_PROVIDER_MAP = {
       # ... existing providers
       "providername": OpenAICompatibleProvider,
   }
   ```

3. Add default base URL logic in `get_provider_config()` if needed.

## Troubleshooting

### Provider Not Discovered
- Verify the API key environment variable is set correctly
- Check that the provider name matches exactly (case-insensitive)
- Review logs for configuration loading messages

### Authentication Errors
- Ensure the API key is valid and has not expired
- Verify the API base URL is correct (especially for Azure deployments)
- Check that the API key has permissions for the requested models

### Rate Limit Errors
- The system automatically handles rate limits by falling back to other providers
- Consider increasing retry delays in `config/router_config.yaml`
- Monitor `provider_status.db` for rate limit status updates

## Security Best Practices

1. **Never commit API keys** - Always use environment variables or secure vaults
2. **Use restricted keys** - Create API keys with minimal required permissions
3. **Monitor usage** - Regularly check the request logs for unexpected usage patterns
4. **Rotate keys** - Periodically rotate API keys and update environment variables
