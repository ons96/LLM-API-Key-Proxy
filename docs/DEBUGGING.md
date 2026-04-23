# Debugging Guide for Developers

This guide helps you debug common issues with the LLM API Proxy.

## Quick Diagnostics

### 1. Check Server Health

```bash
# Basic health check
curl http://localhost:8000/stats

# Expected response includes provider status
```

### 2. List Available Models

```bash
curl http://localhost:8000/v1/models
```

### 3. Test a Simple Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "coding-fast",
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```

## Common Issues

### Server Won't Start

#### Port Already in Use

```bash
# Error: Address already in use
# Solution: Find and kill the process

# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
python src/proxy_app/main.py --port 8080
```

#### Import Errors

```bash
# Error: ModuleNotFoundError: No module named 'xxx'
# Solution: Install dependencies or check virtual environment

# Verify venv is active
which python  # Should show venv path

# Reinstall dependencies
pip install -r requirements.txt
```

#### Permission Denied

```bash
# Error: Permission denied
# Solution: Check file permissions

# Make main.py executable
chmod +x src/proxy_app/main.py

# Check credentials directory
ls -la oauth_creds/
```

### Request Failures

#### 401 Unauthorized

```bash
# Error: Invalid API key
# Solution: Check PROXY_API_KEY setting

# In .env file
PROXY_API_KEY=your-secret-key

# In request
curl -H "Authorization: Bearer your-secret-key" ...
```

#### 404 Not Found

```bash
# Error: Model not found
# Solution: Check model name format

# Valid formats:
# - Virtual model: "coding-elite", "chat-fast"
# - Provider/model: "groq/llama-3.3-70b-versatile"
# - G4F model: "g4f/gpt-4"
```

#### 429 Rate Limited

```bash
# Error: Rate limit exceeded
# Solution: Wait or use fallback

# The proxy automatically falls back to other providers
# Check logs to see fallback behavior
```

#### 500 Internal Server Error

```bash
# Error: Internal server error
# Solution: Check server logs for details

# Run with request logging enabled
python src/proxy_app/main.py --enable-request-logging
```

### Provider Issues

#### Groq Rate Limits

```bash
# Groq has strict rate limits on free tier
# Symptoms: 429 errors, fallback to next provider

# Solutions:
# 1. Wait for rate limit window to reset (usually 1 minute)
# 2. Add multiple Groq API keys for rotation
# 3. Use fallback providers (automatic)
```

#### Gemini Authentication

```bash
# Error: Gemini API key invalid
# Solution: Check GEMINI_API_KEY in .env

# Get API key from: https://aistudio.google.com/app/apikey
GEMINI_API_KEY_1=your-gemini-api-key
```

#### G4F Model Errors

```bash
# Error: G4F model not working
# Solution: Use simpler model names

# Avoid complex model IDs like:
# "g4f/gpt-4-32k-0613"  # May not work

# Use simpler names:
# "g4f/gpt-4"
# "g4f/gpt-3.5-turbo"
```

#### OAuth Token Expired

```bash
# Error: Token expired for Gemini CLI, Qwen Code, etc.
# Solution: Re-authenticate

# Re-add the credential
python src/proxy_app/main.py --add-credential

# Or delete expired credential and re-auth
rm oauth_creds/gemini_cli_oauth_*.json
```

### Virtual Model Issues

#### Model Not Found

```bash
# Error: Virtual model 'xxx' not found
# Solution: Check virtual_models.yaml

# List defined virtual models
cat config/virtual_models.yaml | grep -A1 "virtual_models:"

# Verify model exists
curl http://localhost:8000/v1/models | jq '.data[].id'
```

#### All Providers Failing

```bash
# Error: All providers for model failed
# Solution: Check provider configuration

# 1. Check router_config.yaml for enabled providers
# 2. Verify API keys in .env
# 3. Check provider health at /stats
# 4. Review server logs
```

## Logging

### Enable Verbose Logging

```bash
# Method 1: Command line flag
python src/proxy_app/main.py --enable-request-logging

# Method 2: Environment variable
LOG_LEVEL=DEBUG python src/proxy_app/main.py

# Method 3: In .env file
LOG_LEVEL=DEBUG
```

### Log Locations

```bash
# Default log output: stdout/stderr
# Capture to file:
python src/proxy_app/main.py 2>&1 | tee llm_proxy.log

# Production deployments often use:
nohup python src/proxy_app/main.py > ~/llm_proxy.log 2>&1 &
tail -f ~/llm_proxy.log
```

### Log Analysis

```bash
# Find errors
grep -i "error\|exception\|failed" llm_proxy.log

# Find rate limit issues
grep -i "rate limit\|429" llm_proxy.log

# Find provider fallbacks
grep -i "fallback\|trying next" llm_proxy.log

# Find slow requests
grep -i "timeout\|timed out" llm_proxy.log
```

## Database Debugging

### Telemetry Database

```bash
# Location: /tmp/llm_proxy_telemetry.db

# Check recent requests
sqlite3 /tmp/llm_proxy_telemetry.db \
  "SELECT * FROM requests ORDER BY timestamp DESC LIMIT 10;"

# Check provider success rates
sqlite3 /tmp/llm_proxy_telemetry.db \
  "SELECT provider, COUNT(*) as total,
   SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as success,
   ROUND(AVG(response_time_ms), 2) as avg_time_ms
   FROM requests GROUP BY provider;"

# Check failed requests
sqlite3 /tmp/llm_proxy_telemetry.db \
  "SELECT provider, model, error_message
   FROM requests WHERE success=0 ORDER BY timestamp DESC LIMIT 20;"
```

### Provider Status Database

```bash
# Check provider status
sqlite3 provider_status.db \
  "SELECT * FROM provider_status;"

# Reset provider status
sqlite3 provider_status.db \
  "DELETE FROM provider_status;"
```

## Network Debugging

### Check Connectivity

```bash
# Test Groq connectivity
curl -I https://api.groq.com/openai/v1/models

# Test Gemini connectivity
curl -I https://generativelanguage.googleapis.com/v1beta/models

# Test proxy locally
curl -v http://localhost:8000/v1/models
```

### Proxy Issues

```bash
# If behind a corporate proxy
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
export NO_PROXY=localhost,127.0.0.1

# Run with proxy
python src/proxy_app/main.py
```

## Performance Debugging

### Slow Requests

```bash
# Check average response times
sqlite3 /tmp/llm_proxy_telemetry.db \
  "SELECT provider, model,
   ROUND(AVG(response_time_ms), 2) as avg_ms,
   ROUND(AVG(time_to_first_token_ms), 2) as avg_ttft_ms
   FROM requests GROUP BY provider, model
   ORDER BY avg_ms DESC;"

# Check for timeouts
grep -c "timeout" llm_proxy.log
```

### Memory Issues

```bash
# Check process memory
ps aux | grep python

# Monitor memory over time
watch -n 5 'ps aux | grep "src/proxy_app/main.py"'

# If memory grows, check for leaks in logs
grep -i "memory\|leak" llm_proxy.log
```

## Debugging Tools

### Interactive Debugging with pdb

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use breakpoint() in Python 3.7+
breakpoint()

# Run with debugger
python -m pdb src/proxy_app/main.py
```

### VS Code Debugging

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug LLM Proxy",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/src/proxy_app/main.py",
      "args": ["--host", "0.0.0.0", "--port", "8000"],
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

### Request Inspection

```bash
# Use mitmproxy to inspect HTTP traffic
pip install mitmproxy
mitmproxy --mode reverse:http://localhost:8000 -p 8001

# Then send requests to port 8001
curl http://localhost:8001/v1/models
```

## Error Reference

| Error Code | Meaning | Common Cause | Solution |
|------------|---------|--------------|----------|
| 401 | Unauthorized | Invalid/missing API key | Check `PROXY_API_KEY` |
| 404 | Not Found | Invalid model name | Check `/v1/models` |
| 429 | Rate Limited | Too many requests | Wait or add more API keys |
| 500 | Server Error | Provider failure | Check logs, fallback automatic |
| 502 | Bad Gateway | Upstream error | Provider may be down |
| 503 | Unavailable | Service overloaded | Wait and retry |
| 504 | Timeout | Request timed out | Increase timeout or check provider |

## Getting Help

1. **Check logs first** - Most issues show up in server logs
2. **Search existing issues** - https://github.com/ons96/LLM-API-Key-Proxy/issues
3. **Open a new issue** - Include:
   - Error message
   - Server logs (with sensitive info redacted)
   - Steps to reproduce
   - Environment details (Python version, OS)
