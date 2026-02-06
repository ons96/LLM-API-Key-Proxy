# OpenCode + Custom LLM Gateway Integration Guide

## ⚠️ Important: Configuration Reverted

I had to revert the configuration changes because OpenCode's config schema doesn't support custom `baseUrl`/`apiKey` in the provider section.

**OpenCode should work now** - try running `opencode` again.

---

## Alternative Integration Methods

Since OpenCode doesn't directly support custom OpenAI-compatible endpoints in the config file, here are working alternatives:

### Option 1: Use OpenCode's Built-in OpenAI Provider with Environment Variables

Some versions of OpenCode support overriding the OpenAI endpoint via environment variables:

```bash
# Set environment variables before running opencode
export OPENAI_BASE_URL="http://40.233.101.233:8000/v1"
export OPENAI_API_KEY="CHANGE_ME_TO_A_STRONG_SECRET_KEY"

# Then run opencode
opencode
```

**Note:** This might not work depending on your OpenCode version.

---

### Option 2: Create an MCP (Model Context Protocol) Server

Create a custom MCP server that proxies requests to your VPS gateway:

**File: `~/mcp-vps-gateway/server.js`**

```javascript
#!/usr/bin/env node
const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const axios = require('axios');

const server = new Server({
  name: 'vps-gateway-mcp',
  version: '1.0.0',
}, {
  capabilities: {
    tools: {},
  },
});

// Proxy chat completions to your VPS
server.setRequestHandler('tools/call', async (request) => {
  if (request.params.name === 'chat_completion') {
    const response = await axios.post(
      'http://40.233.101.233:8000/v1/chat/completions',
      request.params.arguments,
      {
        headers: {
          'Authorization': 'Bearer CHANGE_ME_TO_A_STRONG_SECRET_KEY',
          'Content-Type': 'application/json',
        },
      }
    );
    return {
      content: [
        {
          type: 'text',
          text: response.data.choices[0].message.content,
        },
      ],
    };
  }
  throw new Error(`Unknown tool: ${request.params.name}`);
});

const transport = new StdioServerTransport();
server.connect(transport);
```

Then add to `~/.config/opencode/oh-my-opencode.json`:

```json
"mcps": {
  "vps-gateway": {
    "command": "node",
    "args": ["~/mcp-vps-gateway/server.js"],
    "description": "VPS LLM Gateway MCP Server"
  }
}
```

---

### Option 3: Use Litellm Proxy (Recommended)

Set up a local litellm proxy that forwards to your VPS:

**Install litellm:**
```bash
pip install litellm
```

**Create config: `~/litellm-config.yaml`**

```yaml
model_list:
  - model_name: coding-elite
    litellm_params:
      model: openai/coding-elite
      api_base: http://40.233.101.233:8000/v1
      api_key: CHANGE_ME_TO_A_STRONG_SECRET_KEY
  - model_name: coding-fast
    litellm_params:
      model: openai/coding-fast
      api_base: http://40.233.101.233:8000/v1
      api_key: CHANGE_ME_TO_A_STRONG_SECRET_KEY
```

**Start proxy:**
```bash
litellm --config ~/litellm-config.yaml --port 9000
```

**Configure OpenCode** to use local proxy (if supported) or set:
```bash
export OPENAI_BASE_URL="http://localhost:9000"
export OPENAI_API_KEY="dummy-key"
```

---

### Option 4: Use a CLI Tool Wrapper

Create a wrapper script that calls your VPS gateway:

**File: `~/bin/opencode-vps`**

```bash
#!/bin/bash
# Wrapper that uses VPS gateway for all OpenAI calls

# Set your VPS as the OpenAI endpoint
export OPENAI_BASE_URL="http://40.233.101.233:8000/v1"
export OPENAI_API_KEY="CHANGE_ME_TO_A_STRONG_SECRET_KEY"

# Run opencode with these settings
opencode "$@"
```

Make it executable:
```bash
chmod +x ~/bin/opencode-vps
```

Then use `opencode-vps` instead of `opencode`.

---

### Option 5: Direct API Usage (No OpenCode Integration)

Use curl or a script to call your VPS gateway directly:

```bash
# Test coding-elite
curl -s http://40.233.101.233:8000/v1/chat/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer CHANGE_ME_TO_A_STRONG_SECRET_KEY" \
  -d '{
    "model": "coding-elite",
    "messages": [{"role": "user", "content": "Your prompt here"}],
    "max_tokens": 2000
  }'
```

---

## Current Status

✅ **OpenCode config reverted** - should work now  
❌ **Direct custom provider integration** - not supported by OpenCode's schema  
✅ **VPS Gateway working** - accessible via curl  
⚠️ **Integration required** - use one of the options above

---

## Recommended Approach

**For now:** Use Option 5 (Direct API) or Option 3 (Litellm Proxy)

**Best long-term:** Wait for OpenCode to add support for custom OpenAI-compatible endpoints, or use Option 2 (MCP server) if you're comfortable with Node.js.

---

## Testing Your Gateway

Always verify your gateway works first:

```bash
# Test from WSL
curl -s http://40.233.101.233:8000/v1/models \
  -H "Authorization: Bearer CHANGE_ME_TO_A_STRONG_SECRET_KEY" \
  | jq '.data | length'

# Should return: 2218
```

If this works, your gateway is ready - you just need to integrate it with OpenCode using one of the methods above.
