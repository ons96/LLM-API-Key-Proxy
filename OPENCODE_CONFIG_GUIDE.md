# OpenCode Configuration for VPS LLM Gateway

## ✅ What's Been Configured

Your OpenCode is now configured to use your **VPS LLM Gateway** with all 5 virtual models!

### Provider Added: `vps-gateway`

**Location:** `~/.config/opencode/opencode.json`

**Settings:**
- **Base URL:** `http://<VPS40_IP>:8000/v1`
- **API Key:** `CHANGE_ME_TO_A_STRONG_SECRET_KEY`

### Available Models:

| Model | Use Case | Status |
|-------|----------|--------|
| `coding-elite` | Best agentic coding | ✅ Available |
| `coding-fast` | Ultra fast coding | ✅ Available |
| `chat-smart` | High intelligence chat | ✅ Available |
| `chat-fast` | Low latency chat | ✅ Available |
| `chat-rp` | Roleplay mode | ✅ Available |

### Default Model

**Current default:** `vps-gateway/coding-elite`

You can change this in `~/.config/opencode/opencode.json`:
```json
"model": "vps-gateway/coding-fast"
```

### Agent Models Updated

The following agents now use your VPS gateway:

- **Sisyphus** (main build agent): `vps-gateway/coding-elite`
- **Prometheus** (planning): `vps-gateway/coding-fast`
- **Atlas** (task execution): `vps-gateway/coding-fast`
- **Oracle** (consulting): `vps-gateway/coding-elite`

## 🚀 How to Use

### Option 1: Set as Default Model

In `~/.config/opencode/opencode.json`, change:
```json
"model": "vps-gateway/coding-elite"
```

### Option 2: Use Per-Agent

Keep default as-is, but specific agents use VPS models (already configured).

### Option 3: Change Model Mid-Conversation

When working with OpenCode, you can request specific models:
- "Use coding-elite for this complex refactoring"
- "Switch to coding-fast for quick edits"

## 🔧 Testing Your Configuration

Run this test to verify everything works:

```bash
# Test models endpoint
curl -s http://<VPS40_IP>:8000/v1/models \
  -H "Authorization: Bearer CHANGE_ME_TO_A_STRONG_SECRET_KEY" \
  | jq '.data | length'

# Should return: 2218
```

## 📝 Configuration Files Modified

1. **`~/.config/opencode/opencode.json`**
   - Added `vps-gateway` provider
   - Added 5 virtual models
   - Set default model to `vps-gateway/coding-elite`

2. **`~/.config/opencode/oh-my-opencode.json`**
   - Updated agent models to use VPS gateway
   - Sisyphus, Prometheus, Atlas, Oracle now use coding-elite/coding-fast

## ⚠️ Important Notes

1. **Security:** The API key is currently `CHANGE_ME_TO_A_STRONG_SECRET_KEY`. 
   - Consider changing this to a stronger key on your VPS
   - If you change it on VPS, update it in `opencode.json` too

2. **Fallback:** If VPS is offline, OpenCode will fall back to free models (kimi-k2.5-free, glm-4.7-free, etc.)

3. **Port:** Gateway running on port 8000
   - If you restart gateway on different port, update `baseUrl` in config

## 🔄 To Revert to Default Models

If you want to go back to OpenCode's free models:

1. Edit `~/.config/opencode/opencode.json`:
   ```json
   "model": "opencode/glm-4.7-free"
   ```

2. Edit `~/.config/opencode/oh-my-opencode.json` and change agent models back to `opencode/kimi-k2.5-free`

## 📊 Bandwidth & Latency

**Your VPS Location:** Oracle Cloud (likely US region)  
**Latency from WSL:** ~50-100ms (depending on your location)  
**Models Available:** 2218 (includes virtual + direct providers)

---

**Your custom LLM gateway is now integrated with OpenCode! 🎉**
