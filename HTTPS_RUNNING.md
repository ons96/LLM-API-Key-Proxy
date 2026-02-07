# ✅ HTTPS Setup Complete!

**Date:** 2026-02-07  
**Status:** HTTPS tunnel running successfully

---

## 🎉 Your HTTPS URL

**HTTPS Gateway URL:** `https://house-compressed-hist-extensions.trycloudflare.com/v1`

This URL now provides **secure HTTPS access** to your LLM gateway, making it compatible with:
- ✅ Kobold Lite
- ✅ SillyTavern
- ✅ Any browser-based tool requiring HTTPS

---

## ✅ Verification Tests

### Model List Test:
```bash
curl https://house-compressed-hist-extensions.trycloudflare.com/v1/models
✅ Returns 100+ models successfully
```

### Chat Completion Test:
```bash
curl -X POST https://house-compressed-hist-extensions.trycloudflare.com/v1/chat/completions \
  -d '{"model": "groq/llama-3.3-70b-versatile", ...}'

✅ Response: "OK"
```

**Both HTTP and HTTPS work!**

---

## 📋 URLs Summary

### HTTP (works for most tools):
- **URL:** `http://40.233.101.233:8000/v1`
- **Works with:** OpenCode, Python scripts, curl, most CLI tools
- **Direct connection:** No tunnel, fastest speed

### HTTPS (required for browser tools):
- **URL:** `https://house-compressed-hist-extensions.trycloudflare.com/v1`
- **Works with:** Kobold Lite, SillyTavern, browser-based tools
- **Cloudflare Tunnel:** Secure, encrypted, browser-compatible

---

## 🎮 Using with Kobold Lite

**Configuration:**
```
API Type: OpenAI
API URL: https://house-compressed-hist-extensions.trycloudflare.com/v1
Model: chat-smart (or any model)
API Key: (leave empty)
```

**Test it now:**
1. Open Kobold Lite in your browser
2. Enter the HTTPS URL above
3. Select model: `chat-smart` or `coding-elite`
4. Start chatting!

---

## 🔧 Managing the Tunnel

### Check Tunnel Status:
```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233 "ps aux | grep cloudflared"
```

### View Tunnel Logs:
```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233 "cat ~/cloudflare-tunnel.log"
```

### Get Current HTTPS URL:
```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233\
  "grep -o 'https://[^[:space:]]*trycloudflare.com' ~/cloudflare-tunnel.log | tail -1"
```

### Restart Tunnel (URL will change):
```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233\
  "pkill cloudflared && nohup cloudflared tunnel --url http://localhost:8000 > ~/cloudflare-tunnel.log 2>&1 &"
  
# Wait 10 seconds, then get new URL
sleep 10
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233\
  "grep -o 'https://[^[:space:]]*trycloudflare.com' ~/cloudflare-tunnel.log | tail -1"
```

---

## ⚠️ Important Notes

### URL Changes on Restart
**Current URL:** `https://house-compressed-hist-extensions.trycloudflare.com/v1`

This is a **Quick Tunnel** URL that changes every time the tunnel restarts. This is fine for personal use.

**If you need a permanent URL:**
1. Follow the instructions in `HTTPS_SETUP_GUIDE.md`
2. Create a named Cloudflare Tunnel
3. You'll get a permanent URL like `https://your-id.cfargotunnel.com`

### Make Tunnel Permanent (Auto-start)

To make the tunnel start automatically on boot:

```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233

# Create systemd service
sudo tee /etc/systemd/system/cloudflare-tunnel.service > /dev/null << 'EOF'
[Unit]
Description=Cloudflare Tunnel
After=network.target llm-gateway.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
ExecStart=/usr/local/bin/cloudflared tunnel --url http://localhost:8000
Restart=always
RestartSec=10
StandardOutput=append:/home/ubuntu/cloudflare-tunnel.log
StandardError=append:/home/ubuntu/cloudflare-tunnel.log

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable cloudflare-tunnel
sudo systemctl restart cloudflare-tunnel

# Get the new URL (wait 10 seconds first)
sleep 10 && grep -o "https://[^[:space:]]*trycloudflare.com" ~/cloudflare-tunnel.log | tail -1
```

**Note:** Even with systemd, the URL still changes on restart. For a permanent URL, use a named tunnel (see guide).

---

## 🔍 Web Search Priority Confirmed

Your web search fallback order is **already configured correctly**:

```yaml
1. Tavily (priority: 1) ✅ First choice
2. Brave (priority: 2)  ✅ Second choice
3. DuckDuckGo (priority: 3) ✅ Fallback
```

**Configuration file:** `config/router_config.yaml` lines 285-303

The gateway will automatically:
- Try Tavily first
- Fall back to Brave if Tavily fails
- Use DuckDuckGo as last resort
- Smart enough to use DuckDuckGo directly for simple queries

No changes needed - it's already set up the way you requested!

---

## 📊 Complete Access Summary

### Your LLM Gateway is accessible via:

| URL | Protocol | Use For | Speed | Browser Required |
|-----|----------|---------|-------|------------------|
| `http://40.233.101.233:8000/v1` | HTTP | OpenCode, Python scripts, CLI tools | ⚡ Fastest (direct) | ❌ No |
| `https://house-compressed-hist-extensions.trycloudflare.com/v1` | HTTPS | Kobold Lite, SillyTavern, browser tools | ✅ Fast (via Cloudflare) | ✅ Yes |

**Both URLs work simultaneously!** Use whichever is appropriate for your tool.

---

## 🎯 Next Steps

1. ✅ **HTTP Gateway** - Running at `http://40.233.101.233:8000/v1`
2. ✅ **HTTPS Tunnel** - Running at `https://house-compressed-hist-extensions.trycloudflare.com/v1`
3. ✅ **Web Search Priority** - Tavily → Brave → DuckDuckGo (already configured)
4. ✅ **OpenCode** - Configured with HTTP URL
5. ✅ **Ready for Kobold Lite** - Use HTTPS URL

**Everything is ready to use!** 🚀

---

## 📝 Quick Reference

### For OpenCode:
```
baseURL: http://40.233.101.233:8000/v1
model: coding-elite
```

### For Kobold Lite:
```
API URL: https://house-compressed-hist-extensions.trycloudflare.com/v1
Model: chat-smart
```

### For SillyTavern:
```
API: OpenAI
Reverse Proxy: https://house-compressed-hist-extensions.trycloudflare.com/v1
Model: chat-smart
```

**Enjoy your free, unlimited LLM access with both HTTP and HTTPS!** 🎉
