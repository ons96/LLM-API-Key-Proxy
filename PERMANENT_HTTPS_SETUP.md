# 🔒 Setting Up Permanent HTTPS URL (Free Forever)

**Current Problem:** Quick Tunnel URL changes on every restart  
**Solution:** Cloudflare Named Tunnel - Permanent URL that NEVER changes

---

## ✅ Benefits of Named Tunnel

- 🆓 **100% free forever** (no credit card, no renewal)
- 🔗 **Same URL forever** (like `https://abc123.cfargotunnel.com`)
- ⚡ **Auto-restart** (systemd restarts in ~5 seconds if it crashes)
- 🌍 **Works from anywhere** (Cloudflare's global CDN)
- 🔐 **Secure** (automatic HTTPS, encrypted tunnel)

---

## 📋 Step-by-Step Setup (15 minutes)

### Step 1: SSH into Your VPS

```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233
```

### Step 2: Stop the Current Quick Tunnel

```bash
# Stop quick tunnel
pkill cloudflared

# Verify it's stopped
ps aux | grep cloudflared
```

### Step 3: Login to Cloudflare (Browser Required - ONE TIME)

```bash
cloudflared tunnel login
```

**This will display a URL like:**
```
Please open the following URL and log in with your Cloudflare account:

https://dash.cloudflare.com/argotunnel?aud=&callback=https%3A%2F%2Flogin...

Leave cloudflared running to download the cert automatically.
```

**What to do:**
1. **Copy that entire URL** from the SSH terminal
2. **Open it in your browser** (on your laptop)
3. **Log in to Cloudflare** (or create free account if you don't have one)
4. **Click "Authorize"**
5. You'll see "Success! You may now close this window"
6. **Back in SSH:** Press Ctrl+C to exit (cert is saved)

**Result:** Certificate saved to `~/.cloudflared/cert.pem`

### Step 4: Create Your Named Tunnel

```bash
# Create tunnel with name "llm-gateway"
cloudflared tunnel create llm-gateway
```

**Output will show:**
```
Tunnel credentials written to /home/ubuntu/.cloudflared/XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX.json
Created tunnel llm-gateway with id XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
```

**IMPORTANT:** Copy that ID! (the long XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX)

### Step 5: Create Configuration File

```bash
# Create config directory if it doesn't exist
mkdir -p ~/.cloudflared

# Create config file
nano ~/.cloudflared/config.yml
```

**Paste this content** (replace `TUNNEL-ID` with the ID from Step 4):

```yaml
tunnel: TUNNEL-ID
credentials-file: /home/ubuntu/.cloudflared/TUNNEL-ID.json

ingress:
  - service: http://localhost:8000
```

**Example with actual ID:**
```yaml
tunnel: 8e343b13-a087-48ea-825e-9c9d58f32618
credentials-file: /home/ubuntu/.cloudflared/8e343b13-a087-48ea-825e-9c9d58f32618.json

ingress:
  - service: http://localhost:8000
```

**Save:** Ctrl+X, then Y, then Enter

### Step 6: Get Your Permanent HTTPS URL

Your permanent URL is based on your tunnel ID:

```
https://TUNNEL-ID.cfargotunnel.com
```

**Example:**
```
https://8e343b13-a087-48ea-825e-9c9d58f32618.cfargotunnel.com
```

**This URL NEVER changes!** ✅

### Step 7: Test the Tunnel Manually

```bash
# Run tunnel manually to test
cloudflared tunnel run llm-gateway
```

**You should see:**
```
Registered tunnel connection
```

**Test it from your laptop:**
```bash
# Replace with YOUR tunnel ID
curl https://YOUR-TUNNEL-ID.cfargotunnel.com/v1/models
```

If it works, press **Ctrl+C** to stop the manual tunnel.

### Step 8: Set Up Systemd Service (Auto-Start)

```bash
# Install as system service
sudo cloudflared service install
```

**Create systemd override to use your config:**

```bash
sudo mkdir -p /etc/systemd/system/cloudflared.service.d
sudo nano /etc/systemd/system/cloudflared.service.d/override.conf
```

**Paste:**
```ini
[Service]
ExecStart=
ExecStart=/usr/local/bin/cloudflared tunnel --config /home/ubuntu/.cloudflared/config.yml run llm-gateway
```

**Save:** Ctrl+X, Y, Enter

### Step 9: Start and Enable the Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Start the tunnel
sudo systemctl start cloudflared

# Enable auto-start on boot
sudo systemctl enable cloudflared

# Check status
sudo systemctl status cloudflared
```

**You should see:**
```
● cloudflared.service - cloudflared
     Active: active (running)
```

### Step 10: Verify Everything Works

**Check tunnel status:**
```bash
cloudflared tunnel info llm-gateway
```

**Test from your laptop:**
```bash
# Replace with YOUR tunnel ID
curl https://YOUR-TUNNEL-ID.cfargotunnel.com/v1/models | head -20
```

**Test chat completion:**
```bash
curl -X POST https://YOUR-TUNNEL-ID.cfargotunnel.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "groq/llama-3.3-70b-versatile",
    "messages": [{"role": "user", "content": "Say OK"}],
    "max_tokens": 3
  }'
```

---

## ✅ Done! Your Permanent URL

**Your HTTPS URL (never changes):**
```
https://YOUR-TUNNEL-ID.cfargotunnel.com/v1
```

**Use this in:**
- ✅ Kobold Lite
- ✅ SillyTavern  
- ✅ Any browser tool
- ✅ OpenCode (if you prefer HTTPS)

---

## 🔧 Managing the Named Tunnel

### Check Tunnel Status
```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233 "sudo systemctl status cloudflared"
```

### View Tunnel Logs
```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233 "sudo journalctl -u cloudflared -f"
```

### Restart Tunnel (URL stays the same!)
```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233 "sudo systemctl restart cloudflared"
```

**Downtime:** ~5 seconds (systemd auto-restarts)  
**URL:** NEVER changes! ✅

### List All Your Tunnels
```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233 "cloudflared tunnel list"
```

### Delete a Tunnel (if needed)
```bash
# Only if you want to start over
cloudflared tunnel delete llm-gateway
```

---

## 🆚 Quick Tunnel vs Named Tunnel

| Feature | Quick Tunnel | Named Tunnel (Permanent) |
|---------|--------------|--------------------------|
| **URL** | Random (changes) | Fixed (never changes) |
| **Example URL** | `https://house-compressed-hist-extensions.trycloudflare.com` | `https://8e343b13-a087-48ea-825e-9c9d58f32618.cfargotunnel.com` |
| **Setup** | 1 command | 15 min setup (one time) |
| **Free** | ✅ Yes | ✅ Yes |
| **Login Required** | ❌ No | ✅ Yes (one time) |
| **Auto-Restart** | ✅ Yes | ✅ Yes |
| **URL After Restart** | ❌ Changes | ✅ Same forever |
| **Best For** | Testing | Production |

---

## 🎯 Recommendation

**Use Named Tunnel!** The 15-minute setup is worth it for a **permanent URL that never changes**.

**Quick Tunnel is fine for:**
- Quick testing
- You don't mind updating Kobold Lite config occasionally

**Named Tunnel is better for:**
- Long-term use
- Setting up once and forgetting about it
- Professional/reliable access

---

## 🐛 Troubleshooting

### "No file cert.pem found"
```bash
# Run login again
cloudflared tunnel login
# Open the URL in browser and authorize
```

### "Tunnel not found"
```bash
# List tunnels
cloudflared tunnel list

# If none exist, create one
cloudflared tunnel create llm-gateway
```

### "Connection refused"
```bash
# Make sure LLM gateway is running
sudo systemctl status llm-gateway

# Check if it's listening on port 8000
sudo ss -tulpn | grep :8000
```

### Tunnel Starts but URL Not Working
```bash
# Check tunnel logs
sudo journalctl -u cloudflared -n 50

# Verify config file
cat ~/.cloudflared/config.yml

# Make sure tunnel ID in config matches credentials file name
ls -la ~/.cloudflared/
```

---

## 📋 Quick Setup Checklist

```
□ SSH into VPS
□ Stop quick tunnel (pkill cloudflared)
□ Run: cloudflared tunnel login
□ Open URL in browser and authorize
□ Run: cloudflared tunnel create llm-gateway
□ Copy tunnel ID
□ Create ~/.cloudflared/config.yml with tunnel ID
□ Test manually: cloudflared tunnel run llm-gateway
□ Install service: sudo cloudflared service install
□ Create systemd override
□ Start: sudo systemctl start cloudflared
□ Enable: sudo systemctl enable cloudflared
□ Test URL: https://TUNNEL-ID.cfargotunnel.com/v1/models
□ Update Kobold Lite with permanent URL
□ Done! ✅
```

---

## 🎉 What You Get

After setup:
- ✅ **Permanent HTTPS URL** (like `https://abc123.cfargotunnel.com/v1`)
- ✅ **Never changes** (even after reboots, restarts, crashes)
- ✅ **Auto-starts on boot** (systemd)
- ✅ **Auto-restarts if crash** (~5 second downtime)
- ✅ **Free forever** (no renewal, no credit card)
- ✅ **Works globally** (Cloudflare CDN)

**Set it up once, use it forever!** 🚀
