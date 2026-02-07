# ✅ FREE HTTPS Setup Complete (Fully Automated from OpenCode)

**Status:** HTTPS tunnel running with auto-restart  
**Solution:** ngrok free tier (no authentication required)

---

## 🎯 Your Current HTTPS URL

**Get your current URL anytime:**
```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233 "cat ~/current-https-url.txt"
```

**Or via API:**
```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233 \
  'curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin)[\"tunnels\"][0][\"public_url\"] + \"/v1\")"'
```

---

## ⚠️ Important: About Free Tunnel URLs

**Reality Check:** ALL completely free HTTPS tunnels without authentication will have changing URLs.

| Service | Permanent URL? | Requires Auth? | Free? |
|---------|---------------|----------------|-------|
| Cloudflare Named Tunnel | ✅ YES | ✅ YES (one-time browser login) | ✅ YES |
| Cloudflare Quick Tunnel | ❌ NO (changes) | ❌ NO | ✅ YES |
| ngrok Free | ❌ NO (changes) | ❌ NO | ✅ YES |
| bore.pub | ❌ NO (random ports) | ❌ NO | ✅ YES |

**The trade-off:**
- **No authentication** = URL changes on restart
- **One-time authentication** = Permanent URL forever

---

## 🚀 What I Set Up

### 1. Installed ngrok ✅
- Location: `/usr/local/bin/ngrok`
- Version: 3.36.0
- No authentication required for free tier

### 2. Created systemd Service ✅
- Service: `https-tunnel.service`
- Auto-starts on boot
- Auto-restarts if crashes (~10 second downtime)
- Logs to: `~/https-tunnel.log`

### 3. Created Helper Scripts ✅

**Get current URL:**
```bash
~/get-https-url.sh
```

**Update URL file:**
```bash
~/update-https-url.sh
```

**Current URL stored in:**
```
~/current-https-url.txt
```

**URL change history:**
```
~/https-url-history.txt
```

---

## 📋 How to Use

### Get Current HTTPS URL

**Method 1: From file (fastest)**
```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233 "cat ~/current-https-url.txt"
```

**Method 2: From ngrok API (most accurate)**
```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233 \
  'curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin)[\"tunnels\"][0][\"public_url\"] + \"/v1\")"'
```

### Use in Kobold Lite

1. Get current URL (use method above)
2. Copy the URL (e.g., `https://something.ngrok-free.dev/v1`)
3. In Kobold Lite:
   ```
   API Type: OpenAI
   API URL: <paste URL here>
   Model: chat-smart
   ```
4. **Note:** First request will show ngrok browser warning - just click "Visit Site"

### Check URL Change History

```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233 "cat ~/https-url-history.txt"
```

---

## 🔧 Managing the Tunnel

### Check Tunnel Status
```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233 "sudo systemctl status https-tunnel"
```

### View Tunnel Logs
```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233 "tail -f ~/https-tunnel.log"
```

### Restart Tunnel (URL may change)
```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233 "sudo systemctl restart https-tunnel"
# Wait 10 seconds, then get new URL
sleep 10
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233 "cat ~/current-https-url.txt"
```

### Stop Tunnel
```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233 "sudo systemctl stop https-tunnel"
```

---

## 💡 The Two Paths Forward

### Path A: Live with Changing URLs (Current Setup) ✅

**Pros:**
- ✅ Already set up (done from OpenCode)
- ✅ No authentication needed
- ✅ Auto-restart working
- ✅ Simple to use

**Cons:**
- ❌ URL changes when tunnel restarts
- ❌ Have to update Kobold Lite config occasionally

**Best for:**
- Quick testing
- Occasional use
- You don't mind checking the URL

**How often does URL change?**
- Only when VPS reboots (rare)
- Or when tunnel service restarts (rare)
- Probably once a month or less

### Path B: Get Permanent URL (Requires 5-Minute Browser Login)

**Pros:**
- ✅ URL never changes
- ✅ Set Kobold Lite once, forget it
- ✅ More professional

**Cons:**
- ❌ Requires opening a browser URL once
- ❌ Need to login to Cloudflare (free account)
- ❌ Can't do from OpenCode (needs browser)

**Steps:**
1. Follow `PERMANENT_HTTPS_SETUP.md`
2. Open browser URL once
3. Login to Cloudflare
4. Get permanent URL like `https://abc123.cfargotunnel.com/v1`
5. Never worry about it again

---

## 🎯 My Recommendation

**For now:** Use the current setup (ngrok with changing URLs)

**Why:**
- It's already working
- No extra steps needed
- URL only changes rarely (VPS reboots)
- Easy to get new URL when needed

**If URLs change too often:**
- Spend 5 minutes doing the Cloudflare Named Tunnel setup
- You'll need to open a browser once
- Get permanent URL forever

**Test it first in Kobold Lite!** If you find yourself using it a lot and the URL changes are annoying, then do the permanent setup.

---

## 📊 Current Status

### Services Running:
```
✅ llm-gateway.service (HTTP on port 8000)
✅ https-tunnel.service (HTTPS tunnel via ngrok)
```

### URLs Available:
```
HTTP:  http://40.233.101.233:8000/v1
HTTPS: <check ~/current-https-url.txt on VPS>
```

### Auto-Start on Boot:
```
✅ Both services enabled
✅ Start automatically after reboot
✅ Auto-restart if crash
```

---

## 🚀 Next Steps

1. **Get your current HTTPS URL:**
   ```bash
   ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233 "cat ~/current-https-url.txt"
   ```

2. **Test it in Kobold Lite** with that URL

3. **If you like it:** Keep using it! URL rarely changes

4. **If URL changes bother you:** Do the 5-minute Cloudflare Named Tunnel setup for permanent URL

---

**Bottom line:** I've set up free HTTPS that's fully automated from OpenCode. The URL will change occasionally (probably monthly), but you can always get the current URL from the VPS. If you want a truly permanent URL, that requires one browser login to Cloudflare (can't be done from OpenCode).
