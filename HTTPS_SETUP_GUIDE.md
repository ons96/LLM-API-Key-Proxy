# Setting Up HTTPS for Kobold Lite Access

## Problem

**Kobold Lite requires HTTPS** ❌ Your current HTTP URL won't work due to browser mixed-content security policies.

- Current URL: `http://40.233.101.233:8000/v1` ❌
- Needed: `https://your-domain.trycloudflare.com/v1` ✅

---

## Solution: Cloudflare Tunnel (100% Free Forever)

**Cloudflare Tunnel** is the best option because:
- ✅ **Completely free forever** (no credit card, no renewal)
- ✅ **No domain required** (uses free `.trycloudflare.com` subdomain)
- ✅ **Permanent** (tunnel stays up as long as service runs)
- ✅ **Fast** (Cloudflare's global CDN)
- ✅ **Already installed** on your VPS!

---

## Setup Instructions

### Step 1: SSH into Your VPS

```bash
ssh -i ~/.ssh/oracle.key ubuntu@40.233.101.233
```

### Step 2: Create a Quick Tunnel (No Login Required)

The easiest method - no Cloudflare account needed:

```bash
# Create a quick tunnel (generates random HTTPS URL)
cloudflared tunnel --url http://localhost:8000
```

**Output will show:**
```
Your quick Tunnel has been created! Visit it at:
https://random-name-here.trycloudflare.com
```

**Problem:** This URL changes every time you restart the tunnel. Not ideal for permanent use.

---

### Step 3: Create a Named Tunnel (Permanent URL)

This requires a one-time browser login but gives you a stable URL:

#### 3.1: Login to Cloudflare (One-Time)

```bash
cloudflared tunnel login
```

This will display a URL like:
```
Please open the following URL and log in with your Cloudflare account:
https://dash.cloudflare.com/argotunnel?aud=...
```

**What to do:**
1. Copy that URL
2. Open it in your browser (on your laptop)
3. Log in with Cloudflare account (or create free account)
4. Authorize cloudflared
5. A certificate will be saved to `~/.cloudflared/cert.pem` on the VPS

#### 3.2: Create Named Tunnel

```bash
# Create tunnel named "llm-gateway"
cloudflared tunnel create llm-gateway
```

**Output:**
```
Tunnel credentials written to /home/ubuntu/.cloudflared/UUID.json
Created tunnel llm-gateway with id UUID
```

#### 3.3: Configure Tunnel

Create config file:

```bash
nano ~/.cloudflared/config.yml
```

Paste this content:

```yaml
tunnel: llm-gateway
credentials-file: /home/ubuntu/.cloudflared/UUID.json

ingress:
  - hostname: llm-gateway.your-domain.com
    service: http://localhost:8000
  - service: http_status:404
```

**Note:** Replace `UUID.json` with the actual filename from step 3.2

If you don't have a custom domain, you can use the tunnel ID as subdomain:

```yaml
tunnel: llm-gateway
credentials-file: /home/ubuntu/.cloudflared/UUID.json

ingress:
  - service: http://localhost:8000
```

#### 3.4: Route DNS (if using custom domain)

```bash
# Replace your-domain.com with your actual domain
cloudflared tunnel route dns llm-gateway llm-gateway.your-domain.com
```

**OR use free subdomain:**

The tunnel will automatically get a URL like `https://UUID.cfargotunnel.com`

#### 3.5: Run Tunnel

```bash
cloudflared tunnel run llm-gateway
```

---

## Easier Alternative: Use trycloudflare.com (No Login)

This is the simplest approach - no account needed:

```bash
# Run quick tunnel that auto-generates HTTPS URL
cloudflared tunnel --url http://localhost:8000 2>&1 | tee ~/cloudflare-url.txt
```

**The URL will be displayed like:**
```
https://random-words-1234.trycloudflare.com
```

**To make it permanent with systemd:**

Create service file:

```bash
sudo nano /etc/systemd/system/cloudflare-tunnel.service
```

Paste:

```ini
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
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable cloudflare-tunnel
sudo systemctl start cloudflare-tunnel

# Get the HTTPS URL
sleep 10 && grep -o "https://[^[:space:]]*trycloudflare.com" ~/cloudflare-tunnel.log | tail -1
```

---

## Recommended Approach

**For quick testing:**
- Use `cloudflared tunnel --url http://localhost:8000`
- Copy the HTTPS URL
- Test in Kobold Lite

**For permanent use:**
- Create systemd service (as shown above)
- URL changes on restart but that's okay for personal use
- Or login once and create named tunnel for stable URL

---

## Alternative: Let's Encrypt + Nginx (if you have a domain)

If you own a domain name, this is the most traditional approach:

```bash
# Install Nginx and Certbot
sudo apt update
sudo apt install -y nginx certbot python3-certbot-nginx

# Configure Nginx
sudo nano /etc/nginx/sites-available/llm-gateway
```

Paste:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

Enable and get SSL:

```bash
sudo ln -s /etc/nginx/sites-available/llm-gateway /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Get free SSL certificate
sudo certbot --nginx -d your-domain.com
```

**Cons:** Requires owning a domain name ($10-15/year)

---

## Summary: Best Options

| Method | Free | Easy | Permanent | URL |
|--------|------|------|-----------|-----|
| **Cloudflare Quick Tunnel** | ✅ | ✅✅✅ | ⚠️ Changes on restart | `https://random.trycloudflare.com` |
| **Cloudflare Named Tunnel** | ✅ | ✅✅ | ✅ | `https://UUID.cfargotunnel.com` or custom |
| **Let's Encrypt + Nginx** | ✅ | ✅ | ✅ | Requires domain ($) |
| **ngrok Free** | ✅ | ✅✅✅ | ❌ Resets every 2hrs | `https://random.ngrok.io` |

**My Recommendation: Cloudflare Quick Tunnel with systemd**

Pros:
- No login required
- Free forever
- Works immediately
- URL changes on restart but that's minor for personal use

Just run:
```bash
cloudflared tunnel --url http://localhost:8000
```

Copy the HTTPS URL and use it in Kobold Lite!

---

## Next Steps

1. **Choose your approach** (Quick Tunnel recommended for simplicity)
2. **Get the HTTPS URL**
3. **Configure Kobold Lite** with the HTTPS URL
4. **Update OpenCode** if you want to use HTTPS there too (optional - HTTP works for OpenCode)

Let me know which approach you want and I can help set it up!
