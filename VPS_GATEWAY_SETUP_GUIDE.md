# OpenCode VPS Gateway - /etc/hosts Method

## ⚠️ Important SSL/TLS Warning

The /etc/hosts redirect has an SSL certificate mismatch issue:
- OpenCode connects to `https://api.openai.com`
- Your VPS gateway is `http://` (no SSL)
- The SSL certificates won't match

## Solutions:

### Solution 1: Quick Test (May Not Work)

Run the setup script and see if OpenCode handles it:

```bash
cd ~/CodingProjects
sudo ./hosts-vps-setup.sh
```

Then test OpenCode. If it gives SSL errors, use Solution 2.

---

### Solution 2: Configure VPS Gateway with SSL (RECOMMENDED)

Add HTTPS support to your VPS gateway. You can use:

**A. Self-signed certificate** (easiest, but OpenCode might reject it)
**B. Let's Encrypt** (free, trusted SSL) - but requires domain name
**C. Cloudflare Tunnel** (free, no domain needed) - see below

---

### Solution 3: Cloudflare Tunnel (EASIEST & BEST)

**Free, secure, no SSL issues, no domain needed!**

**Step 1: Install cloudflared on VPS**
```bash
# On VPS (ubuntu@40.233.101.233)
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
```

**Step 2: Login to Cloudflare**
```bash
cloudflared tunnel login
# This gives you a URL to authenticate - follow it in browser
```

**Step 3: Create tunnel**
```bash
cloudflared tunnel create opencode-gateway
```

**Step 4: Configure tunnel**
Create `~/.cloudflared/config.yml`:
```yaml
tunnel: <tunnel-id-from-step-3>
credentials-file: /home/ubuntu/.cloudflared/<tunnel-id>.json

ingress:
  - hostname: opencode-gateway-yourname.pages.dev
    service: http://localhost:8000
  - service: http_status:404
```

**Step 5: Run tunnel**
```bash
cloudflared tunnel run opencode-gateway
```

**Step 6: Get public URL**
You'll get a URL like: `https://opencode-gateway-yourname.pages.dev`

**Step 7: Configure OpenCode**
Now OpenCode can use this HTTPS URL directly (no hosts file needed)!

---

## Current Files Created:

1. **hosts-vps-setup.sh** - Sets up /etc/hosts redirect
2. **hosts-vps-revert.sh** - Automatically created by setup script to revert changes
3. **Backup of /etc/hosts** - Created with timestamp before any changes

---

## To Use /etc/hosts Method:

```bash
# 1. Run setup
cd ~/CodingProjects
sudo ./hosts-vps-setup.sh

# 2. Test DNS redirect
ping api.openai.com
# Should show: 40.233.101.233

# 3. Test OpenCode
opencode
# Configure to use 'openai' provider with API key: CHANGE_ME_TO_A_STRONG_SECRET_KEY

# 4. If issues, revert anytime:
sudo ./hosts-vps-revert.sh
```

---

## Recommendation:

**Try /etc/hosts first** - it's quick and reversible.

**If SSL errors:** Use **Cloudflare Tunnel** (Solution 3) - it's free, secure, and gives you a proper HTTPS URL that OpenCode can use directly without any hacks.

**Want me to set up Cloudflare Tunnel for you instead?** It's actually easier than it looks!
