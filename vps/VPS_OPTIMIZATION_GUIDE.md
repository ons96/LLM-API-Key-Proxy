# Oracle Free Tier VPS Optimization Guide
Generated: 2026-03-28

## VPS Inventory

| VPS | IP | Role | Spec |
|---|---|---|---|
| VPS1 | 40.233.101.233 | LLM Gateway (llm-api-key-proxy) | VM.Standard.E2.1.Micro: 1 OCPU AMD, 1GB RAM, 50GB disk |
| VPS2 | (check secrets) | ZeroClaw (live-swe-agent, opencode/omo) | Same spec |

## Quick Start (run on each VPS)

```bash
# SSH into VPS1
ssh -i ~/.ssh/oracle_key ubuntu@40.233.101.233

# Clone or pull latest
git clone https://github.com/ons96/LLM-API-Key-Proxy /opt/llm-gateway
# OR if already cloned:
cd /opt/llm-gateway && git pull

# Run optimization (as root)
sudo bash vps/optimize_vps.sh
```

---

## What the Script Does

### 1. zram Swap (512MB compressed in-memory swap)
Oracle free VPS has 1GB RAM. Python FastAPI + LiteLLM + SQLite can easily hit 600-800MB.
zram gives a compressed memory overflow layer with no disk I/O penalty.
- Compression ratio ~2:1 typically → effective 1GB extra
- Swappiness set to 10 (only use swap under pressure)
- Persists across reboots via `/etc/modules`

### 2. Journal Size Cap (50MB, 7-day)
Systemd journal can silently grow to fill disk.
- `SystemMaxUse=50M` hard cap
- Vacuum run immediately on install

### 3. llm-gateway Service Limits
```ini
[Service]
MemoryMax=700M       # Gateway killed before system OOM
MemorySwapMax=256M   # Can use some swap
Restart=always       # Auto-restart on crash
RestartSec=10
```
Create `/etc/systemd/system/llm-gateway.service` first (see below).

### 4. tmpfs /tmp (200MB)
The telemetry SQLite DB lives at `/tmp/llm_proxy_telemetry.db`.
tmpfs = RAM-backed, much faster than disk for frequent small writes.
Note: **data is lost on reboot** — this is fine for telemetry (it rebuilds from calls).
If you want persistence, change `TELEMETRY_DB` in telemetry.py to `/var/lib/llm-gateway/telemetry.db`.

### 5. nginx Reverse Proxy (port 80 → 8000)
Benefits:
- keepalive connections to gateway (avoids TCP overhead per request)
- Proper streaming proxy headers (`X-Accel-Buffering: no`)
- 300s read timeout for long LLM responses
- Health endpoint at `/health`

Port 80 must be open in Oracle Cloud firewall (Security List or Network Security Group).

### 6. OOM Score Adjustment
`oom_score_adj = -100` for the gateway process makes the Linux OOM killer
prefer to kill other processes first.

### 7. Network Tuning
- `tcp_tw_reuse=1`: Faster TIME_WAIT socket reuse (good for high request rate)
- `somaxconn=1024`: More pending connections accepted
- TCP keepalive tuning for long-lived LLM streaming connections

### 8. Cron Jobs
- **Nightly 03:00**: Clean old temp files, vacuum journal
- **Sunday 04:00**: Weekly gateway restart (clears any memory leaks)

---

## llm-gateway systemd Service

Create `/etc/systemd/system/llm-gateway.service`:

```ini
[Unit]
Description=LLM API Key Proxy Gateway
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/llm-gateway
ExecStart=/opt/llm-gateway/.venv/bin/python -m uvicorn src.proxy_app.main:app \
  --host 0.0.0.0 --port 8000 \
  --workers 1 \
  --loop uvloop \
  --timeout-keep-alive 120
EnvironmentFile=/opt/llm-gateway/.env
Restart=always
RestartSec=10
TimeoutStartSec=120
MemoryMax=700M
MemorySwapMax=256M

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
systemctl daemon-reload
systemctl enable llm-gateway
systemctl start llm-gateway
systemctl status llm-gateway
```

---

## Oracle Cloud Firewall Rules

You need these ports open in **both**:
1. Oracle Cloud Security List (VCN → Subnets → Security List → Ingress Rules)
2. The OS-level iptables/ufw on the VPS

| Port | Protocol | Purpose |
|---|---|---|
| 22 | TCP | SSH |
| 80 | TCP | nginx (optional, if using reverse proxy) |
| 8000 | TCP | LLM Gateway direct access |
| 443 | TCP | HTTPS (optional, add SSL later) |

OS firewall (Ubuntu):
```bash
# Allow gateway port
sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 80 -j ACCEPT
# Make persistent
sudo apt-get install -y iptables-persistent
sudo netfilter-persistent save
```

---

## Memory Budget (1GB RAM)

| Process | Typical RSS |
|---|---|
| Ubuntu OS + systemd | ~120MB |
| nginx | ~10MB |
| Python (uvicorn + FastAPI) | ~80MB |
| LiteLLM loaded | ~150MB |
| Router state (providers, virtual models) | ~30MB |
| SQLite telemetry DB (in /tmp) | ~5MB |
| **Total baseline** | **~395MB** |
| **Free for request burst** | **~600MB** |
| **zram buffer** | **~512MB compressed** |

With 1 uvicorn worker: concurrent requests are handled via async I/O, not threads.
This is fine for the proxy workload (mostly network I/O, not CPU-bound).

---

## Monitoring

```bash
# Live memory
watch -n5 free -h

# Gateway logs
journalctl -u llm-gateway -f

# Top memory consumers
ps aux --sort=-%mem | head -10

# nginx access log
tail -f /var/log/nginx/access.log

# Telemetry DB
sqlite3 /tmp/llm_proxy_telemetry.db 'SELECT provider, model, COUNT(*), AVG(tokens_per_second) FROM api_calls GROUP BY provider, model ORDER BY 3 DESC LIMIT 20;'
```

---

## VPS2 (ZeroClaw / opencode) Specific Notes

VPS2 runs opencode/Omo and the live-swe-agent. Different profile:
- Higher memory spikes from agent subprocesses
- Less predictable load (depends on task complexity)
- Recommended: increase swap to 1GB instead of 512MB
- Do NOT run llm-gateway on VPS2 (conflicts with VPS1 service)
- The optimization script still applies (zram, journal, nginx not needed)

For VPS2, skip step 5 (nginx) by commenting it out, or use the `--skip-nginx` flag
(add to script if needed).

