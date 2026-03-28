#!/bin/bash
# Oracle Free Tier VPS Optimization Script
# Supports both VPS1 (VM.Standard.E2.1.Micro: 1 OCPU AMD, 1GB RAM) 
# and VPS2 (same spec or A1.Flex ARM)
# Run as root or with sudo: bash optimize_vps.sh
# Safe to re-run (idempotent)

set -euo pipefail
LOG="/var/log/vps_optimize.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== VPS Optimization $(date -u) ==="

# -------------------------------------------------------------------------
# 1. SWAP: Add 1GB zram swap (in-memory compressed swap, no disk I/O)
# Better than a swapfile on the 50GB boot volume for a 1GB RAM instance.
# -------------------------------------------------------------------------
if ! lsmod | grep -q zram; then
  echo "[swap] Loading zram module..."
  modprobe zram
fi

if [ ! -b /dev/zram0 ] 2>/dev/null || [ "$(cat /sys/block/zram0/disksize 2>/dev/null)" = "0" ]; then
  echo "[swap] Configuring 512MB zram swap..."
  echo lz4 > /sys/block/zram0/comp_algorithm 2>/dev/null || echo zstd > /sys/block/zram0/comp_algorithm || true
  echo 536870912 > /sys/block/zram0/disksize  # 512MB
  mkswap /dev/zram0
  swapon -p 10 /dev/zram0
  echo "[swap] zram0 512MB swap enabled"
else
  echo "[swap] zram0 already configured: $(cat /sys/block/zram0/disksize) bytes"
fi

# Tune swappiness: use swap only under pressure (not proactively)
sysctl -w vm.swappiness=10 > /dev/null
sysctl -w vm.vfs_cache_pressure=50 > /dev/null
echo "[swap] swappiness=10, vfs_cache_pressure=50"

# Make zram persistent across reboots
if ! grep -q zram /etc/modules 2>/dev/null; then
  echo zram >> /etc/modules
fi

# -------------------------------------------------------------------------
# 2. SYSTEMD JOURNAL: cap journal size to prevent disk bloat
# -------------------------------------------------------------------------
mkdir -p /etc/systemd/journald.conf.d/
cat > /etc/systemd/journald.conf.d/size-limit.conf << 'EOF'
[Journal]
SystemMaxUse=50M
RuntimeMaxUse=50M
MaxFileSec=7day
ForwardToSyslog=no
EOF
systemctl restart systemd-journald
echo "[journal] Capped journal to 50MB, 7-day retention"

# Vacuum existing journal
journalctl --vacuum-size=50M --vacuum-time=7d

# -------------------------------------------------------------------------
# 3. PROCESS LIMITS: systemd slice for llm-gateway to prevent OOM cascade
# -------------------------------------------------------------------------
mkdir -p /etc/systemd/system/llm-gateway.service.d/
cat > /etc/systemd/system/llm-gateway.service.d/limits.conf << 'EOF'
[Service]
# Memory limit: 700MB of 1GB RAM, OOM-kills the gateway before system OOM
MemoryMax=700M
MemorySwapMax=256M
# Restart on crash
Restart=always
RestartSec=10
# Give it time to start (model loading)
TimeoutStartSec=120
EOF
systemctl daemon-reload
echo "[limits] Applied memory limits to llm-gateway service"

# -------------------------------------------------------------------------
# 4. TMPFS: Mount /tmp as tmpfs (reduces disk wear, faster temp files)
# -------------------------------------------------------------------------
if ! mount | grep -q 'tmpfs on /tmp'; then
  mount -t tmpfs -o size=200M,mode=1777 tmpfs /tmp
  echo "[tmpfs] Mounted /tmp as 200MB tmpfs"
fi

# Make tmpfs persistent
if ! grep -q 'tmpfs /tmp' /etc/fstab; then
  echo 'tmpfs /tmp tmpfs defaults,size=200M,mode=1777 0 0' >> /etc/fstab
  echo "[fstab] Added tmpfs /tmp to fstab"
fi

# -------------------------------------------------------------------------
# 5. NGINX: Install/configure reverse proxy with keepalive for gateway
# -------------------------------------------------------------------------
if ! command -v nginx >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y nginx --no-install-recommends
fi

cat > /etc/nginx/sites-available/llm-gateway << 'EOF'
upstream llm_gateway {
    server 127.0.0.1:8000;
    keepalive 32;   # Keep 32 connections alive to backend
}

server {
    listen 80;
    server_name _;

    # Forward proxy timeouts tuned for LLM streaming
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
    proxy_connect_timeout 10s;

    # Enable keepalive to upstream
    proxy_http_version 1.1;
    proxy_set_header Connection "";

    # Pass real client IP
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

    location / {
        proxy_pass http://llm_gateway;
        # Required for SSE/streaming responses
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header X-Accel-Buffering no;
    }

    location /health {
        proxy_pass http://llm_gateway/health;
        proxy_read_timeout 5s;
    }
}
EOF

ln -sf /etc/nginx/sites-available/llm-gateway /etc/nginx/sites-enabled/llm-gateway
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx || systemctl start nginx
echo "[nginx] Configured reverse proxy with keepalive on port 80 -> 8000"

# -------------------------------------------------------------------------
# 6. OOM SCORE: Make llm-gateway less likely to be OOM-killed than the OS
# -------------------------------------------------------------------------
if systemctl is-active llm-gateway >/dev/null 2>&1; then
  GATEWAY_PID=$(systemctl show -p MainPID --value llm-gateway)
  if [ "$GATEWAY_PID" != "0" ]; then
    echo -100 > /proc/$GATEWAY_PID/oom_score_adj
    echo "[oom] Set oom_score_adj=-100 for llm-gateway PID $GATEWAY_PID"
  fi
fi

# -------------------------------------------------------------------------
# 7. KERNEL PARAMS: Network tuning for proxy workload
# -------------------------------------------------------------------------
cat > /etc/sysctl.d/99-llm-proxy.conf << 'EOF'
# Increase TCP connection tracking
net.core.somaxconn = 1024
net.ipv4.tcp_max_syn_backlog = 512
# Reuse TIME_WAIT sockets faster
net.ipv4.tcp_tw_reuse = 1
# Keep connections alive longer (good for streaming LLM responses)
net.ipv4.tcp_keepalive_time = 60
net.ipv4.tcp_keepalive_intvl = 10
net.ipv4.tcp_keepalive_probes = 5
# Reduce memory overhead per socket
net.core.rmem_default = 131072
net.core.wmem_default = 131072
EOF
sysctl -p /etc/sysctl.d/99-llm-proxy.conf > /dev/null
echo "[net] Applied network tuning params"

# -------------------------------------------------------------------------
# 8. CRON: nightly cleanup and benchmark
# -------------------------------------------------------------------------
CRONFILE="/etc/cron.d/llm-gateway"
cat > "$CRONFILE" << 'EOF'
# Nightly cleanup at 03:00 UTC
0 3 * * * root /usr/bin/find /tmp -name '*.db' -mtime +7 -delete 2>/dev/null; journalctl --vacuum-size=40M > /dev/null 2>&1
# Weekly restart at 04:00 Sunday to clear any memory leaks
0 4 * * 0 root systemctl restart llm-gateway 2>/dev/null || true
EOF
chmod 644 "$CRONFILE"
echo "[cron] Set up nightly cleanup and weekly restart"

# -------------------------------------------------------------------------
# 9. PRINT SUMMARY
# -------------------------------------------------------------------------
echo ""
echo "=== Optimization Complete ==="
echo "Memory:"
free -h
echo "Swap:"
swapon --show
echo "Disk:"
df -h / /tmp 2>/dev/null
echo ""
echo "Services:"
systemctl is-active nginx 2>/dev/null && echo "  nginx: active" || echo "  nginx: inactive"
systemctl is-active llm-gateway 2>/dev/null && echo "  llm-gateway: active" || echo "  llm-gateway: not found (install separately)"
echo ""
echo "DONE. Log: $LOG"
