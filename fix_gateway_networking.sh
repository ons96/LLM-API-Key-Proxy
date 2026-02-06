#!/bin/bash

# ============================================================================
# Fix Gateway Listening Issue - Check and Restart Properly
# ============================================================================

GATEWAY_PORT="9099"

echo "========================================"
echo "Gateway Network Diagnostics"
echo "========================================"
echo ""

echo "[1] Checking if gateway is running..."
if ps aux | grep -v grep | grep -q "main.py.*${GATEWAY_PORT}"; then
    echo "✓ Gateway process IS running"
    PID=$(ps aux | grep -v grep | grep "main.py.*${GATEWAY_PORT}" | awk '{print $2}')
    echo "  PID: ${PID}"
else
    echo "✗ Gateway process is NOT running"
    echo ""
    echo "Starting gateway with correct host binding..."
    cd ~/CodingProjects/LLM-API-Key-Proxy
    source venv/bin/activate
    nohup python src/proxy_app/main.py --host 0.0.0.0 --port ${GATEWAY_PORT} > ~/llm_proxy.log 2>&1 &
    sleep 5
fi

echo ""
echo "[2] Checking what address gateway is listening on..."
if netstat -tlnp 2>/dev/null | grep -q ":${GATEWAY_PORT} "; then
    echo "Listening addresses:"
    netstat -tlnp 2>/dev/null | grep ":${GATEWAY_PORT}" | awk '{print "  " $4 ":" $5}'
elif ss -tlnp 2>/dev/null | grep -q ":${GATEWAY_PORT}"; then
    echo "Listening addresses:"
    ss -tlnp 2>/dev/null | grep ":${GATEWAY_PORT}" | awk '{print "  " $4 ":" $5}'
else
    echo "✗ Port ${GATEWAY_PORT} is NOT listening"
fi

echo ""
echo "[3] Checking firewall rules..."
if sudo ufw status 2>/dev/null | grep -q "inactive"; then
    echo "✓ UFW firewall is DISABLED (good)"
elif command -v ufw; then
    echo "UFW firewall status:"
    sudo ufw status | grep "${GATEWAY_PORT}" || echo "  (No specific rule for port ${GATEWAY_PORT})"
else
    echo "  UFW not installed (Oracle Cloud uses iptables/security groups)"
fi

echo ""
echo "[4] Testing from VPS localhost..."
if curl -s --connect-timeout 2 http://127.0.0.1:${GATEWAY_PORT}/v1/models > /dev/null 2>&1; then
    echo "✓ Gateway accessible from localhost (127.0.0.1)"
else
    echo "✗ Gateway NOT accessible from localhost"
fi

echo ""
echo "[5] Testing from VPS public IP..."
PUBLIC_IP=$(curl -s --connect-timeout 2 ifconfig.me 2>/dev/null || curl -s --connect-timeout 2 icanhazip.com 2>/dev/null || echo "unknown")
if [ "$PUBLIC_IP" != "unknown" ]; then
    if curl -s --connect-timeout 2 http://${PUBLIC_IP}:${GATEWAY_PORT}/v1/models > /dev/null 2>&1; then
        echo "✓ Gateway accessible from public IP (${PUBLIC_IP})"
    else
        echo "✗ Gateway NOT accessible from public IP (${PUBLIC_IP})"
        echo "  This means firewall or host binding issue"
    fi
else
    echo "⚠ Could not determine public IP"
fi

echo ""
echo "========================================"
echo "Diagnostic Complete"
echo "========================================"
echo ""
echo "If gateway is accessible from localhost but NOT from public IP:"
echo "  1. Firewall blocking port ${GATEWAY_PORT} - Check Oracle Cloud security groups"
echo "  2. Gateway bound to 127.0.0.1 instead of 0.0.0.0"
echo ""
echo "To allow external access on VPS:"
echo "  sudo ufw allow ${GATEWAY_PORT}"
echo ""
echo "To restart gateway on all interfaces (0.0.0.0):"
echo "  pkill -f 'main.py.*${GATEWAY_PORT}'"
echo "  cd ~/CodingProjects/LLM-API-Key-Proxy"
echo "  source venv/bin/activate"
echo "  nohup python src/proxy_app/main.py --host 0.0.0.0 --port ${GATEWAY_PORT} > ~/llm_proxy.log 2>&1 &"
