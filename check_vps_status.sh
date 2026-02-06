#!/bin/bash

# ============================================================================
# Quick VPS Connection Script - Check Gateway Status
# ============================================================================

VPS_HOST="40.233.101.233"
VPS_USER="ubuntu"
SSH_KEY="~/.ssh/oracle.key"
GATEWAY_PORT="9099"

echo "========================================"
echo "Connecting to VPS: ${VPS_USER}@${VPS_HOST}"
echo "========================================"
echo ""

# Connect and check gateway status
ssh -i "${SSH_KEY}" "${VPS_USER}@${VPS_HOST}" << 'ENDSSH'
echo "Checking if LLM Gateway is running..."
echo ""

# Check if process is running
if ps aux | grep -v grep | grep -q "main.py.*${GATEWAY_PORT}"; then
    echo "✓ Gateway process is RUNNING"
    PID=$(ps aux | grep -v grep | grep "main.py.*${GATEWAY_PORT}" | awk '{print $2}')
    echo "  PID: ${PID}"
else
    echo "✗ Gateway process is NOT running"
fi

echo ""
echo "Checking port ${GATEWAY_PORT}..."

# Check if port is listening
if netstat -tlnp 2>/dev/null | grep -q ":${GATEWAY_PORT} "; then
    echo "✓ Port ${GATEWAY_PORT} is LISTENING"
else
    echo "✗ Port ${GATEWAY_PORT} is NOT listening"
    echo ""
    echo "Checking what ports ARE listening on 90xx range..."
    netstat -tlnp 2>/dev/null | grep ":90" || ss -tlnp 2>/dev/null | grep ":90"
fi

echo ""
echo "Checking gateway logs (last 20 lines)..."
if [ -f ~/llm_proxy.log ]; then
    echo "---"
    tail -20 ~/llm_proxy.log
elif [ -f ~/CodingProjects/LLM-API-Key-Proxy/llm_proxy.log ]; then
    echo "---"
    tail -20 ~/CodingProjects/LLM-API-Key-Proxy/llm_proxy.log
else
    echo "✗ No gateway log file found"
    echo "  Searched: ~/llm_proxy.log"
    echo "            ~/CodingProjects/LLM-API-Key-Proxy/llm_proxy.log"
fi

echo ""
echo "========================================"
echo "VPS Check Complete"
echo "========================================"
ENDSSH
