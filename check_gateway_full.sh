#!/bin/bash

# ============================================================================
# Check Gateway Status - Comprehensive
# ============================================================================

echo "========================================"
echo "Gateway Status Check"
echo "========================================"
echo ""

echo "[1] Gateway Process Status..."
if ps aux | grep -v grep | grep -q "main.py"; then
    echo "✓ Gateway process IS running"
    echo ""
    ps aux | grep -v grep | grep "main.py" | head -1
    echo ""
    GATEWAY_PID=$(ps aux | grep -v grep | grep "main.py" | awk '{print $2}' | head -1)
else
    echo "✗ Gateway process is NOT running"
    echo ""
    echo "Starting gateway..."
    cd ~/CodingProjects/LLM-API-Key-Proxy
    source venv/bin/activate
    nohup python src/proxy_app/main.py --host 0.0.0.0 --port 8000 > ~/llm_proxy.log 2>&1 &
    sleep 3
    ps aux | grep "main.py" | head -1
    GATEWAY_PID=$(ps aux | grep "main.py" | awk '{print $2}' | head -1)
fi

echo ""
echo "[2] All Listening Ports (no filter)..."
echo "----------------------------------------"
if ss -tlnp 2>/dev/null; then
    ss -tlnp 2>/dev/null | grep -E ":(80|8000|9099|3000|5000)"
elif netstat -tlnp 2>/dev/null; then
    netstat -tlnp 2>/dev/null | grep -E ":(80|8000|9099|3000|5000)"
else
    echo "Cannot determine ports"
fi

echo ""
echo "[3] Gateway Log (last 10 lines)..."
echo "----------------------------------------"
if [ -f ~/llm_proxy.log ]; then
    tail -10 ~/llm_proxy.log
else
    echo "No log file found at: ~/llm_proxy.log"
    echo "Checking alternative location..."
    if [ -f ~/CodingProjects/LLM-API-Key-Proxy/llm_proxy.log ]; then
        tail -10 ~/CodingProjects/LLM-API-Key-Proxy/llm_proxy.log
    fi
fi

echo ""
echo "========================================"
echo "If gateway is running, test with:"
echo "  curl -s http://<YOUR_IP>:<PORT>/v1/models | jq '.data | length'"
echo "========================================"
