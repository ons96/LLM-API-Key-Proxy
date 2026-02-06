#!/bin/bash

# ============================================================================
# Check Open Ports and Find Available Port for Gateway
# ============================================================================

echo "========================================"
echo "VPS Port Checker"
echo "========================================"
echo ""

echo "[1] Checking currently listening ports..."
echo "----------------------------------------"
if netstat -tlnp 2>/dev/null; then
    netstat -tlnp 2>/dev/null | grep -E ":(90|80|8080|8000|3000|5000)" | head -20
elif ss -tlnp 2>/dev/null; then
    ss -tlnp 2>/dev/null | grep -E ":(90|80|8080|8000|3000|5000)" | head -20
else
    echo "Cannot determine listening ports (netstat/ss not available)"
fi
echo ""

echo "[2] Checking Oracle Cloud firewall (iptables)..."
echo "----------------------------------------"
if command -v iptables; then
    echo "INPUT chain (incoming):"
    sudo iptables -L INPUT -n --line-numbers 2>/dev/null | grep -E "dpt:(90|80|8080|8000|3000|5000)" | head -10
else
    echo "iptables not available"
fi
echo ""

echo "[3] Checking what port gateway is actually using..."
echo "----------------------------------------"
if ps aux | grep -v grep | grep -q "main.py"; then
    GATEWAY_PID=$(ps aux | grep -v grep | grep "main.py" | awk '{print $2}' | head -1)
    PORT=$(sudo lsof -p -n -i -P | grep "$GATEWAY_PID" | grep -oE ":[0-9]+" | head -1 | tr -d :)
    if [ -n "$PORT" ]; then
        echo "Gateway PID: $GATEWAY_PID"
        echo "Gateway listening on port: $PORT"
    else
        echo "Gateway PID: $GATEWAY_PID"
        echo "Could not determine port - might be starting up"
    fi
else
    echo "Gateway process not found"
fi
echo ""

echo "========================================"
echo "Recommended Actions:"
echo "========================================"
echo ""
echo "If 9099 is NOT listening:"
echo "  1. Kill current gateway: pkill -f 'main.py.*9099'"
echo "  2. Try port 8000 (common, likely open)"
echo "  3. Or check Oracle Cloud Console > Networking > Security Groups"
echo ""
echo "To open port in Oracle Cloud Console:"
echo "  1. Go to https://cloud.oracle.com"
echo "  2. Navigate: Compute > Instances > [Your Instance]"
echo "  3. Click: Networking > Security Groups"
echo "  4. Add Ingress Rule for port you want (e.g., 9099, 8000)"
echo "  5. Source CIDR: 0.0.0.0/0 (allow from anywhere)"
echo "  6. Protocol: TCP"
echo ""
