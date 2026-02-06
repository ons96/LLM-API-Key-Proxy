#!/bin/bash

# ============================================================================
# Find and Fix Gateway Directory on VPS
# ============================================================================

echo "========================================"
echo "Gateway Directory Finder"
echo "========================================"
echo ""

echo "[1] Finding where gateway actually is..."
echo "----------------------------------------"

# Search for gateway code in common locations
FOUND=""
for dir in \
    ~/LLM-API-Key-Proxy \
    ~/CodingProjects/LLM-API-Key-Proxy \
    ~/projects/LLM-API-Key-Proxy \
    ~/llm-gateway \
    ~/gateway \
    /home/ubuntu/LLM-API-Key-Proxy \
    /home/ubuntu/gateway \
    ; do
    if [ -d "$dir" ] && [ -f "$dir/src/proxy_app/main.py" ]; then
        echo "✓ FOUND: $dir"
        FOUND="$dir"
        break
    fi
done

if [ -z "$FOUND" ]; then
    echo "✗ NOT FOUND in common locations"
    echo ""
    echo "[2] Searching with find command..."
    echo "----------------------------------------"
    FOUND=$(find ~ -name "main.py" -path "*/proxy_app/*" 2>/dev/null | head -1 | xargs dirname | xargs dirname 2>/dev/null)
    if [ -n "$FOUND" ]; then
        echo "✓ FOUND: $FOUND"
    else
        echo "✗ NOT FOUND with find either"
    fi
fi

echo ""
echo "[3] Checking current directory..."
echo "----------------------------------------"
PWD=$(pwd)
echo "Current directory: $PWD"

if [ -f "./src/proxy_app/main.py" ]; then
    echo "✓ Gateway code is in current directory!"
    FOUND="."
elif [ -f "$PWD/src/proxy_app/main.py" ]; then
    echo "✓ Gateway code is in: $PWD"
    FOUND="$PWD"
fi

echo ""
echo "========================================"
echo "Next Steps:"
echo "========================================"
echo ""

if [ -n "$FOUND" ]; then
    echo "ERROR: Could not find gateway code!"
    echo ""
    echo "Please manually check on VPS:"
    echo "  ls -la ~ | grep -i llm"
    echo "  find ~ -name 'main.py' -path '*/proxy_app/*'"
else
    echo "Gateway found at: $FOUND"
    echo ""
    echo "To restart gateway from correct location:"
    echo "  cd $FOUND"
    echo "  source venv/bin/activate"
    echo "  nohup python src/proxy_app/main.py --host 0.0.0.0 --port 8000 > ~/llm_proxy.log 2>&1 &"
fi
