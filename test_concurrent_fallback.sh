#!/bin/bash
# Concurrent Request Test for LLM Gateway Fallback Verification
# This script sends multiple simultaneous requests to test fallback behavior

set -e

API_KEY="CHANGE_ME_TO_A_STRONG_SECRET_KEY"
BASE_URL="http://40.233.101.233:8000"
MODEL="coding-elite"
NUM_REQUESTS=10
CONCURRENT=5

echo "=========================================="
echo "Concurrent Request Test - Gateway Fallback"
echo "=========================================="
echo "Sending $NUM_REQUESTS requests ($CONCURRENT concurrent)"
echo "Model: $MODEL"
echo "URL: $BASE_URL"
echo ""

# Function to make a single request
make_request() {
    local id=$1
    local start_time=$(date +%s%N)
    
    response=$(curl -s -X POST "$BASE_URL/v1/chat/completions" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Request $id: Say hello briefly\"}],
            \"max_tokens\": 20
        }" 2>&1)
    
    local end_time=$(date +%s%N)
    local duration=$(( (end_time - start_time) / 1000000 ))  # Convert to milliseconds
    
    # Extract model used from response
    model_used=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('model', 'unknown'))" 2>/dev/null || echo "error")
    
    echo "Request $id: ${duration}ms | Model: $model_used"
}

export -f make_request
export API_KEY BASE_URL MODEL

echo "Starting concurrent requests..."
echo ""

# Use parallel processing if available, otherwise sequential
if command -v parallel &> /dev/null; then
    seq 1 $NUM_REQUESTS | parallel -j $CONCURRENT make_request
else
    # Fallback to background processes
    for i in $(seq 1 $NUM_REQUESTS); do
        make_request $i &
        if [ $((i % CONCURRENT)) -eq 0 ]; then
            wait
        fi
    done
    wait
fi

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="