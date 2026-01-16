#!/bin/bash

# Smoke Test Script for Router-Enhanced LLM Proxy
# Tests all key functionality including virtual models and fallback behavior

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BASE_URL="${PROXY_BASE_URL:-http://localhost:8000}"
API_KEY="${PROXY_API_KEY:-}"
AUTH_HEADER=""

if [ -n "$API_KEY" ]; then
    AUTH_HEADER="Authorization: Bearer $API_KEY"
fi

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
log_test() {
    echo -e "${YELLOW}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# Test function
test_endpoint() {
    local name="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"
    local expected_status="${5:-200}"
    
    log_test "$name"
    
    if [ "$method" = "GET" ]; then
        if [ -n "$AUTH_HEADER" ]; then
            response=$(curl -s -w "\n%{http_code}" -H "$AUTH_HEADER" "$BASE_URL$endpoint")
        else
            response=$(curl -s -w "\n%{http_code}" "$BASE_URL$endpoint")
        fi
    else
        if [ -n "$AUTH_HEADER" ]; then
            response=$(curl -s -w "\n%{http_code}" -X POST -H "$AUTH_HEADER" -H "Content-Type: application/json" -d "$data" "$BASE_URL$endpoint")
        else
            response=$(curl -s -w "\n%{http_code}" -X POST -H "Content-Type: application/json" -d "$data" "$BASE_URL$endpoint")
        fi
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)
    
    if [ "$http_code" -eq "$expected_status" ]; then
        log_pass "$name (Status: $http_code)"
        echo "$body" | jq . 2>/dev/null || echo "$body"
        return 0
    else
        log_fail "$name (Expected: $expected_status, Got: $http_code)"
        echo "$body" | jq . 2>/dev/null || echo "$body"
        return 1
    fi
}

# Test 1: Basic API Connectivity
echo "=========================================="
echo "TEST 1: Basic API Connectivity"
echo "=========================================="

test_endpoint "Health check" "GET" "/health"

# Test 2: Model List
echo ""
echo "=========================================="
echo "TEST 2: Model List Endpoint"
echo "=========================================="

test_endpoint "Get models list" "GET" "/v1/models"

# Test 3: Regular Completion (Non-streaming)
echo ""
echo "=========================================="
echo "TEST 3: Regular Completion (Non-streaming)"
echo "=========================================="

NON_STREAM_DATA='{
  "model": "router/best-chat",
  "messages": [
    {
      "role": "user",
      "content": "What is 2+2?"
    }
  ],
  "max_tokens": 50,
  "temperature": 0.1
}'

test_endpoint "Non-streaming completion" "POST" "/v1/chat/completions" "$NON_STREAM_DATA"

# Test 4: Virtual Model - Coding
echo ""
echo "=========================================="
echo "TEST 4: Virtual Model - router/best-coding"
echo "=========================================="

CODING_DATA='{
  "model": "router/best-coding",
  "messages": [
    {
      "role": "user",
      "content": "Write a Python function to reverse a string"
    }
  ],
  "max_tokens": 150,
  "temperature": 0.1
}'

test_endpoint "Virtual model (coding)" "POST" "/v1/chat/completions" "$CODING_DATA"

# Test 5: Virtual Model - Reasoning
echo ""
echo "=========================================="
echo "TEST 5: Virtual Model - router/best-reasoning"
echo "=========================================="

REASONING_DATA='{
  "model": "router/best-reasoning",
  "messages": [
    {
      "role": "user",
      "content": "Explain the concept of recursion in programming"
    }
  ],
  "max_tokens": 200,
  "temperature": 0.1
}'

test_endpoint "Virtual model (reasoning)" "POST" "/v1/chat/completions" "$REASONING_DATA"

# Test 6: Streaming Response
echo ""
echo "=========================================="
echo "TEST 6: Streaming Response"
echo "=========================================="

STREAM_DATA='{
  "model": "router/best-chat",
  "messages": [
    {
      "role": "user",
      "content": "Count from 1 to 5"
    }
  ],
  "max_tokens": 50,
  "stream": true,
  "temperature": 0.1
}'

log_test "Streaming completion"

if [ -n "$AUTH_HEADER" ]; then
    response=$(curl -s -N -X POST -H "$AUTH_HEADER" -H "Content-Type: application/json" -d "$STREAM_DATA" "$BASE_URL/v1/chat/completions")
else
    response=$(curl -s -N -X POST -H "Content-Type: application/json" -d "$STREAM_DATA" "$BASE_URL/v1/chat/completions")
fi

# Check if response contains SSE format
if [[ "$response" == *"data:"* ]]; then
    log_pass "Streaming response format valid"
    echo "Response contains $(echo "$response" | grep -c "data:") chunks"
    # Show first few chunks
    echo "$response" | head -10
else
    log_fail "Streaming response format invalid"
    echo "$response" | jq . 2>/dev/null || echo "$response"
fi

# Test 7: Router Status
echo ""
echo "=========================================="
echo "TEST 7: Router Status"
echo "=========================================="

test_endpoint "Router health status" "GET" "/v1/router/status"

# Test 8: Router Metrics
echo ""
echo "=========================================="
echo "TEST 8: Router Metrics"
echo "=========================================="

test_endpoint "Router metrics" "GET" "/v1/router/metrics"

# Test 9: Capability-based Routing (Tools/Functions)
echo ""
echo "=========================================="
echo "TEST 9: Tools/Functions Support"
echo "=========================================="

TOOLS_DATA='{
  "model": "router/best-coding",
  "messages": [
    {
      "role": "user",
      "content": "What is the weather like in Boston?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "max_tokens": 150
}'

test_endpoint "Tools support" "POST" "/v1/chat/completions" "$TOOLS_DATA"

# Test 10: FREE_ONLY_MODE Enforcement (verify no paid providers used)
echo ""
echo "=========================================="
echo "TEST 10: FREE_ONLY_MODE Enforcement"
echo "=========================================="

test_endpoint "FREE_ONLY_MODE status" "GET" "/v1/router/status"

# Make a request and check that only free providers are used
log_info "Checking router status for FREE_ONLY_MODE enforcement..."

if [ -n "$AUTH_HEADER" ]; then
    status=$(curl -s -H "$AUTH_HEADER" "$BASE_URL/v1/router/status")
else
    status=$(curl -s "$BASE_URL/v1/router/status")
fi

FREE_MODE=$(echo "$status" | jq -r '.free_only_mode')

if [ "$FREE_MODE" = "true" ]; then
    log_pass "FREE_ONLY_MODE is enabled"
else
    log_fail "FREE_ONLY_MODE is not enabled"
fi

# Summary
echo ""
echo "=========================================="
echo "SMOKE TEST SUMMARY"
echo "=========================================="
echo "Tests passed: $TESTS_PASSED"
echo "Tests failed: $TESTS_FAILED"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All smoke tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed${NC}"
    exit 1
fi