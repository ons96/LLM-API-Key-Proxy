#!/bin/bash
#
# CLIProxyAPI Integration Test Script
#
# Tests the integration between CLIProxyAPI sidecar and the LLM gateway.
# Run after both services are configured and running.
#
# Usage:
#   ./test_cliproxyapi_integration.sh [gateway_url] [cliproxyapi_url]
#

set -e

# Configuration
GATEWAY_URL="${1:-http://127.0.0.1:8000}"
CLIPROXYAPI_URL="${2:-http://127.0.0.1:8317}"
API_KEY="${API_KEY:-test-key}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS=0
FAIL=0

test_start() {
    echo -e "\n${BLUE}TEST: $1${NC}"
}

test_pass() {
    echo -e "${GREEN}  ✓ PASS${NC}"
    ((PASS++))
}

test_fail() {
    echo -e "${RED}  ✗ FAIL: $1${NC}"
    ((FAIL++))
}

test_cliProxyapi_health() {
    test_start "CLIProxyAPI Health Check"

    RESPONSE=$(curl -s -w "\n%{http_code}" "${CLIPROXYAPI_URL}/health" 2>/dev/null)
    HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
    BODY=$(echo "$RESPONSE" | head -n -1)

    if [[ "$HTTP_CODE" == "200" ]]; then
        echo "  Response: $BODY"
        test_pass
    else
        test_fail "HTTP $HTTP_CODE - CLIProxyAPI not responding"
    fi
}

test_cliProxyapi_models() {
    test_start "CLIProxyAPI Models Endpoint"

    RESPONSE=$(curl -s -w "\n%{http_code}" "${CLIPROXYAPI_URL}/v1/models" 2>/dev/null)
    HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
    BODY=$(echo "$RESPONSE" | head -n -1)

    if [[ "$HTTP_CODE" == "200" ]]; then
        MODEL_COUNT=$(echo "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('data',[])))" 2>/dev/null || echo "0")
        echo "  Found $MODEL_COUNT models"
        if [[ "$MODEL_COUNT" -gt 0 ]]; then
            test_pass
        else
            test_fail "No models returned - check provider authentication"
        fi
    else
        test_fail "HTTP $HTTP_CODE"
    fi
}

test_cliProxyapi_backend_gemini() {
    test_start "CLIProxyAPI Gemini Backend"

    RESPONSE=$(curl -s "${CLIPROXYAPI_URL}/v1/models" 2>/dev/null)
    HAS_GEMINI=$(echo "$RESPONSE" | grep -c '"gemini/' || echo "0")

    if [[ "$HAS_GEMINI" -gt 0 ]]; then
        echo "  Gemini models found"
        test_pass
    else
        test_fail "No Gemini models - run: ./cliproxyapi -gemini-login"
    fi
}

test_cliProxyapi_backend_iflow() {
    test_start "CLIProxyAPI iFlow Backend"

    RESPONSE=$(curl -s "${CLIPROXYAPI_URL}/v1/models" 2>/dev/null)
    HAS_IFLOW=$(echo "$RESPONSE" | grep -c '"iflow/' || echo "0")

    if [[ "$HAS_IFLOW" -gt 0 ]]; then
        echo "  iFlow models found"
        test_pass
    else
        test_fail "No iFlow models - run: ./cliproxyapi -iflow-cookie"
    fi
}

test_gateway_health() {
    test_start "Gateway Health Check"

    RESPONSE=$(curl -s -w "\n%{http_code}" "${GATEWAY_URL}/health" 2>/dev/null)
    HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)

    if [[ "$HTTP_CODE" == "200" ]]; then
        test_pass
    else
        test_fail "HTTP $HTTP_CODE - Gateway not responding"
    fi
}

test_gateway_cliproxyapi_endpoint() {
    test_start "Gateway CLIProxyAPI Health Endpoint"

    RESPONSE=$(curl -s -w "\n%{http_code}" "${GATEWAY_URL}/api/cliproxyapi/health" 2>/dev/null)
    HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
    BODY=$(echo "$RESPONSE" | head -n -1)

    if [[ "$HTTP_CODE" == "200" ]]; then
        echo "  Response: $BODY"
        test_pass
    elif [[ "$HTTP_CODE" == "404" ]]; then
        test_fail "Endpoint not found - gateway may not have CLIProxyAPI integration"
    else
        test_fail "HTTP $HTTP_CODE"
    fi
}

test_chat_completion_gemini() {
    test_start "Chat Completion via Gemini Backend"

    RESPONSE=$(curl -s -w "\n%{http_code}" \
        -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" \
        -d '{"model":"gemini/gemini-2.5-flash","messages":[{"role":"user","content":"Say hello in one word"}],"max_tokens":5}' \
        "${GATEWAY_URL}/v1/chat/completions" 2>/dev/null)

    HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
    BODY=$(echo "$RESPONSE" | head -n -1)

    if [[ "$HTTP_CODE" == "200" ]]; then
        CONTENT=$(echo "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('choices',[{}])[0].get('message',{}).get('content',''))" 2>/dev/null || echo "")
        echo "  Response: $CONTENT"
        test_pass
    else
        test_fail "HTTP $HTTP_CODE - $BODY"
    fi
}

test_chat_completion_iflow() {
    test_start "Chat Completion via iFlow Backend"

    RESPONSE=$(curl -s -w "\n%{http_code}" \
        -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" \
        -d '{"model":"iflow/glm-4-flash","messages":[{"role":"user","content":"Say hello"}],"max_tokens":5}' \
        "${GATEWAY_URL}/v1/chat/completions" 2>/dev/null)

    HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
    BODY=$(echo "$RESPONSE" | head -n -1)

    if [[ "$HTTP_CODE" == "200" ]]; then
        CONTENT=$(echo "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('choices',[{}])[0].get('message',{}).get('content',''))" 2>/dev/null || echo "")
        echo "  Response: $CONTENT"
        test_pass
    else
        test_fail "HTTP $HTTP_CODE - $BODY"
    fi
}

test_streaming() {
    test_start "Streaming Chat Completion"

    RESPONSE=$(curl -s -w "\n%{http_code}" \
        -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" \
        -d '{"model":"gemini/gemini-2.5-flash","messages":[{"role":"user","content":"Count from 1 to 3"}],"max_tokens":10,"stream":true}' \
        "${GATEWAY_URL}/v1/chat/completions" 2>/dev/null)

    HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
    BODY=$(echo "$RESPONSE" | head -n -1)

    if [[ "$HTTP_CODE" == "200" ]]; then
        HAS_DATA=$(echo "$BODY" | grep -c "^data:" || echo "0")
        if [[ "$HAS_DATA" -gt 0 ]]; then
            echo "  Received $HAS_DATA SSE events"
            test_pass
        else
            test_fail "No SSE data received"
        fi
    else
        test_fail "HTTP $HTTP_CODE"
    fi
}

print_summary() {
    echo ""
    echo "=========================================="
    echo " Test Summary"
    echo "=========================================="
    echo -e "  ${GREEN}Passed: ${PASS}${NC}"
    echo -e "  ${RED}Failed: ${FAIL}${NC}"
    echo ""

    if [[ $FAIL -eq 0 ]]; then
        echo -e "${GREEN}All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}Some tests failed. Check configuration.${NC}"
        exit 1
    fi
}

# Main
main() {
    echo "=========================================="
    echo " CLIProxyAPI Integration Tests"
    echo "=========================================="
    echo "Gateway:       ${GATEWAY_URL}"
    echo "CLIProxyAPI:   ${CLIPROXYAPI_URL}"
    echo ""

    # CLIProxyAPI tests
    echo -e "\n${YELLOW}=== CLIProxyAPI Sidecar Tests ===${NC}"
    test_cliProxyapi_health
    test_cliProxyapi_models
    test_cliProxyapi_backend_gemini
    test_cliProxyapi_backend_iflow

    # Gateway tests
    echo -e "\n${YELLOW}=== Gateway Tests ===${NC}"
    test_gateway_health
    test_gateway_cliproxyapi_endpoint

    # Integration tests
    echo -e "\n${YELLOW}=== Integration Tests ===${NC}"
    test_chat_completion_gemini
    test_chat_completion_iflow
    test_streaming

    print_summary
}

main "$@"
