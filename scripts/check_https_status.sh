#!/bin/bash

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Checking Permanent HTTPS Status...${NC}"

# 1. Check Tailscale
echo -e "\n${YELLOW}--- Tailscale Check ---${NC}"
TS_STATUS=$(tailscale status 2>&1)

if [[ "$TS_STATUS" == *"Logged out"* ]]; then
    echo -e "${RED}Tailscale is NOT authenticated.${NC}"
    echo "Please authenticate by visiting:"
    # Extract the auth link from 'tailscale up' if possible, or just the generic login status
    # We'll run 'tailscale up' in background to refresh link if needed, but it might block.
    # Just show the status message which usually contains the link.
    echo "$TS_STATUS"
else
    echo -e "${GREEN}Tailscale is AUTHENTICATED!${NC}"
    echo "Enabling Funnel..."
    sudo tailscale funnel --bg 8000
    TS_URL=$(tailscale funnel status | grep "https://" | awk '{print $1}')
    if [[ ! -z "$TS_URL" ]]; then
        echo -e "${GREEN}Permanent URL Active: $TS_URL${NC}"
        echo "$TS_URL/v1" > ~/permanent_url.txt
    else
        echo "Funnel enabled, checking status..."
        tailscale funnel status
    fi
fi

# 2. Check zrok
echo -e "\n${YELLOW}--- zrok Check ---${NC}"
ZROK_STATUS=$(zrok status 2>&1)
if [[ "$ZROK_STATUS" == *"enable the zrok"* ]]; then
    echo -e "${RED}zrok is NOT enabled.${NC}"
    echo "Run 'zrok enable <token>' to use zrok."
else
    echo -e "${GREEN}zrok is ENABLED!${NC}"
    # If enabled, check for reserved share
    # implementation detail...
fi

# 3. Check Current Fallback (ngrok)
echo -e "\n${YELLOW}--- Fallback (ngrok) ---${NC}"
if [ -f ~/current-https-url.txt ]; then
    NGROK_URL=$(cat ~/current-https-url.txt)
    echo -e "${GREEN}Active Fallback URL: $NGROK_URL${NC}"
else
    echo -e "${RED}No fallback URL found.${NC}"
fi
