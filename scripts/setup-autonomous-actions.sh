#!/bin/bash
# Setup script for autonomous GitHub Actions secrets
# Run this script to configure all required secrets

set -e

REPO="ons96/LLM-API-Key-Proxy"

echo "=== Autonomous GitHub Actions Setup ==="
echo ""
echo "This script will help you configure all secrets needed for"
echo "autonomous issue processing using free LLMs."
echo ""

# Check if gh is authenticated
if ! gh auth status &>/dev/null; then
    echo "Error: GitHub CLI not authenticated. Run 'gh auth login' first."
    exit 1
fi

echo "Step 1: GitHub App Setup"
echo "========================="
echo ""
echo "You need a GitHub App for the bot to create PRs."
echo ""
echo "Option A: Create a new GitHub App"
echo "  1. Go to: https://github.com/settings/apps/new"
echo "  2. Fill in:"
echo "     - App name: mirrobot-agent"
echo "     - Homepage URL: https://github.com/ons96/LLM-API-Key-Proxy"
echo "     - Webhook: uncheck Active"
echo "  3. Permissions:"
echo "     - Contents: Read and write"
echo "     - Issues: Read and write"
echo "     - Pull requests: Read and write"
echo "  4. Where can this app be installed: Any account"
echo "  5. Click 'Create app'"
echo "  6. Generate a private key and download it"
echo ""
echo "Option B: Use existing bot if you have one"
echo ""

read -p "Do you have a GitHub App? (y/n): " has_app

if [ "$has_app" = "y" ]; then
    read -p "Enter your GitHub App ID: " app_id
    read -p "Enter path to private key file: " key_path

    echo "Setting BOT_APP_ID..."
    gh secret set BOT_APP_ID --repo "$REPO" --body "$app_id"

    echo "Setting BOT_PRIVATE_KEY..."
    gh secret set BOT_PRIVATE_KEY --repo "$REPO" --body "$(cat "$key_path")"
else
    echo ""
    echo "Please create a GitHub App first, then re-run this script."
    echo "Follow the instructions above."
    exit 1
fi

echo ""
echo "Step 2: LLM Provider Secrets"
echo "============================="
echo ""
echo "You need at least one LLM provider configured."
echo ""

# Check existing secrets
existing=$(gh secret list --repo "$REPO" 2>/dev/null || echo "")

if echo "$existing" | grep -q "KILOCODE_API_KEY"; then
    echo "✓ KILOCODE_API_KEY already set"
else
    read -p "Enter Kilo Code API key (or press Enter to skip): " kilo_key
    if [ -n "$kilo_key" ]; then
        gh secret set KILOCODE_API_KEY --repo "$REPO" --body "$kilo_key"
        echo "✓ KILOCODE_API_KEY set"
    fi
fi

if echo "$existing" | grep -q "ZEN_API_KEY"; then
    echo "✓ ZEN_API_KEY already set"
else
    read -p "Enter OpenCode Zen API key (or press Enter to skip): " zen_key
    if [ -n "$zen_key" ]; then
        gh secret set ZEN_API_KEY --repo "$REPO" --body "$zen_key"
        echo "✓ ZEN_API_KEY set"
    fi
fi

# Additional free provider keys
read -p "Enter OpenRouter API key (free tier available, or press Enter to skip): " openrouter_key
if [ -n "$openrouter_key" ]; then
    gh secret set OPENROUTER_API_KEY --repo "$REPO" --body "$openrouter_key"
    echo "✓ OPENROUTER_API_KEY set"
fi

read -p "Enter Groq API key (free tier, or press Enter to skip): " groq_key
if [ -n "$groq_key" ]; then
    gh secret set GROQ_API_KEY --repo "$REPO" --body "$groq_key"
    echo "✓ GROQ_API_KEY set"
fi

read -p "Enter Together API key (or press Enter to skip): " together_key
if [ -n "$together_key" ]; then
    gh secret set TOGETHER_API_KEY --repo "$REPO" --body "$together_key"
    echo "✓ TOGETHER_API_KEY set"
fi

read -p "Enter Cerebras API key (or press Enter to skip): " cerebras_key
if [ -n "$cerebras_key" ]; then
    gh secret set CEREBRAS_API_KEY --repo "$REPO" --body "$cerebras_key"
    echo "✓ CEREBRAS_API_KEY set"
fi

# Set PROXY_API_KEY as fallback
if [ -n "$kilo_key" ]; then
    gh secret set PROXY_API_KEY --repo "$REPO" --body "$kilo_key"
    echo "✓ PROXY_API_KEY set (using Kilo key as fallback)"
elif [ -n "$zen_key" ]; then
    gh secret set PROXY_API_KEY --repo "$REPO" --body "$zen_key"
    echo "✓ PROXY_API_KEY set (using Zen key as fallback)"
fi

echo ""
echo "Step 3: Model Configuration"
echo "============================"
echo ""

# Set default models
gh secret set OPENCODE_MODEL --repo "$REPO" --body "kilocode/auto"
echo "✓ OPENCODE_MODEL set to: kilocode/auto"

gh secret set OPENCODE_FAST_MODEL --repo "$REPO" --body "kilocode/auto-fast"
echo "✓ OPENCODE_FAST_MODEL set to: kilocode/auto-fast"

echo ""
echo "Step 4: Install GitHub App"
echo "=========================="
echo ""
echo "Install the app to your repository:"
echo "1. Go to: https://github.com/settings/apps/mirrobot-agent/installations"
echo "2. Click 'Configure'"
echo "3. Select 'Only select repositories'"
echo "4. Add: ons96/LLM-API-Key-Proxy"
echo "5. Click 'Save'"
echo ""

echo "=== Setup Complete ==="
echo ""
echo "Current secrets:"
gh secret list --repo "$REPO"

echo ""
echo "To test, manually trigger the workflow:"
echo "  gh workflow run ai-batch-issue-fixer.yml --repo $REPO"
echo ""
echo "Or wait for the next scheduled run (every 6 hours)."
