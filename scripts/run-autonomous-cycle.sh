#!/bin/bash
set -e

REPO="ons96/LLM-API-Key-Proxy"

echo "=== Autonomous Agentic Coding Cycle ==="
echo ""

echo "Step 1/4: Discovering all repos..."
gh workflow run multi-repo-orchestrator.yml --repo "$REPO"
sleep 10

echo "Step 2/4: Generating issues for problems..."
gh workflow run ai-issue-generator.yml --repo "$REPO" \
  -f scan_type=full \
  -f max_issues=10
sleep 10

echo "Step 3/4: Fixing issues with free LLMs (20 parallel)..."
gh workflow run ai-batch-issue-fixer.yml --repo "$REPO" \
  -f max_issues=20 \
  -f scan_all_repos=true
sleep 10

echo "Step 4/4: Managing PRs (dry run)..."
gh workflow run pr-manager.yml --repo "$REPO" \
  -f action=all \
  -f dry_run=true

echo ""
echo "=== All workflows triggered! ==="
echo ""
echo "Monitor progress at:"
echo "  https://github.com/$REPO/actions"
echo ""
echo "Or watch with:"
echo "  gh run watch --repo $REPO"
