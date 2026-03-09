#!/usr/bin/env bash
set -euo pipefail

prompt="${1:?prompt is required}"

payload=$(jq -n \
  --arg model "${LLM_MODEL:?LLM_MODEL is required}" \
  --arg prompt "$prompt" \
  '{
    model: $model,
    messages: [
      {role: "user", content: $prompt}
    ]
  }')

curl -fsSL \
  -H "Authorization: Bearer ${LLM_API_KEY:?LLM_API_KEY is required}" \
  -H 'Content-Type: application/json' \
  -d "$payload" \
  "${LLM_BASE_URL:?LLM_BASE_URL is required}/chat/completions" \
| jq -r '.choices[0].message.content // empty'
