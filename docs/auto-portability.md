# 'auto' Virtual Model Portability Guide

Status: **PARTIAL** — `auto` model shipped in PR #272 (branch `feat/auto-semantic-routing`), awaiting merge to `main` and gateway redeploy. This doc covers the config template that makes `auto` portable across opencode instances.

Refs: task-board #203, #197 (PR #272), #194 (parent epic).

## What `auto` does

`model: auto` is a virtual model registered in `config/virtual_models.yaml` on the gateway. When a client sends `{"model": "auto", ...}`, the gateway's `router_wrapper.py` intercepts it, calls `semantic_router.resolve_auto(request)`, and mutates `request_data["model"]` to the resolved chain (e.g. `coding-elite`, `chat-fast`, `chat-rp`) before delegating to `router_core`. The client never sees the resolution — it just gets a normal OpenAI-compatible response from whatever chain the gateway picked.

Routing decision is based on:
1. **Embedding similarity** via `semantic-router[local]` (FastEmbed ONNX, all-MiniLM-L6-v2, ~80MB, local-only, no API key). Falls back to keyword `intent_detector.py` on low confidence or missing library.
2. **Tool-capability guard**: if request carries `tools` or `tool_choice`, only tool-capable chains (`coding-*`, `chat-smart`, `chat-elite`, `agent-*`, `glm5-elite`) are eligible. Non-tool intents (FAST_CHAT, ROLEPLAY) get rerouted to `coding-fast` (safe default).
3. **Env knobs**: `AUTO_ROUTE_MODE=all|ambiguous` (default `all`), `AUTO_ROUTE_THRESHOLD=0.30` (cosine sim cutoff for semantic match).

See `src/proxy_app/semantic_router.py` for full impl.

## Per-machine opencode config (minimal)

Each opencode instance (laptop, VPS-155, future devices) needs only:

1. Gateway provider entry in `opencode.json` (or `.jsonc`) pointing at the gateway base URL.
2. `auto` in the model list for that provider.
3. Tailscale up (or SSH tunnel up) so the gateway URL is reachable.

### Template (drop into `opencode.json` → `provider` block)

```json
{
  "vps-gateway": {
    "baseURL": "http://localhost:8000/v1",
    "apiKey": "{env:VPS_GATEWAY_API_KEY}",
    "models": {
      "auto": {},
      "coding-elite": {},
      "coding-fast": {},
      "chat-fast": {},
      "chat-smart": {},
      "chat-rp": {}
    }
  }
}
```

Notes:
- `auto` is the only model users NEED to select. Others are optional escape hatches for when the user wants to force a specific chain.
- `baseURL` uses `localhost:8000` because the SSH tunnel (`vps-gateway-tunnel.service`) forwards to `100.71.95.75:8000` on Tailscale. If running on a machine with direct Tailscale, use `http://100.71.95.75:8000/v1` directly.
- `apiKey` references the env var, never a literal. The key lives in `~/.env` as `VPS_GATEWAY_API_KEY`.
- `{file:~/.secrets/vps-gateway}` also works (age vault pattern) — both forms are valid per global AGENTS.md.

### VPS-155 specifics

VPS-155 has Tailscale up, so it can reach `100.71.95.75:8000` directly without the SSH tunnel. Config:

```json
{
  "vps-gateway": {
    "baseURL": "http://100.71.95.75:8000/v1",
    "apiKey": "{env:VPS_GATEWAY_API_KEY}",
    "models": { "auto": {} }
  }
}
```

### Laptop specifics

Laptop uses the SSH tunnel (`vps-gateway-tunnel.service`, systemd --user) to forward `localhost:8000` → `100.71.95.75:8000`. Config uses `http://localhost:8000/v1`. Tunnel auto-starts at login; if it dies, restart with `systemctl --user restart vps-gateway-tunnel.service`.

## Verification protocol (post-#197-merge)

Once PR #272 merges to `main` and the gateway is redeployed on VPS-40:

1. **Laptop smoke**:
   ```bash
   curl -s http://localhost:8000/v1/models | jq '.data[].id' | grep -E '"(auto|coding-elite|chat-fast)"'
   curl -s http://localhost:8000/v1/chat/completions \
     -H "Authorization: Bearer $VPS_GATEWAY_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"auto","messages":[{"role":"user","content":"def f(x): return x*2"}]}' \
     | jq '.choices[0].message.content'
   ```
   Expect: `auto` listed, coding prompt returns a coding-capable response (not roleplay).

2. **VPS-155 smoke** (PC unavailable scenario):
   ```bash
   ssh ubuntu@155.248.217.255
   curl -s http://100.71.95.75:8000/v1/models | jq '.data[].id' | grep auto
   curl -s http://100.71.95.75:8000/v1/chat/completions \
     -H "Authorization: Bearer $VPS_GATEWAY_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"auto","messages":[{"role":"user","content":"hi"}]}' \
     | jq '.choices[0].message.content'
   ```
   Expect: `auto` reachable, greeting returns a fast-chat response.

3. **Tool-call guard smoke**:
   ```bash
   curl -s http://localhost:8000/v1/chat/completions \
     -H "Authorization: Bearer $VPS_GATEWAY_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "model":"auto",
       "messages":[{"role":"user","content":"list files in this dir"}],
       "tools":[{"type":"function","function":{"name":"ls","parameters":{}}}]
     }' | jq '.choices[0].message.content'
   ```
   Expect: routed to `coding-fast` or `coding-elite` (tool-capable), NOT `chat-rp` or `chat-fast`.

4. **Check gateway logs** for the auto-route decision:
   ```bash
   ssh ubuntu@40.233.101.233 'journalctl -u llm-gateway -n 50 | grep -i "auto-route\|semantic_router"'
   ```
   Expect: `auto-route source=semantic intent=CODING_COMPLEX chain=coding-elite` style log lines.

## Sync flow

`~/CodingProjects/scripts/sync-opencode-config.sh --host ubuntu@155.248.217.255` merges the local `command.work` + `command.task` templates into VPS-155's `opencode.json`, preserving VPS-155 providers/models/permissions/MCPs/plugins. The gateway provider entry + `auto` model key should be added to the local template once, then synced.

To add `auto` to the sync template:
1. Edit local `~/.config/opencode/opencode.json` → `provider.vps-gateway.models` → add `"auto": {}`.
2. Run `~/CodingProjects/scripts/sync-opencode-config.sh --host ubuntu@155.248.217.255 --dry-run` to preview.
3. Run without `--dry-run` to push.
4. Restart opencode on VPS-155: `ssh ubuntu@155.248.217.255 'systemctl --user restart opencode'` (or however opencode runs there).

## Limitations (current)

- **`auto` not yet on `main`**: PR #272 open, awaiting review/merge. Once merged, redeploy gateway on VPS-40: `ssh ubuntu@40.233.101.233 'cd ~/LLM-API-Key-Proxy && git pull origin main && sudo systemctl restart llm-gateway'`.
- **`semantic-router[local]` not yet in gateway `requirements.txt`**: PR #272 adds it. Without it, gateway falls back to keyword `intent_detector.py` — still works, just less accurate on ambiguous prompts.
- **No per-machine telemetry yet**: gateway logs the auto-route decision, but opencode clients don't currently log which chain was picked. Future enhancement: have the gateway echo `X-Auto-Route-Chain` header.
- **No opencode skill for `auto`**: users still need to manually select `auto` in the opencode model picker. Future: add a skill that defaults to `auto` for all prompts.
