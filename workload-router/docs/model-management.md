# Model management compatibility

Date: 2026-11-03. No provider dispatch, VPS write, or deployment was performed.

The sidecar now serves the remaining safe model-management surface. Session
switches are process-local, bounded, and expiring; they are not a replacement
for the gateway's old in-memory map and are not shared across processes.

## Routes

| Route | Behaviour |
| --- | --- |
| `GET /v1/model-info/stats` | Ledger summary with `ready:false` and `upstream_model_info:false`; no upstream metadata catalog is bundled |
| `GET /v1/model/session/{session_id}` | Current alias for a session, defaulting to `auto-chat` |
| `POST /v1/model/switch` | Switch one session to an accepted sidecar alias |
| `POST /v1/model/detect-intent` | Deterministic regex recommendation from bounded message text |
| `GET /v1/model/list` | 22 aliases plus generated descriptions |
| `GET /v1/model/status` | Default model and accepted aliases |
| `POST /v1/model/parse-command` | Parse `/model`, `/use`, and shortcut commands without switching |

All routes require the normal sidecar Bearer token.

### Session switch

```json
{"model": "coding-elite", "session_id": "opencode-1"}
```

Response:

```json
{
  "success": true,
  "session_id": "opencode-1",
  "previous_model": "auto-chat",
  "current_model": "coding-elite",
  "message": "Model switched to coding-elite"
}
```

- Default session ID: `"default"`.
- Maximum: 1,024 sessions.
- Expiry: 24 hours after the last switch.
- Eviction: oldest updated session.
- Session IDs: 1–512 characters, no control characters.
- Model names are lowercased and must be in the 22-alias set.
- State resets on restart and is not shared between sidecar workers or hosts.

### Intent detection

`POST /v1/model/detect-intent` accepts:

```json
{
  "messages": [{"role": "user", "content": "implement a FastAPI endpoint"}],
  "current_model": "auto-chat"
}
```

It returns `intent`, `confidence`, `suggested_model`, `detected_keywords`, and
`reasoning`. The implementation is the legacy regex/keyword heuristic, bounded
to 128 messages and 32 KiB of combined text. It is **not** a semantic classifier
and it never dispatches a model call.

Suggestions stay inside the sidecar's aliases:

| Detected category | Suggested alias |
| --- | --- |
| Roleplay / uncensored | `chat-rp` |
| Complex coding | `coding-elite` |
| Fast coding | `coding-fast` |
| Complex reasoning | `chat-smart` |
| Simple chat | `chat-fast` |

If nothing matches, the current model is returned when supplied; otherwise the
result is `chat-fast`. The old `uncensored-chat` target is unavailable, so
`chat-rp` is the closest accepted alias.

### Model-info stats

The old gateway returned stats from its upstream model-info registry. The
sidecar has no such registry and does not pretend otherwise:

```json
{
  "ready": false,
  "service": "not_configured",
  "source": "sidecar_ledger",
  "openrouter_count": 0,
  "modelsdev_count": 0,
  "cached_lookups": 0,
  "index_entries": 0,
  "refresh_interval": null,
  "alias_count": 22,
  "configured_route_count": 2,
  "upstream_model_info": false
}
```

`configured_route_count` is read from the ledger. `ready:false` and
`upstream_model_info:false` are intentional; `/v1/models/{model_id}` still returns
the basic alias envelope rather than upstream metadata.

## Limitations

- Session state is local and volatile; multiple sidecar workers would not share
  switches. The current server is one process.
- Intent detection is keyword-based and can misclassify mixed prompts.
- No model-info catalog, pricing enrichment, or upstream model discovery is
  added by `/v1/model-info/stats`.
- CLIProxyAPI routes remain absent; they require a configured external service.
