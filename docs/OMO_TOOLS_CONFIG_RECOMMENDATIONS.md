# Opencode/Omo Tools, Skills, MCP & Config Recommendations
Generated: 2026-03-28

## Current MCP Servers (from ~/.config/opencode/opencode.json)

| MCP Server | Tool | Status | Notes |
|---|---|---|---|
| `ripgrep` | mcp-ripgrep | ENABLED | Fast file search. Essential. |
| `tavily` | tavily-mcp | ENABLED | Best web search for factual queries. |
| `brave-search` | @anthropic-ai/brave-search-mcp | ENABLED | Good fallback web search. |
| `searxng` | mcp-searxng | ENABLED | Self-hosted, private. Railway instance. |
| `searxng-fallback` | mcp-searxng | ENABLED | Second SearXNG instance (workers.dev). |
| `exa` | exa-mcp-server | ENABLED | Best for code/dev search. |

**Assessment**: Web search is well covered with 3 independent providers + 2 SearXNG instances.
No gaps here.

---

## Missing MCP Servers (High Value)

### 1. `mcp-filesystem` — Direct file system operations
- **Why**: Allows read/write/list without shell commands. Faster for simple file ops.
- **Install**: `npx -y @modelcontextprotocol/server-filesystem`
- **Config**:
```json
"filesystem": {
  "type": "local",
  "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/home/osees/CodingProjects"],
  "enabled": true
}
```
- **Priority**: Medium. Opencode already has native file tools, but MCP filesystem is faster for
  bulk directory ops.

### 2. `mcp-github` — GitHub API access
- **Why**: Read PRs, issues, file contents from other repos without cloning. Useful for
  ZeroClaw monitoring (check run status, PR list, workflow logs).
- **Install**: `npx -y @modelcontextprotocol/server-github`
- **Config**:
```json
"github": {
  "type": "local",
  "command": ["npx", "-y", "@modelcontextprotocol/server-github"],
  "enabled": true,
  "environment": {
    "GITHUB_PERSONAL_ACCESS_TOKEN": "<your GH_TOKEN>"
  }
}
```
- **Priority**: High. Especially useful for monitoring ZeroClaw runs from opencode.

### 3. `mcp-sqlite` — Direct SQLite queries
- **Why**: Your telemetry DB and provider_status.db are SQLite. Direct SQL queries from
  the agent are faster than writing Python scripts.
- **Install**: `npx -y @modelcontextprotocol/server-sqlite`
- **Config**:
```json
"sqlite-telemetry": {
  "type": "local",
  "command": ["npx", "-y", "@modelcontextprotocol/server-sqlite", "/tmp/llm_proxy_telemetry.db"],
  "enabled": true
}
```
- **Priority**: High for Phase 3 (telemetry wiring). Lets agent query live TPS data directly.

### 4. `context7` — Official library documentation
- **Why**: Fetches current, accurate docs for any npm/pip library. Prevents hallucinated API usage.
  Already available in this Cursor/Sisyphus session as a tool — add to opencode config too.
- **Install**: `npx -y @context7/mcp-server`
- **Config**:
```json
"context7": {
  "type": "local",
  "command": ["npx", "-y", "@context7/mcp-server"],
  "enabled": true
}
```
- **Priority**: Medium-High. Saves time when working with unfamiliar APIs.

### 5. `mcp-sequentialthinking` — Structured reasoning
- **Why**: Forces step-by-step thinking for complex architectural decisions. Reduces hallucinated
  solutions on hard problems.
- **Install**: `npx -y @modelcontextprotocol/server-sequential-thinking`
- **Priority**: Low-Medium. More useful for complex planning tasks than fast coding.

---

## Current Provider Stack Assessment

### Primary: `google/antigravity-gemini-3.1-pro`
- **Status**: GOOD. Best free model for coding tasks. 1M context.
- **Variants**: low/high thinking available. Use `high` for complex tasks.
- **Concern**: If antigravity quota runs out, fallback is supacoder.

### Fallback chain in omo-config-template.json
Current chain (35+ models) is comprehensive but has some notes:
- `cursor-proxy/*` models require local prochatxy proxy running on port 4666 (laptop only,
  not available on VPS2). These will fail on VPS2 GH Actions runners.
- `opencode/minimax-m2.5-free` — good free model, keep near top
- `aihubmix/coding-glm-5-free` — good for coding, keep
- `nvidia/glm-5-free` — good backup

**Recommendation**: Split omo-config-template.json into two versions:
- `omo-config-local.json` — includes cursor-proxy models (for laptop use)
- `omo-config-vps.json` — excludes cursor-proxy models (for VPS2/GH Actions)

---

## Opencode Agent Role Configs

Current agents defined: `build`, `sisyphus`, `ultrawork` (identical configs).

**Recommendations**:

1. **Add a `fast` agent role** using `coding-fast` virtual model for:
   - Quick file edits, linting fixes, simple refactors
   - Reduces cost/quota on trivial tasks

2. **Add a `research` agent role** using `chat-smart` virtual model for:
   - Web search + synthesis tasks
   - Architecture research before implementation

3. **Current agents are identical** — consider differentiating:
   - `build`: coding-smart (balanced quality+speed for implementation)
   - `ultrawork`: coding-elite (best quality for complex problems)
   - `sisyphus`: coding-smart (orchestration, doesn't need elite)

---

## Speed Optimizations for Fast Task Completion

1. **Enable streaming by default** — reduces perceived latency even if total tokens same
2. **Reduce `max_fallback_attempts` from 15 to 8** — 15 attempts adds latency on failure chains
3. **Set `runtime_fallback.cooldown_seconds: 30`** (current is 60) for coding-fast category
4. **Use `coding-fast` virtual model** for sub-tasks that are: file creation, simple edits,
   running tests, checking syntax — NOT for actual coding logic
5. **ripgrep MCP** is already enabled — good, much faster than grep for large codebases

---

## What NOT to Add

| Tool | Reason to Skip |
|---|---|
| `mcp-playwright` | Already available as opencode skill. Don't double-add. |
| `mcp-puppeteer` | Playwright is better. Skip. |
| Any paid MCP server | Against free-only constraint. |
| `mcp-memory` | Vector memory adds latency. Not needed for coding tasks. |

---

## Priority Action List

1. Add `mcp-github` to opencode.json — high value for ZeroClaw monitoring
2. Add `mcp-sqlite` to opencode.json — needed for telemetry queries (Phase 3)
3. Split omo-config-template.json into local vs VPS variants
4. Add `fast` agent role using `coding-fast` virtual model
5. Reduce `max_fallback_attempts` to 8 in omo-config templates
6. Add `context7` MCP for library docs
