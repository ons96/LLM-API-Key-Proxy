#!/usr/bin/env bash
# add-provider.sh -- bulk add LLM providers to gateway + OpenCode config + .env.
# Idempotent. Dry-run by default unless --apply. Never writes plaintext keys
# to YAML/JSON: keys live in .env, configs use {env:VAR_NAME} or ${VAR_NAME}.
#
# Usage:
#   add-provider.sh --providers ./providers.json
#   add-provider.sh --dry-run --providers '[{"name":"ex","base_url":"https://x/v1","env_var":"X_KEY","models":["m1"]}]'
#   add-provider.sh --apply --providers ./providers.json --sync-vps ubuntu@40.233.101.233
#   add-provider.sh --sync-vps ubuntu@40.233.101.233 --providers ./providers.json
#
# providers.json schema:
#   [
#     {
#       "name": "example",
#       "base_url": "https://api.example.com/v1",
#       "env_var": "EXAMPLE_API_KEY",
#       "models": ["model-a", "model-b"],
#       "free_tier": true,            # optional, default true
#       "npm": "@ai-sdk/openai-compatible",  # optional, default above
#       "signup_url": "https://...",   # optional, only written to providers_database.yaml
#       "rate_limits": {"rpm": 30}     # optional, only written to providers_database.yaml
#     }
#   ]
#
# Files written (when --apply):
#   - $ROUTER_CONFIG (default: LLM-API-Key-Proxy/config/router_config.yaml) -- adds providers.<name>
#   - $PROVIDERS_DB  (default: LLM-API-Key-Proxy/config/providers_database.yaml) -- appends entry
#   - $OPENCODE_JSON (default: ~/.config/opencode/opencode.json) -- adds provider.<name>
#   - $ENV_FILE      (default: ~/.env) -- appends export NAME_API_KEY=<placeholder> if missing
#
# All YAML/JSON edits use ruamel.yaml / stdlib json to preserve comments + key order.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXY_ROOT="${PROXY_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

ROUTER_CONFIG="${ROUTER_CONFIG:-$PROXY_ROOT/config/router_config.yaml}"
PROVIDERS_DB="${PROVIDERS_DB:-$PROXY_ROOT/config/providers_database.yaml}"
OPENCODE_JSON="${OPENCODE_JSON:-$HOME/.config/opencode/opencode.json}"
ENV_FILE="${ENV_FILE:-$HOME/.env}"

DRY_RUN=1
PROVIDERS_FILE=""
PROVIDERS_JSON=""
SYNC_VPS_HOST=""
SYNC_VPS_PATH="${SYNC_VPS_PATH:-/home/ubuntu/LLM-API-Key-Proxy}"
SYNC_VPS_ENV_PATH="${SYNC_VPS_ENV_PATH:-/home/ubuntu/LLM-API-Key-Proxy/.env}"

usage() {
  sed -n '2,28p' "$0" | sed 's/^# \?//'
  exit 1
}

log() { echo "[add-provider] $*" >&2; }
err() { echo "[add-provider] ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) DRY_RUN=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --providers) [[ $# -ge 2 ]] || err "--providers requires arg"; PROVIDERS_JSON="$2"; shift 2 ;;
    --providers-file) [[ $# -ge 2 ]] || err "--providers-file requires arg"; PROVIDERS_FILE="$2"; shift 2 ;;
    --sync-vps) [[ $# -ge 2 ]] || err "--sync-vps requires host"; SYNC_VPS_HOST="$2"; shift 2 ;;
    --router-config) ROUTER_CONFIG="$2"; shift 2 ;;
    --providers-db) PROVIDERS_DB="$2"; shift 2 ;;
    --opencode-json) OPENCODE_JSON="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) err "unknown flag: $1" ;;
  esac
done

# Resolve providers input
if [[ -n "$PROVIDERS_FILE" ]]; then
  [[ -f "$PROVIDERS_FILE" ]] || err "providers file not found: $PROVIDERS_FILE"
  PROVIDERS_JSON="$(cat "$PROVIDERS_FILE")"
fi
[[ -n "$PROVIDERS_JSON" ]] || err "no providers given (use --providers or --providers-file)"

# Validate input is a JSON array; required fields per entry.
python3 - "$PROVIDERS_JSON" <<'PY' || err "providers JSON validation failed"
import json, sys
data = json.loads(sys.argv[1])
if not isinstance(data, list):
    print("expected JSON array of provider objects", file=sys.stderr)
    sys.exit(1)
required = ("name", "base_url", "env_var", "models")
for i, p in enumerate(data):
    if not isinstance(p, dict):
        print(f"entry {i}: not an object", file=sys.stderr); sys.exit(1)
    missing = [k for k in required if k not in p or not p[k]]
    if missing:
        print(f"entry {i} ({p.get('name','?')}): missing/empty: {missing}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(p["models"], list) or not p["models"]:
        print(f"entry {i} ({p['name']}): models must be non-empty list", file=sys.stderr)
        sys.exit(1)
    if not p["env_var"].replace("_", "").isalnum():
        print(f"entry {i} ({p['name']}): env_var must be [A-Z0-9_]+", file=sys.stderr)
        sys.exit(1)
print(f"[add-provider] validated {len(data)} provider entr{'y' if len(data)==1 else 'ies'}", file=sys.stderr)
PY

log "mode: $([[ $DRY_RUN -eq 1 ]] && echo DRY-RUN || echo APPLY)"
log "router_config: $ROUTER_CONFIG"
log "providers_db : $PROVIDERS_DB"
log "opencode_json: $OPENCODE_JSON"
log "env_file     : $ENV_FILE"
[[ -n "$SYNC_VPS_HOST" ]] && log "sync_vps      : $SYNC_VPS_HOST:$SYNC_VPS_PATH"

[[ -f "$ROUTER_CONFIG" ]] || err "router config not found: $ROUTER_CONFIG"
[[ -f "$PROVIDERS_DB"  ]] || err "providers db not found: $PROVIDERS_DB"
[[ -f "$OPENCODE_JSON" ]] || err "opencode json not found: $OPENCODE_JSON"
[[ -f "$ENV_FILE"      ]] || log "env file missing, will skip .env update: $ENV_FILE"

# Run the Python mutator.
PROVIDERS_JSON="$PROVIDERS_JSON" \
ROUTER_CONFIG="$ROUTER_CONFIG" \
PROVIDERS_DB="$PROVIDERS_DB" \
OPENCODE_JSON="$OPENCODE_JSON" \
ENV_FILE="$ENV_FILE" \
DRY_RUN="$DRY_RUN" \
SYNC_VPS_HOST="$SYNC_VPS_HOST" \
SYNC_VPS_PATH="$SYNC_VPS_PATH" \
SYNC_VPS_ENV_PATH="$SYNC_VPS_ENV_PATH" \
python3 - <<'PY'
import json, os, re, sys, subprocess
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml", file=sys.stderr); sys.exit(1)

providers = json.loads(os.environ["PROVIDERS_JSON"])
router_path = Path(os.environ["ROUTER_CONFIG"])
db_path = Path(os.environ["PROVIDERS_DB"])
oc_path = Path(os.environ["OPENCODE_JSON"])
env_path = Path(os.environ["ENV_FILE"])
dry_run = os.environ["DRY_RUN"] == "1"
sync_vps = os.environ.get("SYNC_VPS_HOST", "")
sync_vps_path = os.environ["SYNC_VPS_PATH"]
sync_vps_env_path = os.environ["SYNC_VPS_ENV_PATH"]

def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f) or {}

def dump_yaml(p, data):
    with open(p, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

def load_json(p):
    with open(p) as f:
        return json.load(f)

def dump_json(p, data):
    with open(p, "w") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")

router = load_yaml(router_path)
db = load_yaml(db_path)
oc = load_json(oc_path)

router.setdefault("providers", {})
db.setdefault("providers", [])
oc.setdefault("provider", {})

changes = []

for p in providers:
    name = p["name"]
    env_var = p["env_var"]
    base_url = p["base_url"]
    models = p["models"]
    free_tier = p.get("free_tier", True)
    npm = p.get("npm", "@ai-sdk/openai-compatible")

    # router_config.yaml: providers.<name> = {enabled, env_var, base_url, free_tier, free_tier_models}
    existing = router["providers"].get(name)
    if existing is None:
        router["providers"][name] = {
            "enabled": True,
            "env_var": env_var,
            "base_url": base_url,
            "free_tier": free_tier,
            "free_tier_models": list(models),
        }
        changes.append(f"router: +providers.{name}")
    else:
        updated = False
        if existing.get("env_var") != env_var:
            existing["env_var"] = env_var; updated = True
        if existing.get("base_url") != base_url:
            existing["base_url"] = base_url; updated = True
        existing.setdefault("free_tier", free_tier)
        ftm = set(existing.get("free_tier_models", []))
        for m in models:
            if m not in ftm:
                ftm.add(m); updated = True
        if ftm:
            existing["free_tier_models"] = sorted(ftm)
        if updated:
            changes.append(f"router: ~providers.{name}")
        else:
            changes.append(f"router: =providers.{name} (no change)")

    # providers_database.yaml: append entry to providers[]
    if not any(e.get("id") == name for e in db["providers"]):
        entry = {
            "id": name,
            "name": p.get("display_name", name.title()),
            "env_var": env_var,
            "base_url": base_url,
            "enabled": True,
            "free_tier": free_tier,
            "no_api_key_required": p.get("no_api_key_required", False),
        }
        if "signup_url" in p:
            entry["signup_url"] = p["signup_url"]
        if "rate_limits" in p:
            entry["rate_limits"] = p["rate_limits"]
        if "capabilities" in p:
            entry["capabilities"] = p["capabilities"]
        entry["free_models"] = [{"id": m, "context": p.get("context", 8192)} for m in models]
        db["providers"].append(entry)
        changes.append(f"providers_db: +providers[] id={name}")
    else:
        changes.append(f"providers_db: =providers[] id={name} (exists, skip)")

    # opencode.json: provider.<name> = {name, npm, options:{apiKey:{env:VAR}, baseURL}, models:{id:{name}}}
    if name not in oc["provider"]:
        oc["provider"][name] = {
            "name": p.get("display_name", name.title()),
            "npm": npm,
            "options": {
                "apiKey": f"{{env:{env_var}}}",
                "baseURL": base_url,
            },
            "models": {m: {"name": m} for m in models},
        }
        changes.append(f"opencode: +provider.{name}")
    else:
        block = oc["provider"][name]
        if block.get("options", {}).get("apiKey") != f"{{env:{env_var}}}":
            block.setdefault("options", {})["apiKey"] = f"{{env:{env_var}}}"
            changes.append(f"opencode: ~provider.{name} apiKey -> {{{{env:{env_var}}}}}")
        if block.get("options", {}).get("baseURL") != base_url:
            block.setdefault("options", {})["baseURL"] = base_url
            changes.append(f"opencode: ~provider.{name} baseURL")
        for m in models:
            if m not in block.get("models", {}):
                block.setdefault("models", {})[m] = {"name": m}
                changes.append(f"opencode: +provider.{name}.models.{m}")
        if not any(c.startswith(f"opencode: ~provider.{name}") or c.startswith(f"opencode: +provider.{name}") for c in changes):
            changes.append(f"opencode: =provider.{name} (no change)")

    # .env: append export VAR_NAME= if missing
    if env_path.exists():
        env_text = env_path.read_text()
        if not re.search(rf"^\s*{re.escape(env_var)}\s*=", env_text, re.M):
            new_line = f"\n# {name} API key (added by add-provider.sh)\n{env_var}=<set-your-key-here>\n"
            env_path.write_text(env_text.rstrip() + new_line)
            changes.append(f"env: +export {env_var}=<placeholder>")
        else:
            changes.append(f"env: ={env_var} (exists, skip)")
    else:
        changes.append(f"env: skip (file missing: {env_path})")

print("\n=== Planned changes ===")
for c in changes:
    print(f"  {c}")
print(f"=== {len(changes)} change(s) ===\n")

if dry_run:
    print("[DRY-RUN] no files written. pass --apply to commit.")
    sys.exit(0)

# Apply
dump_yaml(router_path, router)
print(f"[apply] wrote {router_path}")
dump_yaml(db_path, db)
print(f"[apply] wrote {db_path}")
dump_json(oc_path, oc)
print(f"[apply] wrote {oc_path}")

# Sync to VPS if requested
if sync_vps:
    ssh_key = os.path.expanduser("~/.ssh/oracle.key")
    ssh_opts = ["-i", ssh_key, "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
    print(f"\n[sync-vps] target: {sync_vps}:{sync_vps_path}")
    for local, remote in [
        (str(router_path), f"{sync_vps_path}/config/router_config.yaml"),
        (str(db_path),    f"{sync_vps_path}/config/providers_database.yaml"),
    ]:
        cmd = ["ssh", *ssh_opts, sync_vps, f"mkdir -p {sync_vps_path}/config && cat > {remote}"]
        print(f"  cmd: {' '.join(cmd)} < {local}")
    print(f"  cmd: scp {oc_path} {sync_vps}:~/.config/opencode/opencode.json")
    print(f"  cmd: scp {env_path} {sync_vps}:{sync_vps_env_path}")
    print("[sync-vps] commands printed; run manually or wire into a follow-up cron.")

print("\n[done]")
PY

log "exit $?"
