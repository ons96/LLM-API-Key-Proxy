#!/usr/bin/env python3
"""
Active provider performance probe (#332).

Sends a canonical short coding prompt to each enabled LLM provider+model
declared in config/router_config.yaml, measures TTFT + TPS + latency via
streaming, and writes results to the telemetry DB (api_calls table) via
TelemetryManager.record_call().

The /v1/tps-stats endpoint reads api_calls automatically -- no new endpoint
needed. Probe data flows through end-to-end: probe -> DB -> endpoint.

Usage:
    python scripts/probe_providers.py                    # probe all enabled providers
    python scripts/probe_providers.py --provider groq    # probe one provider
    python scripts/probe_providers.py --dry-run          # list targets, no calls
    python scripts/probe_providers.py --max-models 3     # cap models per provider
    python scripts/probe_providers.py --interval 5       # seconds between probes
    python scripts/probe_providers.py --status-file /tmp/probe-status.yaml

Each provider gets one endpoint check before any model is probed. A failed
endpoint skips all of that provider's models and is tagged in
config/provider_probe_status.yaml. The tag is advisory: every later run retries
the endpoint and clears the tag when the provider recovers. The router's static
config/dead_providers.yaml blocklist is not modified by this script.
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml
from openai import OpenAI

# ponytail: load .env so standalone probe sees provider API keys (gateway loads
# it at startup via dotenv; standalone scripts do not). Graceful no-op if missing.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Make src importable when run from repo root. TelemetryManager is imported
# lazily inside run() so this module can be imported without litellm (which
# rotator_library/__init__ pulls in transitively) -- keeps unit tests hermetic.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("probe_providers")

# Search-only / non-LLM providers -- not probe targets.
SEARCH_ONLY = {"brave_search", "tavily", "duckduckgo", "exa", "jina"}

# Canonical short coding prompt (~10 input tokens, ~30-50 output).
CANONICAL_PROMPT = (
    "Write a Python function to reverse a string. Reply with code only."
)
MAX_OUTPUT_TOKENS = 64
REQUEST_TIMEOUT_S = 30
ENDPOINT_TIMEOUT_S = 5.0
DEFAULT_STATUS_PATH = REPO_ROOT / "config" / "provider_probe_status.yaml"
DEFAULT_PROVIDERS_DB = REPO_ROOT / "config" / "providers_database.yaml"
# Cloudflare-fronted providers such as crax reject Python/default user-agents
# with Error 1010. A normal browser UA is required for both health and chat.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)

# Env var resolution helper for ${VAR} and ${VAR:-default} in YAML values.
def _resolve_env(value: str) -> str:
    if not isinstance(value, str) or "${" not in value:
        return value
    # ${VAR:-default}
    idx = value.find("${")
    end = value.find("}", idx)
    if end == -1:
        return value
    expr = value[idx + 2 : end]
    if ":-" in expr:
        var, default = expr.split(":-", 1)
        resolved = os.environ.get(var.strip(), default)
    else:
        resolved = os.environ.get(expr.strip(), "")
    return value[:idx] + resolved + value[end + 1 :]


def check_endpoint(
    base_url: str,
    api_key: str = "",
    timeout: float = ENDPOINT_TIMEOUT_S,
) -> Tuple[bool, str]:
    """Return (healthy, detail) for one provider endpoint.

    The probe checks <base_url>/models, not the API root, because some APIs
    return 403 on their root even with a valid key. Any HTTP response proves
    the host is reachable. The provider's existing key is sent as a bearer
    header so a 401/403 means a real auth problem rather than an
    unauthenticated page. Rate-limit, timeout, and 5xx responses are marked
    unhealthy because model probes would fail or waste quota. A 404/405 on
    /models is reachable so custom providers without that route are not
    falsely buried.
    """
    root = str(base_url or "").strip().rstrip("/")
    if not root:
        return False, "missing base_url"
    url = root if root.endswith("/models") else f"{root}/models"
    headers = {"User-Agent": BROWSER_USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            response.read(1)
            return True, f"HTTP {response.status}"
    except HTTPError as exc:
        code = exc.code
        if code in (401, 403):
            return False, f"HTTP {code} auth_error"
        if code == 408:
            return False, f"HTTP {code} timeout"
        if code == 429:
            return False, f"HTTP {code} rate_limited"
        if code >= 500:
            return False, f"HTTP {code} server_error"
        return True, f"HTTP {code}"
    except (URLError, TimeoutError, socket.timeout) as exc:
        reason = getattr(exc, "reason", exc)
        return False, f"connection_error: {reason}"[:200]
    except ValueError as exc:
        return False, f"invalid_url: {exc}"[:200]


def load_one_time_providers(path: Path = DEFAULT_PROVIDERS_DB) -> set[str]:
    """Return provider ids whose free quota is one-time, not recurring."""
    if not path.exists():
        return set()
    try:
        with open(path) as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        log.warning("could not read provider DB %s: %s; one-time skip disabled", path, exc)
        return set()
    one_time: set[str] = set()
    providers = data.get("providers") if isinstance(data, dict) else None
    if not isinstance(providers, list):
        return one_time
    for provider in providers:
        if not isinstance(provider, dict) or not provider.get("id"):
            continue
        quota = provider.get("free_quota") or {}
        free_type = str(quota.get("free_type", "")).strip().lower()
        if free_type.startswith("one_time") or provider.get("free_one_time") is True:
            one_time.add(str(provider["id"]))
    return one_time


def load_probe_status(path: Path) -> Dict[str, Any]:
    """Load the advisory provider probe status, or start a fresh document."""
    if not path.exists():
        return {"version": 1, "updated": None, "providers": {}}
    try:
        with open(path) as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        log.warning("could not read probe status %s: %s; starting fresh", path, exc)
        return {"version": 1, "updated": None, "providers": {}}
    if not isinstance(data, dict):
        log.warning("probe status %s is not a mapping; starting fresh", path)
        return {"version": 1, "updated": None, "providers": {}}
    data.setdefault("version", 1)
    data.setdefault("providers", {})
    if not isinstance(data["providers"], dict):
        data["providers"] = {}
    return data


def save_probe_status(path: Path, status: Dict[str, Any]) -> None:
    """Atomically write the advisory status file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    status["providers"] = dict(sorted((status.get("providers") or {}).items()))
    rendered = yaml.safe_dump(status, sort_keys=False, allow_unicode=False)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(rendered, encoding="ascii")
    temporary.replace(path)


def check_targets(
    targets: List[Tuple[str, str, str, str]],
    status: Dict[str, Any],
    checker: Callable[[str, str], Tuple[bool, str]] = check_endpoint,
    now: Optional[str] = None,
) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, int]]:
    """Check each provider once; drop every model target when it is unhealthy."""
    checked_at = now or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    providers = status.setdefault("providers", {})
    by_provider: Dict[str, List[Tuple[str, str, str, str]]] = {}
    for target in targets:
        by_provider.setdefault(target[0], []).append(target)

    healthy: List[Tuple[str, str, str, str]] = []
    alive = 0
    dead = 0
    skipped_models = 0
    for pid, provider_targets in sorted(by_provider.items()):
        ok, detail = checker(provider_targets[0][2], provider_targets[0][3])
        entry = providers.setdefault(pid, {})
        was_dead = entry.get("status") == "dead"
        entry["status"] = "alive" if ok else "dead"
        entry["checked_at"] = checked_at
        if ok:
            alive += 1
            entry["last_alive_at"] = checked_at
            entry.pop("reason", None)
            entry.pop("dead_since", None)
            healthy.extend(provider_targets)
            log.info("endpoint OK %s: %s", pid, detail)
        else:
            dead += 1
            skipped_models += len(provider_targets)
            if not was_dead or not entry.get("dead_since"):
                entry["dead_since"] = checked_at
            entry["last_dead_at"] = checked_at
            entry["reason"] = detail
            log.warning("endpoint DEAD %s: %s; skipping %d models", pid, detail, len(provider_targets))

    status["updated"] = checked_at
    return healthy, {
        "providers": len(by_provider),
        "alive": alive,
        "dead": dead,
        "skipped_models": skipped_models,
    }


def load_provider_config(config_path: Path) -> Dict[str, Any]:
    """Load router_config.yaml, return providers dict."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("providers", {})


def select_targets(
    providers: Dict[str, Any],
    only: Optional[str] = None,
    max_models: Optional[int] = None,
    skip_providers: Optional[set[str]] = None,
) -> List[Tuple[str, str, str, str]]:
    """Build list of (provider_id, model, base_url, api_key) targets.

    Skips: disabled, dead, search-only, one-time-credit providers, and providers
    without models.
    """
    targets: List[Tuple[str, str, str, str]] = []
    skip_providers = skip_providers or set()
    for pid, pdata in providers.items():
        if pid in SEARCH_ONLY:
            continue
        if not pdata.get("enabled", False):
            continue
        if pdata.get("_dead"):
            continue
        if pid in skip_providers:
            log.info("skip %s: one-time-credit provider", pid)
            continue
        if only and pid != only:
            continue
        env_var = pdata.get("env_var")
        base_url = pdata.get("base_url") or pdata.get("api_base")
        if not base_url:
            continue
        base_url = _resolve_env(base_url)
        api_key = ""
        if env_var:
            api_key = os.environ.get(env_var, "")
        if not api_key and not pdata.get("no_api_key_required"):
            log.debug("skip %s: no api key (%s)", pid, env_var)
            continue
        models = pdata.get("free_tier_models") or []
        # Filter out auto/free meta-models that aren't real endpoints.
        models = [
            m
            for m in models
            if m
            and "auto" not in m.lower().split("/")[-1]
            and not m.endswith(":free")
        ]
        if max_models:
            models = models[:max_models]
        for model in models:
            targets.append((pid, model, base_url, api_key))
    return targets


def probe_one(
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
) -> Dict[str, Any]:
    """Send canonical prompt via streaming, measure TTFT + TPS + latency.

    Returns dict suitable for TelemetryManager.record_call kwargs.
    """
    client = OpenAI(
        api_key=api_key or "dummy",
        base_url=base_url,
        timeout=REQUEST_TIMEOUT_S,
        max_retries=0,
        default_headers={"User-Agent": BROWSER_USER_AGENT},
    )
    start = time.perf_counter()
    ttft_ms: Optional[int] = None
    output_tokens = 0
    first_chunk_time: Optional[float] = None
    last_chunk_time: Optional[float] = None
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": CANONICAL_PROMPT}],
            max_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            now = time.perf_counter()
            if first_chunk_time is None:
                first_chunk_time = now
                ttft_ms = int((now - start) * 1000)
            last_chunk_time = now
            # Count content tokens (rough: non-empty delta = 1 token).
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and getattr(delta, "content", None):
                    output_tokens += 1
            if chunk.usage:
                output_tokens = max(output_tokens, chunk.usage.completion_tokens or 0)
        total_ms = int((time.perf_counter() - start) * 1000)
        # TPS = tokens / stream_duration_seconds
        stream_dur = (
            (last_chunk_time - first_chunk_time) if last_chunk_time and first_chunk_time else 0
        )
        tps: Optional[float] = None
        if stream_dur > 0 and output_tokens > 0:
            tps = round(output_tokens / stream_dur, 2)
        return {
            "provider": provider,
            "model": model,
            "success": True,
            "response_time_ms": total_ms,
            "time_to_first_token_ms": ttft_ms,
            "tokens_per_second": tps,
            "input_tokens": 10,
            "output_tokens": output_tokens,
            "error_reason": None,
        }
    except Exception as exc:
        total_ms = int((time.perf_counter() - start) * 1000)
        err_str = str(exc)[:200]
        err_type = type(exc).__name__
        # Classify common error reasons for DB filtering.
        reason = err_type
        if "429" in err_str or "rate_limit" in err_str.lower():
            reason = "rate_limited"
        elif "401" in err_str or "auth" in err_str.lower():
            reason = "auth_error"
        elif "404" in err_str:
            reason = "model_not_found"
        elif "timeout" in err_str.lower():
            reason = "timeout"
        elif "5" in err_str[:1] and err_str[:3].isdigit():
            reason = "server_error"
        return {
            "provider": provider,
            "model": model,
            "success": False,
            "response_time_ms": total_ms,
            "time_to_first_token_ms": None,
            "tokens_per_second": None,
            "input_tokens": None,
            "output_tokens": None,
            "error_reason": reason,
        }


def run(
    config_path: Path,
    db_path: str,
    only: Optional[str] = None,
    max_models: Optional[int] = None,
    interval_s: float = 3.0,
    dry_run: bool = False,
    status_path: Path = DEFAULT_STATUS_PATH,
    endpoint_timeout: float = ENDPOINT_TIMEOUT_S,
    provider_db_path: Path = DEFAULT_PROVIDERS_DB,
) -> int:
    providers = load_provider_config(config_path)
    one_time = load_one_time_providers(provider_db_path)
    if one_time:
        log.info("skipping one-time-credit providers: %s", ", ".join(sorted(one_time)))
    targets = select_targets(
        providers,
        only=only,
        max_models=max_models,
        skip_providers=one_time,
    )
    log.info("selected %d probe targets (%d providers)", len(targets), len({t[0] for t in targets}))
    if dry_run:
        for pid, model, base_url, _ in targets:
            print(f"{pid}\t{model}\t{base_url}")
        return 0
    status = load_probe_status(status_path)
    targets, health = check_targets(
        targets,
        status,
        checker=lambda url, key: check_endpoint(url, api_key=key, timeout=endpoint_timeout),
    )
    save_probe_status(status_path, status)
    log.info(
        "endpoint checks: %d alive, %d dead, %d model probes skipped",
        health["alive"],
        health["dead"],
        health["skipped_models"],
    )
    if not targets:
        log.warning("no healthy provider targets left")
        return 1
    # Import telemetry.py directly via importlib to avoid rotator_library/__init__
    # which imports client.py -> litellm (not always installed in run env).
    import importlib.util
    _tpath = REPO_ROOT / "src" / "rotator_library" / "telemetry.py"
    _spec = importlib.util.spec_from_file_location("_probe_tm", _tpath)
    _tmod = importlib.util.module_from_spec(_spec)
    sys.modules["_probe_tm"] = _tmod
    _spec.loader.exec_module(_tmod)
    TelemetryManager = _tmod.TelemetryManager
    tm = TelemetryManager(db_path=db_path)
    ok = 0
    fail = 0
    for i, (pid, model, base_url, api_key) in enumerate(targets):
        log.info("[%d/%d] %s / %s", i + 1, len(targets), pid, model)
        result = probe_one(pid, model, base_url, api_key)
        tm.record_call(
            provider=result["provider"],
            model=result["model"],
            success=result["success"],
            response_time_ms=result["response_time_ms"],
            error_reason=result["error_reason"],
            time_to_first_token_ms=result["time_to_first_token_ms"],
            tokens_per_second=result["tokens_per_second"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
        )
        if result["success"]:
            ok += 1
            log.info(
                "  OK  TTFT=%dms TPS=%s tok=%d total=%dms",
                result["time_to_first_token_ms"] or 0,
                result["tokens_per_second"],
                result["output_tokens"],
                result["response_time_ms"],
            )
        else:
            fail += 1
            log.info("  FAIL %s (%dms)", result["error_reason"], result["response_time_ms"])
        if i + 1 < len(targets):
            time.sleep(interval_s)
    log.info("done: %d ok, %d fail", ok, fail)
    return 0 if ok > 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Active provider performance probe (#332)")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "router_config.yaml"),
        help="path to router_config.yaml",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("TELEMETRY_DB_PATH", "/tmp/llm_proxy_telemetry.db"),
        help="telemetry DB path",
    )
    parser.add_argument("--provider", help="probe only this provider id")
    parser.add_argument("--max-models", type=int, help="cap models per provider")
    parser.add_argument("--interval", type=float, default=3.0, help="seconds between probes")
    parser.add_argument(
        "--status-file",
        default=str(DEFAULT_STATUS_PATH),
        help="advisory provider endpoint status YAML",
    )
    parser.add_argument(
        "--endpoint-timeout",
        type=float,
        default=ENDPOINT_TIMEOUT_S,
        help="seconds for the endpoint check",
    )
    parser.add_argument(
        "--provider-db",
        default=str(DEFAULT_PROVIDERS_DB),
        help="providers_database.yaml used to skip one-time-credit providers",
    )
    parser.add_argument("--dry-run", action="store_true", help="list targets, no calls")
    args = parser.parse_args()
    return run(
        Path(args.config),
        args.db,
        only=args.provider,
        max_models=args.max_models,
        interval_s=args.interval,
        dry_run=args.dry_run,
        status_path=Path(args.status_file),
        endpoint_timeout=args.endpoint_timeout,
        provider_db_path=Path(args.provider_db),
    )


if __name__ == "__main__":
    sys.exit(main())
