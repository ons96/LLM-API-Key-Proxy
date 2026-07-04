#!/usr/bin/env python3
"""
Active provider performance probe (#332).

Sends a canonical short coding prompt to each enabled LLM provider+model
declared in config/router_config.yaml, measures TTFT + TPS + latency via
streaming, and writes results to the telemetry DB (api_calls table) via
TelemetryManager.record_call().

The /v1/tps-stats endpoint reads api_calls automatically — no new endpoint
needed. Probe data flows through end-to-end: probe -> DB -> endpoint.

Usage:
    python scripts/probe_providers.py                    # probe all enabled providers
    python scripts/probe_providers.py --provider groq    # probe one provider
    python scripts/probe_providers.py --dry-run          # list targets, no calls
    python scripts/probe_providers.py --max-models 3     # cap models per provider
    python scripts/probe_providers.py --interval 5       # seconds between probes
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
# rotator_library/__init__ pulls in transitively) — keeps unit tests hermetic.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("probe_providers")

# Search-only / non-LLM providers — not probe targets.
SEARCH_ONLY = {"brave_search", "tavily", "duckduckgo", "exa", "jina"}

# Canonical short coding prompt (~10 input tokens, ~30-50 output).
CANONICAL_PROMPT = (
    "Write a Python function to reverse a string. Reply with code only."
)
MAX_OUTPUT_TOKENS = 64
REQUEST_TIMEOUT_S = 30

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


def load_provider_config(config_path: Path) -> Dict[str, Any]:
    """Load router_config.yaml, return providers dict."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("providers", {})


def select_targets(
    providers: Dict[str, Any],
    only: Optional[str] = None,
    max_models: Optional[int] = None,
) -> List[Tuple[str, str, str, str]]:
    """Build list of (provider_id, model, base_url, api_key) targets.

    Skips: disabled, dead, search-only, providers without models.
    """
    targets: List[Tuple[str, str, str, str]] = []
    for pid, pdata in providers.items():
        if pid in SEARCH_ONLY:
            continue
        if not pdata.get("enabled", False):
            continue
        if pdata.get("_dead"):
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
) -> int:
    providers = load_provider_config(config_path)
    targets = select_targets(providers, only=only, max_models=max_models)
    log.info("selected %d probe targets (%d providers)", len(targets), len({t[0] for t in targets}))
    if dry_run:
        for pid, model, base_url, _ in targets:
            print(f"{pid}\t{model}\t{base_url}")
        return 0
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
    parser.add_argument("--dry-run", action="store_true", help="list targets, no calls")
    args = parser.parse_args()
    return run(
        Path(args.config),
        args.db,
        only=args.provider,
        max_models=args.max_models,
        interval_s=args.interval,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
