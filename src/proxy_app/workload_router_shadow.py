"""Dry-run (shadow) evaluation of the vendored ``workload-router`` core.

Task-board #1070, acceptance criterion: "Dry-run mode logging
tier/deployment/why per request without changing live routing."

This module is *observe only*:

- Disabled unless ``WORKLOAD_ROUTER_DRY_RUN`` is truthy. When disabled it does
  no I/O, loads no modules, and returns immediately.
- When enabled it computes what the parallel ``workload-router/`` component
  *would* select for the live request and appends one bounded, prompt-free
  JSON line to ``WORKLOAD_ROUTER_DRY_RUN_LOG``. The result is discarded by the
  caller: no return value of this module reaches the routing decision.
- It never mutates the request and never lets an exception escape, so live
  routing (``config/virtual_models.yaml`` chains and the existing fallback
  behaviour) is byte-for-byte unchanged whether the flag is on or off.

Recorded fields are metadata only -- model alias, tier (capability group),
deployment id, routing reason, counts and timing. Prompt text, tool
definitions and upstream error strings are never written to the log.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

ENV_ENABLE = "WORKLOAD_ROUTER_DRY_RUN"
ENV_LOG_PATH = "WORKLOAD_ROUTER_DRY_RUN_LOG"
ENV_CATALOG = "WORKLOAD_ROUTER_DRY_RUN_CATALOG"
ENV_MAX_BYTES = "WORKLOAD_ROUTER_DRY_RUN_MAX_BYTES"

DEFAULT_LOG_PATH = "logs/workload-router-dry-run.jsonl"
DEFAULT_MAX_BYTES = 8 * 1024 * 1024

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROUTER_DIR = _REPO_ROOT / "workload-router"

# The gateway's opencode-phase-router header vocabulary, mapped onto the
# workload-router's bounded phase vocabulary. Unknown values stay neutral.
_PHASE_ALIASES = {
    "build_exploration": "explore",
    "build_iteration": "implement",
    "build_verification": "verify",
    "build_debug": "debug",
    "deep_debug": "deep_debug",
}

_TRUTHY = {"1", "true", "yes", "on"}

# Lazy, process-wide handles. Loading the vendored core is expensive enough
# that it must happen at most once per process, and only when enabled.
_core: Any = None
_config: Any = None
_catalog_path: str | None = None
_catalog: Any = None
_lock = threading.Lock()
_log_lock = threading.Lock()


def is_enabled() -> bool:
    """Return True when the dry-run flag is set (default: off)."""
    return os.environ.get(ENV_ENABLE, "").strip().lower() in _TRUTHY


def _log_path() -> Path:
    raw = os.environ.get(ENV_LOG_PATH, "").strip()
    return Path(raw) if raw else Path(DEFAULT_LOG_PATH)


def _max_bytes() -> int:
    raw = os.environ.get(ENV_MAX_BYTES, "").strip()
    try:
        value = int(raw) if raw else DEFAULT_MAX_BYTES
    except (TypeError, ValueError):
        return DEFAULT_MAX_BYTES
    return value if value > 0 else DEFAULT_MAX_BYTES


def _load_vendored() -> tuple[Any, Any]:
    """Import ``workload-router/router_core.py`` and ``router_config.py``.

    The vendored directory is not a package (and its name contains a hyphen),
    so both files are loaded by path. ``router_config`` imports ``router_core``
    under its top-level name; that name is bound only for the duration of the
    import and the previous ``sys.modules`` entry is restored afterwards, so
    the gateway's own ``proxy_app.router_core`` can never be shadowed.
    """
    global _core, _config
    if _core is not None and _config is not None:
        return _core, _config
    core_path = _ROUTER_DIR / "router_core.py"
    config_path = _ROUTER_DIR / "router_config.py"
    if not core_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("vendored workload-router core not found")
    core_spec = importlib.util.spec_from_file_location("_workload_router_core", core_path)
    if core_spec is None or core_spec.loader is None:
        raise ImportError("cannot load workload-router core")
    core = importlib.util.module_from_spec(core_spec)
    core_spec.loader.exec_module(core)

    previous = sys.modules.get("router_core")
    sys.modules["router_core"] = core
    try:
        config_spec = importlib.util.spec_from_file_location(
            "_workload_router_config", config_path
        )
        if config_spec is None or config_spec.loader is None:
            raise ImportError("cannot load workload-router config")
        config = importlib.util.module_from_spec(config_spec)
        config_spec.loader.exec_module(config)
    finally:
        if previous is None:
            sys.modules.pop("router_core", None)
        else:
            sys.modules["router_core"] = previous

    with _lock:
        _core, _config = core, config
    return core, config


def _deployments(config: Any) -> Any:
    """Load (and cache) the credential-free deployment catalog."""
    global _catalog, _catalog_path
    explicit = os.environ.get(ENV_CATALOG, "").strip() or None
    if _catalog is not None and _catalog_path == explicit:
        return _catalog
    with _lock:
        if _catalog is not None and _catalog_path == explicit:
            return _catalog
        catalog = config.load_deployments(explicit)
        _catalog, _catalog_path = catalog, explicit
    return catalog


def _phase_from(headers: Mapping[str, str] | None) -> str:
    if not headers:
        return ""
    try:
        lowered = {str(key).lower(): value for key, value in headers.items()}
    except (AttributeError, TypeError, ValueError):
        return ""
    raw = lowered.get("x-phase-router-phase") or lowered.get("x-router-phase") or ""
    value = str(raw).strip().lower().replace("-", "_")[:64]
    core, _ = _load_vendored()
    phase = core.normalize_phase(value)
    if phase:
        return phase
    return _PHASE_ALIASES.get(value, "")


def _prompt_of(request_data: Mapping[str, Any]) -> str:
    messages = request_data.get("messages")
    if not isinstance(messages, list):
        return ""
    return "\n".join(
        str(item.get("content", "")) for item in messages if isinstance(item, dict)
    )


def decide(request_data: Mapping[str, Any], headers: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Return ``{"tier", "deployment", "why"}`` for a request.

    Pure evaluation against the vendored core: no I/O, no request mutation.
    Expected misses (no eligible deployment) are reported as a reason rather
    than raised, because a shadow decision must never break the caller.
    """
    core, config = _load_vendored()
    messages = request_data.get("messages")
    message_list = messages if isinstance(messages, list) else []
    prompt = _prompt_of(request_data)
    features = core.RequestFeatures(
        prompt=prompt,
        context_tokens=core.estimate_context_tokens(
            [m for m in message_list if isinstance(m, dict)]
        ),
        tools=bool(request_data.get("tools") or request_data.get("tool_choice")),
        structured_output=bool(request_data.get("response_format")),
        phase=_phase_from(headers),
        model_group="",
        continuation=bool(request_data.get("continuation", False)),
    )
    tier = core.group_for(features)
    phase = features.phase
    try:
        plan = core.route_plan(features, _deployments(config))
    except LookupError:
        return {"tier": tier, "deployment": None, "why": "no_eligible_deployment", "phase": phase}
    selected = plan.candidates[0]
    return {
        "tier": tier,
        "deployment": selected.deployment_id,
        "why": plan.reason,
        "capability": selected.capability.value,
        "phase": phase,
    }


def _bounded(value: Any, limit: int = 128) -> str:
    text = "" if value is None else str(value)
    return text[:limit]


def _rotate(path: Path, limit: int) -> None:
    """Keep the shadow log bounded on a long-running gateway (1 GB VPS)."""
    try:
        if path.is_file() and path.stat().st_size >= limit:
            backup = path.with_name(path.name + ".1")
            if backup.exists():
                backup.unlink()
            path.replace(backup)
    except OSError as exc:  # rotation is best effort; logging must continue
        logger.debug("dry-run log rotation skipped: %s", exc)


def _append(record: Mapping[str, Any]) -> None:
    path = _log_path()
    limit = _max_bytes()
    line = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
    with _log_lock:
        parent = path.parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        _rotate(path, limit)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def observe(
    request_data: Any,
    *,
    request_id: str = "",
    live_model: str = "",
    headers: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Record one dry-run decision. Returns the record, or ``None`` if off.

    Contract: never raises, never mutates ``request_data``, never writes prompt
    content. Callers discard the result, so it cannot influence routing.
    """
    if not is_enabled():
        return None
    try:
        snapshot = dict(request_data) if isinstance(request_data, Mapping) else {}
        decision = decide(snapshot, headers)
        messages = snapshot.get("messages")
        record = {
            "ts": round(time.time(), 3),
            "mode": "dry_run",
            "request_id": _bounded(request_id),
            "live_model": _bounded(live_model or snapshot.get("model", "")),
            "tier": _bounded(decision.get("tier")),
            "deployment": _bounded(decision.get("deployment")),
            "why": _bounded(decision.get("why")),
            "capability": _bounded(decision.get("capability")),
            "phase": _bounded(decision.get("phase", "")),
            "tools": bool(snapshot.get("tools") or snapshot.get("tool_choice")),
            "structured_output": bool(snapshot.get("response_format")),
            "messages": len(messages) if isinstance(messages, list) else 0,
            "continuation": bool(snapshot.get("continuation", False)),
        }
        _append(record)
        return record
    except Exception as exc:  # containment is the contract for a shadow path
        logger.debug(
            "workload-router dry-run observation failed: %s", type(exc).__name__
        )
        return None
