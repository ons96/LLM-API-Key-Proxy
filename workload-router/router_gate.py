"""Gated cutover controls for the workload router.

The router runs parallel to the gateway: every decision it makes is observed
in dry-run logs, and nothing is applied to live routing until the cutover flag
is explicitly enabled. The flag defaults OFF, so importing this module or
running the server never changes virtual_models.yaml fallback behavior.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Mapping
from typing import IO, Any

CUTOVER_ENV = "ROUTER_CUTOVER_ENABLED"
DRY_RUN_ENV = "ROUTER_DRY_RUN_LOG"

_TRUTHY = {"1", "true", "yes", "on"}


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in _TRUTHY


def cutover_enabled(env: Mapping[str, str] | None = None) -> bool:
    """True only when ROUTER_CUTOVER_ENABLED is explicitly truthy; default OFF."""
    return _flag(os.environ if env is None else env, CUTOVER_ENV, False)


def dry_run_logging_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Dry-run decision logging stays on by default; it never mutates routing."""
    return _flag(os.environ if env is None else env, DRY_RUN_ENV, True)


def dry_run_record(
    deployment: Any,
    reason: str,
    *,
    session_id: str = "",
    phase: str = "",
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the tier/deployment/why record for one routed request."""
    return {
        "ts": round(time.time(), 3),
        "mode": "cutover" if cutover_enabled(env) else "dry_run",
        "tier": getattr(deployment, "chain_priority", None),
        "deployment": getattr(deployment, "deployment_id", ""),
        "why": reason,
        "session": session_id[:64],
        "phase": phase[:32],
    }


def dry_run_line(
    deployment: Any,
    reason: str,
    *,
    session_id: str = "",
    phase: str = "",
    env: Mapping[str, str] | None = None,
) -> str:
    """One JSON line describing tier/deployment/why for a routed request."""
    return json.dumps(dry_run_record(deployment, reason, session_id=session_id, phase=phase, env=env), sort_keys=True)


def log_dry_run(
    deployment: Any,
    reason: str,
    *,
    session_id: str = "",
    phase: str = "",
    env: Mapping[str, str] | None = None,
    stream: IO[str] | None = None,
) -> None:
    """Emit the dry-run decision line unless logging is explicitly disabled."""
    if not dry_run_logging_enabled(env):
        return
    out = sys.stderr if stream is None else stream
    print(dry_run_line(deployment, reason, session_id=session_id, phase=phase, env=env), file=out, flush=True)
