"""Redacted, opt-in observations for the gateway smart-router boundary.

The observer is deliberately separate from selection. It records enough
information to compare a gated router decision with the gateway's existing
fallback path without retaining prompts, tool schemas, credentials, or raw
session identifiers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}
_MAX_CHAIN_ENTRIES = 16
_MAX_ESCALATION_ENTRIES = 8


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in _TRUTHY


def cutover_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the new selection order may affect live routing."""
    return _flag(os.environ if env is None else env, "ROUTER_CUTOVER_ENABLED", False)


def logging_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether structured decision observations should be emitted."""
    return _flag(os.environ if env is None else env, "ROUTER_DRY_RUN_LOG", True)


def _session_fingerprint(value: Any) -> str:
    """Return a short non-reversible identifier for a session value."""
    if value is None:
        return ""
    raw = str(value).encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()[:16]


def _metadata_value(value: Any) -> Any:
    """Return a bounded JSON-safe scalar for diagnostic metadata."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)[:128]


def _candidate_record(candidate: Any) -> dict[str, Any]:
    """Serialize only stable, non-prompt candidate metadata."""
    record: dict[str, Any] = {
        "provider": str(getattr(candidate, "provider", ""))[:128],
        "model": str(getattr(candidate, "model", ""))[:256],
        "priority": _metadata_value(getattr(candidate, "priority", None)),
        "reason": str(getattr(candidate, "reason", ""))[:64],
    }
    for field in ("e_time_ms", "confidence", "source"):
        value = getattr(candidate, field, None)
        if value is not None:
            record[field] = (
                str(value)[:64]
                if field == "source"
                else _metadata_value(value)
            )
    return record


def build_decision_record(
    result: Any,
    group: str,
    tier: Any,
    *,
    request: Mapping[str, Any] | None = None,
    request_id: str = "",
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a bounded decision record without copying request content."""
    request = request or {}
    chain = list(getattr(result, "chain", []) or [])
    escalation = list(getattr(result, "escalation", []) or [])
    selected = _candidate_record(chain[0]) if chain else None
    raw_tier = getattr(tier, "value", tier)
    try:
        raw_tier = int(raw_tier)
    except (TypeError, ValueError, OverflowError):
        raw_tier = None
    phase = request.get("phase") or request.get("router_phase") or request.get(
        "x-router-phase", ""
    )
    return {
        "mode": "cutover" if cutover_enabled(env) else "dry_run",
        "cutover_enabled": cutover_enabled(env),
        "request_id": str(request_id)[:64],
        "session": _session_fingerprint(
            request.get("session_id") or request.get("session")
        ),
        "phase": str(phase)[:32],
        "group": str(group)[:128],
        "tier": raw_tier,
        "selected": selected,
        "chain": [_candidate_record(c) for c in chain[:_MAX_CHAIN_ENTRIES]],
        "escalation": [
            _candidate_record(c) for c in escalation[:_MAX_ESCALATION_ENTRIES]
        ],
        "static_fallback": bool(getattr(result, "static_fallback", False)),
        "reason": (
            str(getattr(chain[0], "reason", ""))[:64]
            if chain and getattr(chain[0], "reason", "")
            else "static_fallback"
            if getattr(result, "static_fallback", False)
            else "no_candidate"
        ),
    }


def log_decision(
    result: Any,
    group: str,
    tier: Any,
    *,
    request: Mapping[str, Any] | None = None,
    request_id: str = "",
    env: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Emit one JSON decision record when observation logging is enabled."""
    if not logging_enabled(env):
        return None
    record = build_decision_record(
        result,
        group,
        tier,
        request=request,
        request_id=request_id,
        env=env,
    )
    logger.info("router_decision %s", json.dumps(record, sort_keys=True))
    return record
