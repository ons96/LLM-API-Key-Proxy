"""Validated, credential-free deployment catalog loading."""

import json
from pathlib import Path

from router_core import Capability, Deployment


def default_deployments() -> list[Deployment]:
    return [
        Deployment("general", Capability.FAST_GENERAL, 16_384),
        Deployment("tool", Capability.FAST_TOOL, 16_384, tools=True),
        Deployment("implement", Capability.CODE_IMPLEMENT, 32_768, tools=True),
        Deployment("verify", Capability.CODE_VERIFY, 32_768, tools=True),
        Deployment("debug", Capability.CODE_DEBUG, 32_768, tools=True),
    ]


def load_deployments(path: str | Path | None = None) -> list[Deployment]:
    """Load profiles, falling back safely when optional config is unavailable."""
    if path is None:
        return default_deployments()
    try:
        entries = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(entries, list) or not entries:
            raise ValueError("catalog must be a non-empty list")
        deployments = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("catalog entries must be objects")
            context_limit = int(entry["context_limit"])
            latency_ms = float(entry.get("latency_ms", 1000.0))
            success_rate = float(entry.get("success_rate", 0.5))
            quota_remaining = None if entry.get("quota_remaining") is None else int(entry["quota_remaining"])
            if context_limit <= 0 or latency_ms <= 0 or not 0 <= success_rate <= 1:
                raise ValueError("invalid deployment performance range")
            if quota_remaining is not None and quota_remaining < 0:
                raise ValueError("quota_remaining cannot be negative")
            deployments.append(Deployment(
                deployment_id=str(entry["deployment_id"]),
                capability=Capability(str(entry["capability"])),
                context_limit=context_limit,
                model=str(entry.get("model", entry["deployment_id"])),
                tools=bool(entry.get("tools", False)),
                structured_output=bool(entry.get("structured_output", False)),
                latency_ms=latency_ms,
                success_rate=success_rate,
                provider=str(entry.get("provider", "unconfigured")),
                config_fingerprint=str(entry.get("config_fingerprint", "default")),
                quota_remaining=quota_remaining,
                quota_class=str(entry.get("quota_class", "unknown")),
            ))
        return deployments
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return default_deployments()
