"""Non-destructive fallback-chain policy shared by chain writers.

This module only filters chain entries. It never edits provider catalogs,
benchmark data, or other historical records.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


DEFAULT_POLICY_PATH = Path(__file__).resolve().parent.parent / "config" / "dead_providers.yaml"


def _normalized(value: Any) -> str:
    return str(value).strip().lower()


def _model_component(model: Any) -> str:
    return _normalized(model).rsplit("/", 1)[-1]


def candidate_key(provider: Any, model: Any) -> Tuple[str, str]:
    """Return a normalized provider/model identity for chain comparison."""
    return _normalized(provider), _normalized(model)


def _entry_values(entry: Mapping[str, Any]) -> Tuple[str, str]:
    provider = entry.get("provider")
    model = entry.get("model")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("fallback entry requires a non-empty string provider")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("fallback entry requires a non-empty string model")
    return provider.strip(), model.strip()


def load_policy(path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load and validate the reversible fallback-chain exclusion policy."""
    policy_path = Path(path) if path is not None else DEFAULT_POLICY_PATH
    with policy_path.open() as policy_file:
        policy = yaml.safe_load(policy_file) or {}
    if not isinstance(policy, dict):
        raise ValueError(f"chain policy must be a mapping: {policy_path}")

    for key in (
        "blocked_providers",
        "blocked_provider_prefixes",
        "blocked_models",
        "direct_free_fallbacks",
    ):
        value = policy.setdefault(key, [])
        if not isinstance(value, list):
            raise ValueError(f"chain policy {key} must be a list")
        if not all(isinstance(item, dict) for item in value):
            raise ValueError(f"chain policy {key} entries must be mappings")

    if not policy["direct_free_fallbacks"]:
        raise ValueError("chain policy requires at least one direct free fallback")
    for rule in policy["blocked_models"]:
        allowed_models = rule.get("allow_models", [])
        if not isinstance(allowed_models, list) or not all(
            isinstance(model, str) and model.strip() for model in allowed_models
        ):
            raise ValueError("chain policy allow_models must contain non-empty strings")
    for fallback in policy["direct_free_fallbacks"]:
        provider, model = _entry_values(fallback)
        if blocked_reason(provider, model, policy):
            raise ValueError(f"configured direct fallback is blocklisted: {provider}/{model}")
    return policy


def blocked_reason(provider: Any, model: Any, policy: Mapping[str, Any]) -> Optional[str]:
    """Return why a candidate is excluded, or ``None`` when it is allowed."""
    normalized_provider = _normalized(provider)
    normalized_model = _model_component(model)

    for rule in policy.get("blocked_providers", []):
        blocked_provider = _normalized(rule.get("provider", ""))
        if blocked_provider and normalized_provider == blocked_provider:
            return str(rule.get("reason", "blocked provider"))

    for rule in policy.get("blocked_provider_prefixes", []):
        provider_prefix = _normalized(rule.get("prefix", ""))
        if provider_prefix and normalized_provider.startswith(provider_prefix):
            return str(rule.get("reason", "blocked provider prefix"))

    for rule in policy.get("blocked_models", []):
        rule_provider = _normalized(rule.get("provider", ""))
        if not rule_provider or normalized_provider != rule_provider:
            continue
        exact_model = rule.get("model")
        if exact_model is not None and normalized_model == _model_component(exact_model):
            return str(rule.get("reason", "blocked model"))
        model_prefix = rule.get("model_prefix")
        normalized_prefix = _model_component(model_prefix) if model_prefix is not None else ""
        if not normalized_prefix or not normalized_model.startswith(normalized_prefix):
            continue
        allowed_models = rule.get("allow_models", [])
        if any(normalized_model == _model_component(allowed) for allowed in allowed_models):
            continue
        return str(rule.get("reason", "blocked model prefix"))
    return None


def _direct_fallback(policy: Mapping[str, Any]) -> Dict[str, str]:
    for fallback in policy.get("direct_free_fallbacks", []):
        provider, model = _entry_values(fallback)
        if not blocked_reason(provider, model, policy):
            return {"provider": provider, "model": model}
    raise ValueError("chain policy has no allowed direct free fallback")


def _has_direct_fallback(entries: Iterable[Mapping[str, Any]], policy: Mapping[str, Any]) -> bool:
    direct_keys = {
        candidate_key(*_entry_values(fallback))
        for fallback in policy.get("direct_free_fallbacks", [])
    }
    return any(candidate_key(*_entry_values(entry)) in direct_keys for entry in entries)


def validate_chain(entries: Iterable[Mapping[str, Any]], policy: Mapping[str, Any]) -> None:
    """Raise when a chain still violates its membership safety invariants."""
    entries = list(entries)
    if not entries:
        raise ValueError("fallback chain cannot be empty")
    if not _has_direct_fallback(entries, policy):
        raise ValueError("fallback chain has no direct free fallback")
    for entry in entries:
        provider, model = _entry_values(entry)
        reason = blocked_reason(provider, model, policy)
        if reason:
            raise ValueError(f"blocklisted fallback entry {provider}/{model}: {reason}")


def write_yaml_atomic(path: Path, document: Any, *, allow_unicode: bool = False) -> None:
    """Replace a YAML file without exposing a partially written config."""
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    rendered = yaml.safe_dump(
        document,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=allow_unicode,
    )
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.chmod(mode)
        temporary.replace(path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def sanitize_chain(
    entries: Iterable[Mapping[str, Any]],
    policy: Mapping[str, Any],
    max_entries: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Filter a chain, retain entry metadata, and guarantee a direct fallback."""
    if max_entries is not None and max_entries < 1:
        raise ValueError("max_entries must be positive")

    sanitized: List[Dict[str, Any]] = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("fallback chain entries must be mappings")
        provider, model = _entry_values(entry)
        if blocked_reason(provider, model, policy):
            continue
        key = candidate_key(provider, model)
        if key in seen:
            continue
        seen.add(key)
        copied = dict(entry)
        copied["provider"] = provider
        copied["model"] = model
        sanitized.append(copied)
        if max_entries is not None and len(sanitized) >= max_entries:
            break

    if not _has_direct_fallback(sanitized, policy):
        fallback = _direct_fallback(policy)
        if max_entries is not None and len(sanitized) >= max_entries:
            sanitized[-1] = fallback
        else:
            sanitized.append(fallback)

    for priority, entry in enumerate(sanitized, start=1):
        entry["priority"] = priority
    validate_chain(sanitized, policy)
    return sanitized
