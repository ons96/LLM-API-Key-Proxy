"""Per-task-class max_tokens budget for the smart router (#479).

"Speed guarantee": total latency = TTFT + tokens/TPS, so the cheapest lever is
to stop small tasks from generating massive outputs. A greeting must not burn
4096 tokens (the LiteLLM/fallback default).

Computes an intelligent per-request ``max_tokens`` cap based on the request's
task class (from ``request_features``) and tier, then clamps it to the
provider's context window minus the estimated input tokens.

Rules (see task-board #479):
  * Applied ONLY when the client did NOT supply its own ``max_tokens`` /
    ``max_completion_tokens`` - user intent always wins (never overridden).
  * ``max_tokens <= model_context_limit - estimated_input_tokens - safety_margin``.
  * Config map: ``config/token_budget.yaml`` (class defaults + tier multipliers),
    overridable at runtime via the ``TOKEN_BUDGET_OVERRIDES`` env (JSON string
    mapping task_class -> int).
  * Returns an ``X-Route-Max-Tokens`` header value for debug/dry-run surfacing.

Pure functions plus a thin config loader; stdlib-only (pyyaml for the config is
already a project dependency). Safe to import without the full router.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_SAFETY_MARGIN = 256

# Built-in defaults (mirror config/token_budget.yaml). task_class values match
# request_features.TaskClass. A missing class falls back to a conservative 512.
CLASS_DEFAULTS: Dict[str, int] = {
    "greeting": 128,
    "short-qa": 512,
    "code-edit": 1024,
    "code-gen": 2048,
    "reasoning": 4096,
    "agentic-multi-step": 8192,
    "summarization": 1500,
    "vision-caption": 512,
    "file-analysis": 2048,
}

# FLOOR if the clamp ceiling (context - input - margin) is unhelpfully small
# (e.g. a degenerate request whose input already exceeds the context window).
FLOOR_TOKEN_BUDGET = 16

# Largest provider context to assume when provider_caps.yaml is missing/empty.
FALLBACK_CONTEXT_LIMIT = 128_000


def load_budget_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Merge a token_budget.yaml (if present) over the built-in defaults.

    Returns {'class_defaults': {...}, 'tier_multiplier': {...},
             'safety_margin': int}.
    """
    cfg: Dict[str, Any] = {
        "class_defaults": dict(CLASS_DEFAULTS),
        "tier_multiplier": {"0": 1.0, "1": 1.0, "2": 1.2, "3": 1.5, "4": 2.0},
        "safety_margin": DEFAULT_SAFETY_MARGIN,
    }
    if path:
        p = Path(path)
        if p.exists():
            try:
                import yaml

                data = yaml.safe_load(p.read_text()) or {}
                cdefs = data.get("class_defaults") or {}
                if isinstance(cdefs, dict):
                    cfg["class_defaults"].update(cdefs)
                tm = data.get("tier_multiplier") or {}
                if isinstance(tm, dict):
                    cfg["tier_multiplier"].update({str(k): float(v) for k, v in tm.items()})
                sm = data.get("safety_margin")
                if isinstance(sm, (int, float)):
                    cfg["safety_margin"] = int(sm)
            except Exception:
                pass
    return cfg


def env_overrides() -> Dict[str, int]:
    """Parse TOKEN_BUDGET_OVERRIDES env as JSON {task_class: int}."""
    raw = os.environ.get("TOKEN_BUDGET_OVERRIDES", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return {str(k).lower(): int(v) for k, v in parsed.items() if isinstance(v, (int, float))}
    except Exception:
        return {}


def _tier_multiplier(tier: Optional[int], tm: Dict[str, float]) -> float:
    if tier is None:
        return 1.0
    return float(tm.get(str(tier), 1.0))


def compute_max_tokens(
    task_class: str,
    estimated_input_tokens: int,
    context_limit: int,
    *,
    tier: Optional[int] = None,
    class_defaults: Optional[Dict[str, int]] = None,
    tier_multiplier: Optional[Dict[str, float]] = None,
    safety_margin: int = DEFAULT_SAFETY_MARGIN,
    overrides: Optional[Dict[str, int]] = None,
) -> int:
    """Return the effective max_tokens cap for a request's task class.

    ``context_limit`` is the provider's max context window (tokens). The result
    is always >= FLOOR_TOKEN_BUDGET and <= class_default.
    """
    overrides = overrides or env_overrides()
    base = overrides.get(task_class) or (class_defaults or CLASS_DEFAULTS).get(task_class, 512)
    mult = _tier_multiplier(tier, tier_multiplier or {})
    base = max(1, int(round(base * mult)))

    ceiling = max(
        FLOOR_TOKEN_BUDGET,
        context_limit - int(estimated_input_tokens or 0) - int(safety_margin),
    )
    return max(FLOOR_TOKEN_BUDGET, min(base, ceiling))


def apply_token_budget(
    request: Dict[str, Any],
    features,
    context_limit: int,
    *,
    tier: Optional[int] = None,
    budget_config: Optional[str] = None,
    enabled: bool = False,
) -> Optional[str]:
    """Apply the token budget to ``request`` in place.

    Returns an ``X-Route-Max-Tokens`` debug header string (or None if the
    client supplied its own max_tokens / nothing to cap). ``request`` is only
    mutated (``max_tokens`` set) when ``enabled=True`` (ROUTER_ENABLED); when
    ``enabled=False`` the value is still computed and returned so dry-run callers
    can log the intended cap without changing the outgoing request.

    ``features`` must expose ``task_class``, ``estimated_input_tokens`` and
    ``has_max_tokens`` (a RequestFeatures instance).
    """
    if getattr(features, "has_max_tokens", False):
        return None

    cfg = load_budget_config(budget_config)
    task = features.task_class.value if hasattr(features.task_class, "value") else str(features.task_class)
    cap = compute_max_tokens(
        task,
        getattr(features, "estimated_input_tokens", 0),
        context_limit,
        tier=tier,
        class_defaults=cfg["class_defaults"],
        tier_multiplier=cfg["tier_multiplier"],
        safety_margin=cfg["safety_margin"],
    )

    header = f"{cap}"
    if enabled:
        # OpenAI-compat: update the key the client/request actually uses.
        if "max_completion_tokens" in request:
            request["max_completion_tokens"] = cap
        else:
            request["max_tokens"] = cap
        header += " (applied)"
    else:
        header += " (dry-run, not applied)"
    return header
