"""Free-only invariant: quota tiers, allowed models, and the typed paid error.

The gateway has to answer three questions cheaply, per request:

  1. is free-only mode on?
  2. what quota tier is (provider, model)?
  3. is this model allowed in an automatic chain?

Tier source, in order:
  a. the catalog artifact (``config/catalog.json`` or ``config/catalog.json.gz``)
     produced by ``llm-provider-manager/scripts/build_gateway_catalog.py``
  b. ``config/router_config.yaml`` declarations, for anything the artifact lacks

The gateway never calls a provider's ``/v1/models`` at request time; this module
only reads local files, so it costs microseconds.

Tier vocabulary is the *commercial* tier, deliberately distinct from
``tier_classifier.Tier`` (task difficulty). See DECISIONS.md D4.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# provider_type -> commercial tier. Same mapping as
# llm-provider-manager/gateway_catalog/model.py; duplicated deliberately so the
# gateway does not import the catalog package.
QUOTA_TIERS: dict[str, int] = {
    "fully_free": 0,
    "internal": 0,
    "free_tier_limited": 1,
    "free_daily": 1,
    "free_rpd": 2,
    "free_one_time": 3,
    "paid": 4,
    "unknown": 4,
}

PAID_TIER = 4

_CACHE: dict[str, Any] = {"stamp": None, "tiers": {}, "types": {}, "models": {}}
_LOCK = threading.Lock()


class PaidModelError(Exception):
    """A request named a model that free-only mode forbids.

    Typed on purpose: the caller must be able to tell "you asked for a paid model"
    apart from "everything failed", and it must never fall back silently.
    """

    error_type = "paid_model_rejected"
    status_code = 402

    def __init__(self, provider: str, model: str, tier: int, reason: str):
        self.provider = provider
        self.model = model
        self.tier = tier
        self.reason = reason
        super().__init__(
            f"model '{provider}/{model}' is tier {tier} and free-only mode is on: {reason}"
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "message": str(self),
                "type": self.error_type,
                "code": "paid_model_rejected",
                "param": "model",
                "provider": self.provider,
                "model": self.model,
                "tier": self.tier,
                "reason": self.reason,
            }
        }


def _router_config_path(root_dir: Path) -> Path:
    return Path(root_dir) / "config" / "router_config.yaml"


def read_router_config(root_dir: Path | str) -> dict[str, Any]:
    """Read config/router_config.yaml. Returns {} when it is missing or invalid."""
    path = _router_config_path(Path(root_dir))
    if not path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # yaml absent, unreadable, malformed
        logger.warning(f"free_policy: cannot read {path}: {exc}")
        return {}


def free_only_from_disk(root_dir: Path | str) -> bool:
    """free_only_mode straight from the config file, router object not required.

    ``/v1/models`` must respect the invariant even when no RouterCore was built
    (local harnesses, tests, a misconfigured boot).
    """
    return free_only_enabled(read_router_config(root_dir))


def _catalog_paths(root_dir: Path) -> list[Path]:
    config = root_dir / "config"
    return [config / "catalog.json", config / "catalog.json.gz"]


def _read_catalog(root_dir: Path) -> Optional[dict[str, Any]]:
    for path in _catalog_paths(root_dir):
        if not path.exists():
            continue
        try:
            if path.suffix == ".gz":
                with gzip.open(path, "rb") as fh:
                    return json.loads(fh.read().decode("utf-8"))
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning(f"free_policy: cannot read {path}: {exc}")
    return None


def load_tiers(
    root_dir: Path | str,
    router_config: dict[str, Any] | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Load provider tiers. Cached until the catalog file changes on disk."""
    root = Path(root_dir)
    if router_config is None:
        router_config = read_router_config(root)
    paths = _catalog_paths(root)
    stamp = tuple(p.stat().st_mtime if p.exists() else 0 for p in paths)

    with _LOCK:
        if _CACHE["stamp"] == stamp and not force:
            return _CACHE

        tiers: dict[str, int] = {}
        types: dict[str, str] = {}
        models: dict[str, set[str]] = {}

        catalog = _read_catalog(root)
        if catalog:
            for provider in catalog.get("providers") or []:
                pid = provider.get("provider_id")
                if not pid:
                    continue
                ptype = provider.get("provider_type") or "unknown"
                types[pid] = ptype
                tiers[pid] = int(provider.get("quota_tier", QUOTA_TIERS.get(ptype, PAID_TIER)))
                models[pid] = {
                    m.get("model_id")
                    for m in provider.get("models") or []
                    if m.get("is_free")
                }

        # Fill gaps from router_config so a missing artifact does not open the gate.
        providers_cfg = ((router_config or {}).get("providers") or {})
        for name, cfg in providers_cfg.items():
            if not isinstance(cfg, dict) or name in tiers:
                continue
            if not cfg.get("enabled", False):
                tiers.setdefault(name, PAID_TIER)
                types.setdefault(name, "unknown")
                continue
            if cfg.get("free_tier") or cfg.get("free_tier_models"):
                tiers[name] = 0
                types.setdefault(name, "fully_free")
                models.setdefault(
                    name, set(cfg.get("free_tier_models") or [])
                )
            else:
                # Enabled but not declared free: fail closed.
                tiers[name] = PAID_TIER
                types.setdefault(name, "unknown")

        _CACHE.update(
            {"stamp": stamp, "tiers": tiers, "types": types, "models": models}
        )
        return _CACHE


def _normalize(name: str) -> str:
    """Provider ids differ between the DB and the gateway config: the catalog
    writes ``blaze-free`` / ``nous-inference`` while config declares
    ``blaze_free`` / ``nous_inference``. Normalise before every lookup, or a
    paid provider slips through under the other spelling."""
    return str(name or "").strip().lower().replace("-", "_")


def _lookup(mapping: dict[str, Any], provider: str, default: Any) -> Any:
    if provider in mapping:
        return mapping[provider]
    normalized = _normalize(provider)
    for key, value in mapping.items():
        if _normalize(key) == normalized:
            return value
    return default


def provider_tier(cache: dict[str, Any], provider: str) -> int:
    return int(_lookup(cache["tiers"], provider, PAID_TIER))


def provider_type(cache: dict[str, Any], provider: str) -> str:
    return str(_lookup(cache["types"], provider, "unknown"))


def model_is_free(cache: dict[str, Any], provider: str, model: str) -> bool:
    """Free if the catalog says so, or if the model id itself is marked free."""
    if str(model).endswith(":free"):
        return True
    free_models = _lookup(cache["models"], provider, set()) or set()
    return bool(free_models) and model in free_models


def policy_skipped() -> bool:
    """True when a hermetic test asked to bypass the real catalog.

    Unit tests build mock router configs whose providers do not exist in the
    catalog artifact, so every one of them would be tier 4 and silently dropped.
    Same convention as ``SKIP_REAL_DEAD_PROVIDERS``: only tests set this.
    """
    return os.getenv("SKIP_REAL_FREE_POLICY", "false").strip().lower() == "true"


def free_only_enabled(router_config: dict[str, Any] | None = None) -> bool:
    """Router config governs; the env var can force it on for a test."""
    env = os.getenv("FREE_ONLY_MODE", "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    if env in {"0", "false", "no", "off"}:
        return False
    return bool((router_config or {}).get("free_only_mode", False))


def allowed_in_free_mode(
    cache: dict[str, Any],
    provider: str,
    model: str,
    *,
    free_only: bool,
    pinned: bool = False,
) -> tuple[bool, str]:
    """Whether (provider, model) may be used. Returns (allowed, reason).

    Pinned access is the only way to reach tier 3 in free-only mode; tier 4 is
    never allowed automatically.
    """
    if not free_only:
        return True, "free_only_mode off"
    tier = provider_tier(cache, provider)
    if tier == PAID_TIER:
        return False, f"tier {tier} ({provider_type(cache, provider)}) is paid"
    if tier == 3:
        # Tier 3 is a last resort, not a forbidden tier: the brief reserves the
        # credit rather than excluding it, and refusing it outright empties every
        # chain that depends on a one-time-credit relay. Ordering (chain_order.py)
        # and the ledger's per-day budget are what keep it from being drained.
        return True, "tier 3 (one-time credit): allowed last, after tier 0/1/2"
    if tier <= 2 and not model_is_free(cache, provider, model):
        known = _lookup(cache["models"], provider, set()) or set()
        if known:
            return False, "model is not in the catalog's free set for this provider"
    return True, f"tier {tier} allowed"


def filter_model_ids(
    model_ids: list[str], cache: dict[str, Any], *, free_only: bool
) -> list[str]:
    """Drop paid-tier model ids from an OpenAI-compatible ``/v1/models`` list.

    Ids without a ``/`` are virtual aliases, not provider models, so they are kept:
    the alias itself routes, and each alias is filtered at routing time.
    """
    if not free_only:
        return list(model_ids)
    kept: list[str] = []
    for model_id in model_ids:
        if "/" in model_id:
            provider, model = model_id.split("/", 1)
            allowed, _reason = allowed_in_free_mode(
                cache, provider, model, free_only=True
            )
            if not allowed:
                continue
        kept.append(model_id)
    return kept
