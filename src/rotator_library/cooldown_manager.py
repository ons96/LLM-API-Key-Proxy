import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

log = logging.getLogger(__name__)

_DEFAULT_POLICY = {"wait_on_429": False, "max_wait_s": 0, "reason": "default-rotate"}


class CooldownManager:
    """
    Manages global cooldown periods for API providers to handle IP-based rate limiting.
    This ensures that once a 429 error is received for a provider, all subsequent
    requests to that provider are paused for a specified duration.
    """
    def __init__(self):
        self._cooldowns: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def is_cooling_down(self, provider: str) -> bool:
        """Checks if a provider is currently in a cooldown period."""
        async with self._lock:
            return provider in self._cooldowns and time.time() < self._cooldowns[provider]

    async def start_cooldown(self, provider: str, duration: int):
        """
        Initiates or extends a cooldown period for a provider.
        The cooldown is set to the current time plus the specified duration.
        """
        async with self._lock:
            self._cooldowns[provider] = time.time() + duration

    async def get_cooldown_remaining(self, provider: str) -> float:
        """
        Returns the remaining cooldown time in seconds for a provider.
        Returns 0 if the provider is not in a cooldown period.
        """
        async with self._lock:
            if provider in self._cooldowns:
                remaining = self._cooldowns[provider] - time.time()
                return max(0, remaining)
            return 0


class CooldownPolicy:
    """
    Per-provider 429 wait-vs-fallback policy (#218, generalized by #1069).

    Explicit per-provider wait settings win (back-compat). New providers are
    tagged with `quota_class` (unlimited/credit/trial/paid) + `supports_cache`
    and the wait-vs-fallback decision derives from the quota_classes table:
    cache-supporting free-credit/trial tiers wait to preserve credit and warm
    cache, unlimited-free prefers the faster fallback, paid rotates
    immediately. Unknown classes keep the legacy default (rotate now).
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        if config_path is None:
            # ponytail: resolve relative to this file so tests + prod both work
            config_path = str(Path(__file__).resolve().parents[2] / "config" / "cooldown_policy.yaml")
        self._policies: Dict[str, dict] = {}
        self._classes: Dict[str, dict] = {}
        self._load(config_path)

    def _load(self, config_path: str) -> None:
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f) or {}
            providers = data.get("providers", {}) or {}
            self._default = data.get("default", _DEFAULT_POLICY) or _DEFAULT_POLICY
            # #1069: quota-class table -> derived wait-vs-fallback policies.
            # In-memory dict lookups only; no hot-path cost.
            self._classes = {}
            for cls_name, cfg in (data.get("quota_classes", {}) or {}).items():
                if not isinstance(cfg, dict):
                    continue
                self._classes[str(cls_name).lower()] = {
                    "wait_on_429": bool(cfg.get("wait_on_429", False)),
                    "max_wait_s": int(cfg.get("max_wait_s", 0) or 0),
                    "require_cache": bool(cfg.get("require_cache", False)),
                    "reason": str(cfg.get("reason", "quota-class")),
                }
            # ponytail: normalize keys to lowercase, drop missing fields
            for name, cfg in providers.items():
                if not isinstance(cfg, dict):
                    continue
                self._policies[name.lower()] = {
                    "wait_on_429": bool(cfg.get("wait_on_429", False)),
                    "max_wait_s": int(cfg.get("max_wait_s", 0) or 0),
                    "reason": str(cfg.get("reason", "configured")),
                    # #1069: scalable metadata path for new providers
                    "quota_class": str(cfg.get("quota_class", "") or "").lower(),
                    "supports_cache": bool(cfg.get("supports_cache", False)),
                    # explicit wait fields always win over the class table (back-compat)
                    "explicit_wait": ("wait_on_429" in cfg) or ("max_wait_s" in cfg),
                }
            # ensure default has all keys
            self._default = {
                "wait_on_429": bool(self._default.get("wait_on_429", False)),
                "max_wait_s": int(self._default.get("max_wait_s", 0) or 0),
                "reason": str(self._default.get("reason", "default-rotate")),
            }
            log.info(
                "CooldownPolicy loaded from %s: %d providers configured, default wait_on_429=%s",
                config_path,
                len(self._policies),
                self._default["wait_on_429"],
            )
        except FileNotFoundError:
            log.warning("cooldown_policy.yaml not found at %s, using default (no wait)", config_path)
            self._policies = {}
            self._classes = {}
            self._default = _DEFAULT_POLICY.copy()
        except Exception as e:
            log.error("failed to load cooldown_policy.yaml: %s — using default (no wait)", e)
            self._policies = {}
            self._classes = {}
            self._default = _DEFAULT_POLICY.copy()

    def _class_policy(self, quota_class: str, supports_cache: bool) -> dict:
        """#1069: derive the wait-vs-fallback policy from quota class + cache support.

        Unknown class -> the configured default (current behavior preserved).
        Classes with require_cache rotate immediately when the deployment does
        not support prompt caching (#218 eligibility: cache benefit required).
        """
        cls = self._classes.get(quota_class)
        if cls is None:
            return self._default
        if cls["require_cache"] and not supports_cache:
            return {
                "wait_on_429": False,
                "max_wait_s": 0,
                "reason": f"{quota_class}-no-cache-rotate",
            }
        return {
            "wait_on_429": cls["wait_on_429"],
            "max_wait_s": cls["max_wait_s"],
            "reason": cls["reason"],
        }

    def get_policy(self, provider: str) -> dict:
        """Return merged policy for a provider (falls back to default).

        Resolution order: explicit wait_on_429/max_wait_s (back-compat, #218)
        > quota_class + supports_cache metadata (#1069) > default.
        """
        entry = self._policies.get(provider.lower())
        if entry is None:
            return self._default
        if entry.get("explicit_wait"):
            return {
                "wait_on_429": entry["wait_on_429"],
                "max_wait_s": entry["max_wait_s"],
                "reason": entry["reason"],
            }
        return self._class_policy(entry.get("quota_class", ""), entry.get("supports_cache", False))

    def should_wait_on_429(
        self, provider: str, retry_after_s: Optional[float]
    ) -> Tuple[bool, float]:
        """
        Decide whether the current request should wait out a 429 instead of rotating.

        Returns (wait: bool, deadline_ts: float). deadline_ts is the monotonic-ish
        time.time() at which the wait expires (only meaningful if wait=True).
        """
        if retry_after_s is None or retry_after_s <= 0:
            return (False, 0.0)
        policy = self.get_policy(provider)
        if not policy["wait_on_429"]:
            return (False, 0.0)
        max_wait_s = policy["max_wait_s"]
        if max_wait_s <= 0 or retry_after_s > max_wait_s:
            # ponytail: cap exceeded -> fall back immediately, don't wait
            return (False, 0.0)
        return (True, time.time() + float(retry_after_s))

    async def wait_until(self, provider: str, deadline_ts: float) -> bool:
        """
        Block until deadline_ts (or until cancelled). Returns True if the wait
        completed (caller should retry same provider), False if interrupted/timeout.
        """
        remaining = deadline_ts - time.time()
        if remaining <= 0:
            return True
        try:
            await asyncio.sleep(remaining)
            return True
        except asyncio.CancelledError:
            log.warning("wait_until for %s cancelled after %.2fs", provider, remaining)
            return False
        except Exception as e:
            log.error("wait_until for %s failed: %s", provider, e)
            return False
