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
    Per-provider 429 wait-vs-fallback policy (#218).

    For cache-supporting free-credit providers (anthropic, openai, gemini),
    waiting out a 429 rate-limit preserves prompt cache benefit. For other
    providers, the default is to rotate immediately (legacy behavior).
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        if config_path is None:
            # ponytail: resolve relative to this file so tests + prod both work
            config_path = str(Path(__file__).resolve().parents[2] / "config" / "cooldown_policy.yaml")
        self._policies: Dict[str, dict] = {}
        self._load(config_path)

    def _load(self, config_path: str) -> None:
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f) or {}
            providers = data.get("providers", {}) or {}
            self._default = data.get("default", _DEFAULT_POLICY) or _DEFAULT_POLICY
            # ponytail: normalize keys to lowercase, drop missing fields
            for name, cfg in providers.items():
                if not isinstance(cfg, dict):
                    continue
                self._policies[name.lower()] = {
                    "wait_on_429": bool(cfg.get("wait_on_429", False)),
                    "max_wait_s": int(cfg.get("max_wait_s", 0) or 0),
                    "reason": str(cfg.get("reason", "configured")),
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
            self._default = _DEFAULT_POLICY.copy()
        except Exception as e:
            log.error("failed to load cooldown_policy.yaml: %s — using default (no wait)", e)
            self._policies = {}
            self._default = _DEFAULT_POLICY.copy()

    def get_policy(self, provider: str) -> dict:
        """Return merged policy for a provider (falls back to default)."""
        return self._policies.get(provider.lower(), self._default)

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
