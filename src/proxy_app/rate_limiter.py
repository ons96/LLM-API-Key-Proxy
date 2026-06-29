import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date as date_cls
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

try:
    from zoneinfo import ZoneInfo

    _HAS_ZONEINFO = True
except ImportError:  # pragma: no cover - Python < 3.9
    _HAS_ZONEINFO = False

logger = logging.getLogger(__name__)

# Standardized fallback reason codes for logging and debugging.
REASON_RATE_LIMIT = "RATE_LIMIT"
REASON_USAGE_CAP = "USAGE_CAP"
REASON_AUTH_FAIL = "AUTH_FAIL"
REASON_PROVIDER_DOWN = "PROVIDER_DOWN"
REASON_MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
REASON_TIMEOUT = "TIMEOUT"

# Active-day tracking constants
ACTIVE_DAYS_FILE_VERSION = 1
DEFAULT_ACTIVE_DAYS_WINDOW_DAYS = 12
DEFAULT_ACTIVE_DAYS_MAX = 3


def _parse_reset_time_utc(time_str: str) -> datetime:
    """Parse a 'HH:MM' UTC time string into the next occurrence of that time."""
    parts = time_str.strip().split(":")
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0
    now = datetime.now(timezone.utc)
    reset = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if reset <= now:
        reset += timedelta(days=1)
    return reset


def _resolve_tz(tz_name: str):
    """Return a tzinfo for the given name; fall back to UTC if unavailable."""
    if not tz_name or str(tz_name).upper() == "UTC":
        return timezone.utc
    if _HAS_ZONEINFO:
        try:
            return ZoneInfo(tz_name)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                f"Unknown timezone '{tz_name}', falling back to UTC: {e}"
            )
    return timezone.utc


def _today_in_tz(tz) -> date_cls:
    """Return the calendar date for the given tz (UTC fallback)."""
    try:
        return datetime.now(tz).date()
    except Exception:  # pragma: no cover - defensive
        return datetime.now(timezone.utc).date()


@dataclass
class RateLimitStatus:
    """Status of rate limits for a provider/model."""

    requests_this_minute: int = 0
    requests_today: int = 0
    tokens_this_minute: int = 0
    tokens_today: int = 0
    last_reset_minute: float = field(default_factory=time.time)
    last_reset_day: float = field(default_factory=time.time)
    rate_limited_until: float = 0.0
    block_reason: str = ""


class RateLimitTracker:
    """
    Tracks rate limits and usage for providers.
    Thread-safe (async-safe) implementation.

    Also tracks per-provider "active days" — distinct calendar dates on which a
    provider was used. When a provider has a configured active_days_window and
    the number of active days inside the window exceeds the configured cap, the
    provider is treated as usage-capped and blocked until the oldest active day
    rolls off the window. Persistence is JSON at ``data/active_days.json`` by
    default (path injectable for tests).
    """

    def __init__(self, active_days_path: Optional[Path] = None):
        self._usage: Dict[str, RateLimitStatus] = {}
        self._lock = asyncio.Lock()
        # Provider-level reset windows: provider_name -> {"daily_reset_utc": "00:00", ...}
        self._reset_windows: Dict[str, Dict[str, Any]] = {}
        # Active-days configuration: provider -> {"window_days", "max_active_days", "tz", "tzinfo"}
        self._active_days_windows: Dict[str, Dict[str, Any]] = {}
        # Active-days state: provider -> set of YYYY-MM-DD strings
        self._active_days: Dict[str, Set[str]] = {}
        # Persistence path. Default: <repo_root>/data/active_days.json
        if active_days_path is None:
            repo_root = Path(__file__).resolve().parent.parent.parent
            active_days_path = repo_root / "data" / "active_days.json"
        self._active_days_path: Optional[Path] = active_days_path
        self._load_active_days()

    # ------------------------------------------------------------------ #
    # Active-days configuration + persistence                            #
    # ------------------------------------------------------------------ #

    def _load_active_days(self) -> None:
        """Load persisted active-day state from disk if present."""
        if not self._active_days_path or not self._active_days_path.exists():
            return
        try:
            with open(self._active_days_path, "r") as f:
                data = json.load(f)
            providers = data.get("providers", {}) or {}
            loaded = 0
            for pname, pdata in providers.items():
                if not isinstance(pdata, dict):
                    continue
                dates = pdata.get("dates", []) or []
                cleaned: Set[str] = {
                    d for d in dates if isinstance(d, str) and len(d) == 10
                }
                if cleaned:
                    self._active_days[pname] = cleaned
                    loaded += 1
            if loaded:
                logger.info(
                    f"Loaded active days for {loaded} providers from "
                    f"{self._active_days_path}"
                )
        except Exception as e:
            logger.warning(
                f"Failed to load active days from {self._active_days_path}: {e}"
            )

    def _save_active_days(self) -> None:
        """Atomically persist active-day state to disk. No-op if path unset."""
        if not self._active_days_path:
            return
        try:
            self._active_days_path.parent.mkdir(parents=True, exist_ok=True)
            payload: Dict[str, Any] = {
                "version": ACTIVE_DAYS_FILE_VERSION,
                "providers": {
                    pname: {"dates": sorted(dates)}
                    for pname, dates in self._active_days.items()
                    if dates
                },
            }
            tmp = self._active_days_path.with_suffix(
                self._active_days_path.suffix + ".tmp"
            )
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            os.replace(tmp, self._active_days_path)
        except Exception as e:
            logger.warning(
                f"Failed to save active days to {self._active_days_path}: {e}"
            )

    def configure_active_days_windows(
        self, windows: Dict[str, Dict[str, Any]]
    ) -> None:
        """Configure per-provider active-day window caps.

        Example::

            {"g4f": {"window_days": 12, "max_active_days": 3, "tz": "UTC"}}
        """
        self._active_days_windows = {}
        for pname, cfg in (windows or {}).items():
            if not isinstance(cfg, dict):
                continue
            wd = int(cfg.get("window_days", DEFAULT_ACTIVE_DAYS_WINDOW_DAYS))
            md = int(cfg.get("max_active_days", DEFAULT_ACTIVE_DAYS_MAX))
            tz_name = str(cfg.get("tz", "UTC"))
            self._active_days_windows[pname] = {
                "window_days": wd,
                "max_active_days": md,
                "tz": tz_name,
                "tzinfo": _resolve_tz(tz_name),
            }
        logger.info(
            f"Configured active-days windows for "
            f"{len(self._active_days_windows)} providers"
        )

    def _prune_active_days(
        self, provider: str, today: date_cls
    ) -> Set[str]:
        """Drop active dates older than (today - window_days + 1)."""
        cfg = self._active_days_windows.get(provider)
        if not cfg:
            return set()
        window_days = cfg["window_days"]
        cutoff_iso = (today - timedelta(days=window_days - 1)).isoformat()
        existing = self._active_days.setdefault(provider, set())
        to_remove = {d for d in existing if d < cutoff_iso}
        if to_remove:
            existing -= to_remove
        return existing

    def check_active_days(
        self, provider: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Check whether a provider may be used under its active-day window.

        Returns ``(allowed, reason, info)``. ``info`` is a dict with::

            {
                "provider", "window_days", "max_active_days", "tz",
                "used", "limit", "remaining",
                "active_dates": [...],
                "oldest_active_date": "YYYY-MM-DD" or None,
                "next_slot_unlocks_on": ISO8601 string or None,
            }
        """
        cfg = self._active_days_windows.get(provider)
        if not cfg:
            return True, "", {}
        tz = cfg["tzinfo"]
        today = _today_in_tz(tz)
        pruned = self._prune_active_days(provider, today)
        used = len(pruned)
        limit = cfg["max_active_days"]
        info: Dict[str, Any] = {
            "provider": provider,
            "window_days": cfg["window_days"],
            "max_active_days": limit,
            "tz": cfg["tz"],
            "used": used,
            "limit": limit,
            "remaining": max(0, limit - used),
            "active_dates": sorted(pruned),
            "oldest_active_date": min(pruned) if pruned else None,
            "next_slot_unlocks_on": None,
        }
        if used >= limit and pruned:
            oldest = date_cls.fromisoformat(min(pruned))
            unlock_date = oldest + timedelta(days=cfg["window_days"])
            unlock_dt = datetime.combine(
                unlock_date, dt_time(0, 0, 0), tzinfo=tz
            )
            info["next_slot_unlocks_on"] = unlock_dt.isoformat()
            return False, REASON_USAGE_CAP, info
        return True, "", info

    async def record_active_day(
        self, provider: str, date_str: Optional[str] = None
    ) -> bool:
        """Record that the provider was used on ``date_str`` (default: today in tz).

        Returns True if the date was newly added to the set, False if it was
        already present (or the provider is not configured for active-day
        tracking). Persists to disk only when the set changes.
        """
        cfg = self._active_days_windows.get(provider)
        if not cfg:
            return False
        if date_str is None:
            today = _today_in_tz(cfg["tzinfo"])
            date_str = today.isoformat()
        existing = self._active_days.setdefault(provider, set())
        if date_str in existing:
            return False
        existing.add(date_str)
        # Opportunistic prune on record
        today = _today_in_tz(cfg["tzinfo"])
        cutoff_iso = (
            today - timedelta(days=cfg["window_days"] - 1)
        ).isoformat()
        to_remove = {d for d in existing if d < cutoff_iso}
        if to_remove:
            existing -= to_remove
        self._save_active_days()
        return True

    def get_active_days_info(self, provider: str) -> Dict[str, Any]:
        """Public read-only snapshot of active-day info for a provider."""
        _, _, info = self.check_active_days(provider)
        return info

    def list_active_days_providers(self) -> Dict[str, Dict[str, Any]]:
        """Return a snapshot of active-day state for all known providers."""
        result: Dict[str, Dict[str, Any]] = {}
        providers = set(self._active_days_windows.keys()) | set(
            self._active_days.keys()
        )
        for pname in sorted(providers):
            info = self.get_active_days_info(pname)
            cfg = self._active_days_windows.get(pname, {})
            result[pname] = {
                "configured": bool(cfg),
                **info,
            }
        return result

    # ------------------------------------------------------------------ #
    # Existing rate-limit logic                                          #
    # ------------------------------------------------------------------ #

    def configure_reset_windows(self, windows: Dict[str, Dict[str, Any]]):
        """Set known reset windows for providers.

        Example:
            {"groq": {"daily_reset_utc": "00:00"},
             "together": {"daily_reset_utc": "00:00"},
             "gemini": {"daily_reset_utc": "07:00"}}
        """
        self._reset_windows = dict(windows)
        logger.info(f"Configured reset windows for {len(windows)} providers")

    def _get_key(self, provider: str, model: str) -> str:
        return f"{provider}/{model}"

    def _provider_from_key(self, key: str) -> str:
        return key.split("/", 1)[0] if "/" in key else key

    async def can_use_provider(
        self, provider: str, model: str, limits: Dict[str, int]
    ) -> Tuple[bool, str]:
        """
        Check if a provider can be used based on limits.
        limits dict can contain: 'rpm', 'daily', 'tpm', 'daily_tokens'

        Returns (allowed, reason_code) where reason_code is empty if allowed.
        """
        key = self._get_key(provider, model)

        async with self._lock:
            if key not in self._usage:
                self._usage[key] = RateLimitStatus()

            status = self._usage[key]
            now = time.time()
            prov_name = self._provider_from_key(key)

            # Check hard rate limit backoff
            if now < status.rate_limited_until:
                return False, status.block_reason or REASON_RATE_LIMIT

            # Active-days window check (on-the-fly; must run under lock so
            # pruning + check are consistent with the in-memory set). If the
            # window has rolled since the last call, this naturally recovers
            # without needing to clear any cached backoff. Active-day state is
            # intentionally NOT written to status.rate_limited_until: that
            # field is reserved for record_rate_limit_hit-set cooldowns (e.g.
            # upstream 429 responses). Active-day cap is in
            # get_usage_stats["active_days"] and recomputed per call.
            ad_allowed, ad_reason, ad_info = self.check_active_days(prov_name)
            if not ad_allowed:
                logger.debug(
                    f"Provider {prov_name} blocked by active-day window "
                    f"({ad_info.get('used')}/{ad_info.get('limit')}); "
                    f"unlocks {ad_info.get('next_slot_unlocks_on')}"
                )
                return False, ad_reason

            # Reset counters if windows have passed
            if now - status.last_reset_minute > 60:
                status.requests_this_minute = 0
                status.tokens_this_minute = 0
                status.last_reset_minute = now

            # Check if daily counter should reset based on known reset window
            reset_window = self._reset_windows.get(prov_name, {})
            daily_reset_utc = reset_window.get("daily_reset_utc")

            if daily_reset_utc:
                # Use provider's known reset time instead of sliding 24h window
                next_reset = _parse_reset_time_utc(daily_reset_utc)
                prev_reset = next_reset - timedelta(days=1)
                reset_epoch = prev_reset.timestamp()
                if status.last_reset_day < reset_epoch:
                    status.requests_today = 0
                    status.tokens_today = 0
                    status.last_reset_day = now
            else:
                # Fallback: simple 24h sliding window
                if now - status.last_reset_day > 86400:
                    status.requests_today = 0
                    status.tokens_today = 0
                    status.last_reset_day = now

            # Check RPM
            rpm_limit = limits.get("rpm")
            if rpm_limit and status.requests_this_minute >= rpm_limit:
                logger.debug(
                    f"Rate limit hit (RPM) for {key}: "
                    f"{status.requests_this_minute}/{rpm_limit}"
                )
                return False, REASON_RATE_LIMIT

            # Check TPM
            tpm_limit = limits.get("tpm")
            if tpm_limit and status.tokens_this_minute >= tpm_limit:
                logger.debug(
                    f"Rate limit hit (TPM) for {key}: "
                    f"{status.tokens_this_minute}/{tpm_limit}"
                )
                return False, REASON_RATE_LIMIT

            # Check Daily Requests
            daily_limit = limits.get("daily")
            if daily_limit and status.requests_today >= daily_limit:
                logger.debug(
                    f"Rate limit hit (Daily) for {key}: "
                    f"{status.requests_today}/{daily_limit}"
                )
                return False, REASON_USAGE_CAP

            # Check Daily Tokens
            daily_tokens_limit = limits.get("daily_tokens")
            if daily_tokens_limit and status.tokens_today >= daily_tokens_limit:
                logger.debug(
                    f"Rate limit hit (Daily Tokens) for {key}: "
                    f"{status.tokens_today}/{daily_tokens_limit}"
                )
                return False, REASON_USAGE_CAP

            return True, ""

    async def record_request(self, provider: str, model: str):
        """Record a request attempt and (if configured) today's active-day stamp."""
        key = self._get_key(provider, model)
        async with self._lock:
            if key not in self._usage:
                self._usage[key] = RateLimitStatus()

            self._usage[key].requests_this_minute += 1
            self._usage[key].requests_today += 1
            # Stamp active day for this provider (no-op if not configured)
            await self.record_active_day(provider)

    async def record_tokens(self, provider: str, model: str, token_count: int):
        """Record tokens used in a request."""
        key = self._get_key(provider, model)
        async with self._lock:
            if key not in self._usage:
                self._usage[key] = RateLimitStatus()

            self._usage[key].tokens_this_minute += token_count
            self._usage[key].tokens_today += token_count

    async def record_rate_limit_hit(
        self,
        provider: str,
        model: str,
        retry_after: float = 60.0,
        reason: str = REASON_RATE_LIMIT,
    ):
        """Record a 429/Rate Limit error.

        If the provider has a known daily reset window and the reason is
        USAGE_CAP, block until the reset time instead of the default
        retry_after period.
        """
        key = self._get_key(provider, model)
        prov_name = self._provider_from_key(key)

        async with self._lock:
            if key not in self._usage:
                self._usage[key] = RateLimitStatus()

            block_until = time.time() + retry_after

            # For usage cap hits, use the known reset window if available
            if reason == REASON_USAGE_CAP:
                reset_window = self._reset_windows.get(prov_name, {})
                daily_reset_utc = reset_window.get("daily_reset_utc")
                if daily_reset_utc:
                    next_reset = _parse_reset_time_utc(daily_reset_utc)
                    # Add a small buffer (30s) past the reset time
                    block_until = next_reset.timestamp() + 30
                    logger.info(
                        f"Provider {key} hit usage cap, blocking until "
                        f"daily reset at {daily_reset_utc} UTC "
                        f"({next_reset.isoformat()})"
                    )

            self._usage[key].rate_limited_until = block_until
            self._usage[key].block_reason = reason
            logger.warning(
                f"Provider {key} blocked ({reason}) until "
                f"{datetime.fromtimestamp(block_until, tz=timezone.utc).isoformat()}"
            )

    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics, including active-day state per provider."""
        now = time.time()
        async with self._lock:
            usage = {
                k: {
                    "rpm": v.requests_this_minute,
                    "daily": v.requests_today,
                    "limited": now < v.rate_limited_until,
                    "block_reason": v.block_reason
                    if now < v.rate_limited_until
                    else "",
                    "blocked_until": (
                        datetime.fromtimestamp(
                            v.rate_limited_until, tz=timezone.utc
                        ).isoformat()
                        if now < v.rate_limited_until
                        else None
                    ),
                }
                for k, v in self._usage.items()
            }
            active_days = self.list_active_days_providers()
        return {"usage": usage, "active_days": active_days}
