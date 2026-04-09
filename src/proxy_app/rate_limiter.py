import time
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Standardized fallback reason codes for logging and debugging.
REASON_RATE_LIMIT = "RATE_LIMIT"
REASON_USAGE_CAP = "USAGE_CAP"
REASON_AUTH_FAIL = "AUTH_FAIL"
REASON_PROVIDER_DOWN = "PROVIDER_DOWN"
REASON_MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
REASON_TIMEOUT = "TIMEOUT"


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
    """

    def __init__(self):
        self._usage: Dict[str, RateLimitStatus] = {}
        self._lock = asyncio.Lock()
        # Provider-level reset windows: provider_name -> {"daily_reset_utc": "00:00", "monthly_reset_day": 1}
        self._reset_windows: Dict[str, Dict[str, Any]] = {}

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

            # Check hard rate limit backoff
            if now < status.rate_limited_until:
                return False, status.block_reason or REASON_RATE_LIMIT

            # Reset counters if windows have passed
            if now - status.last_reset_minute > 60:
                status.requests_this_minute = 0
                status.tokens_this_minute = 0
                status.last_reset_minute = now

            # Check if daily counter should reset based on known reset window
            prov_name = self._provider_from_key(key)
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
                    f"Rate limit hit (RPM) for {key}: {status.requests_this_minute}/{rpm_limit}"
                )
                return False, REASON_RATE_LIMIT

            # Check TPM
            tpm_limit = limits.get("tpm")
            if tpm_limit and status.tokens_this_minute >= tpm_limit:
                logger.debug(
                    f"Rate limit hit (TPM) for {key}: {status.tokens_this_minute}/{tpm_limit}"
                )
                return False, REASON_RATE_LIMIT

            # Check Daily Requests
            daily_limit = limits.get("daily")
            if daily_limit and status.requests_today >= daily_limit:
                logger.debug(
                    f"Rate limit hit (Daily) for {key}: {status.requests_today}/{daily_limit}"
                )
                return False, REASON_USAGE_CAP

            # Check Daily Tokens
            daily_tokens_limit = limits.get("daily_tokens")
            if daily_tokens_limit and status.tokens_today >= daily_tokens_limit:
                logger.debug(
                    f"Rate limit hit (Daily Tokens) for {key}: {status.tokens_today}/{daily_tokens_limit}"
                )
                return False, REASON_USAGE_CAP

            return True, ""

    async def record_request(self, provider: str, model: str):
        """Record a request attempt."""
        key = self._get_key(provider, model)
        async with self._lock:
            if key not in self._usage:
                self._usage[key] = RateLimitStatus()

            self._usage[key].requests_this_minute += 1
            self._usage[key].requests_today += 1

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
        """Get current usage statistics."""
        now = time.time()
        async with self._lock:
            return {
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
