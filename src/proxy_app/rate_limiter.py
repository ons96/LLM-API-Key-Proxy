import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitStatus:
    """Status of rate limits for a provider/model."""

    requests_this_minute: int = 0
    requests_today: int = 0
    active_requests: int = 0
    last_reset_minute: float = field(default_factory=time.time)
    last_reset_day: float = field(default_factory=time.time)
    rate_limited_until: float = 0.0


class RateLimitTracker:
    """
    Tracks rate limits and usage for providers.
    Thread-safe (async-safe) implementation.
    """

    def __init__(self):
        self._usage: Dict[str, RateLimitStatus] = {}
        self._lock = asyncio.Lock()

    def _get_key(self, provider: str, model: str) -> str:
        return f"{provider}/{model}"

    async def can_use_provider(
        self, provider: str, model: str, limits: Dict[str, int]
    ) -> bool:
        """
        Check if a provider can be used based on limits.
        limits dict can contain: 'rpm', 'daily'
        """
        key = self._get_key(provider, model)

        async with self._lock:
            if key not in self._usage:
                self._usage[key] = RateLimitStatus()

            status = self._usage[key]
            now = time.time()

            # Check hard rate limit backoff
            if now < status.rate_limited_until:
                return False

            # Reset counters if windows have passed
            if now - status.last_reset_minute > 60:
                status.requests_this_minute = 0
                status.last_reset_minute = now

            # Approximate day reset (24h sliding window start or fixed check)
            # Simple 24h reset for now
            if now - status.last_reset_day > 86400:
                status.requests_today = 0
                status.last_reset_day = now

            # Check Concurrency
            concurrency_limit = limits.get("concurrency")
            if concurrency_limit and status.active_requests >= concurrency_limit:
                logger.debug(
                    f"Concurrency limit hit for {key}: {status.active_requests}/{concurrency_limit}"
                )
                return False

            # Check RPM
            rpm_limit = limits.get("rpm")
            if rpm_limit and status.requests_this_minute >= rpm_limit:
                logger.debug(
                    f"Rate limit hit (RPM) for {key}: {status.requests_this_minute}/{rpm_limit}"
                )
                return False

            # Check Daily
            daily_limit = limits.get("daily")
            if daily_limit and status.requests_today >= daily_limit:
                logger.debug(
                    f"Rate limit hit (Daily) for {key}: {status.requests_today}/{daily_limit}"
                )
                return False

            return True

    async def record_request(self, provider: str, model: str):
        """Record a request attempt."""
        key = self._get_key(provider, model)
        async with self._lock:
            if key not in self._usage:
                self._usage[key] = RateLimitStatus()

            self._usage[key].requests_this_minute += 1
            self._usage[key].requests_today += 1
            self._usage[key].active_requests += 1

    async def release_request(self, provider: str, model: str):
        """Record that a request has completed."""
        key = self._get_key(provider, model)
        async with self._lock:
            if key in self._usage:
                self._usage[key].active_requests = max(0, self._usage[key].active_requests - 1)

    async def record_rate_limit_hit(
        self, provider: str, model: str, retry_after: float = 60.0
    ):
        """Record a 429/Rate Limit error."""
        key = self._get_key(provider, model)
        async with self._lock:
            if key not in self._usage:
                self._usage[key] = RateLimitStatus()

            self._usage[key].rate_limited_until = time.time() + retry_after
            logger.warning(
                f"Provider {key} rate limited until {self._usage[key].rate_limited_until}"
            )

    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        async with self._lock:
            return {
                k: {
                    "rpm": v.requests_this_minute,
                    "daily": v.requests_today,
                    "limited": time.time() < v.rate_limited_until,
                }
                for k, v in self._usage.items()
            }
