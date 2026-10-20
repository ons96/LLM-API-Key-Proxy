"""Quota ledger: what each provider has left right now, and whether to wait.

``CooldownPolicy`` reacts to a 429 after it happens. This module holds the state
that makes the decision correct before the failure:

  key      (provider, credential_id, model, window)
  windows  minute | hour | day | month | one_time
  state    limit, used, remaining, reset_at, source, confidence, last_429_at,
           last_ok_at, status
  status   healthy | throttled | exhausted | unknown | disabled

Waiting decision, explicit and unit-tested (``decide``):
  * short window, reset inside ``short_wait_ceiling_s``, and budget remains
    -> WAIT, bounded, retry once
  * long window (daily/monthly/one_time) or reset beyond the ceiling
    -> FALLBACK immediately and mark the key exhausted until reset_at
  * one-time credits -> never wait; consume only when nothing better is eligible

Persistence is SQLite in WAL mode so exhausted state survives a restart. Reset
windows are applied lazily on read. A missing or corrupt ledger degrades to "no
opinion" (every call proceeds) rather than failing requests.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

WINDOWS = ("minute", "hour", "day", "month", "one_time")

WINDOW_SECONDS = {
    "minute": 60,
    "hour": 3600,
    "day": 86400,
    "month": 30 * 86400,
}

SHORT_WINDOWS = {"minute", "hour"}

PROCEED = "proceed"
WAIT = "wait"
FALLBACK = "fallback"

# A 429 whose reset is further out than this is not worth holding a request for.
DEFAULT_SHORT_WAIT_CEILING_S = 60.0
# Hard budget per request so a "bounded wait" can never become a hang.
DEFAULT_MAX_TOTAL_WAIT_S = 45.0

SOURCES = ("header", "config", "observed_429", "assumed")

# Header names, OpenAI-style first, then the vendor spellings actually seen in
# the wild. Values are matched case-insensitively.
_LIMIT_HEADERS = (
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "ratelimit-limit",
    "x-rate-limit-limit",
    "anthropic-ratelimit-requests-limit",
)
_REMAINING_HEADERS = (
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "ratelimit-remaining",
    "x-rate-limit-remaining",
    "anthropic-ratelimit-requests-remaining",
)
_RESET_HEADERS = (
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
    "ratelimit-reset",
    "x-rate-limit-reset",
    "anthropic-ratelimit-requests-reset",
)
_RETRY_AFTER_HEADERS = ("retry-after", "retry-after-ms")


@dataclass
class Decision:
    action: str
    wait_s: float
    window: str
    reason: str
    reset_at: Optional[float] = None

    @property
    def should_wait(self) -> bool:
        return self.action == WAIT

    @property
    def should_fallback(self) -> bool:
        return self.action == FALLBACK


def _now(now: Optional[float] = None) -> float:
    return time.time() if now is None else now


def _window_for_reset(seconds: Optional[float]) -> str:
    if seconds is None or seconds <= 0:
        return "unknown"
    if seconds <= 120:
        return "minute"
    if seconds <= 2 * 3600:
        return "hour"
    if seconds <= 2 * 86400:
        return "day"
    if seconds <= 45 * 86400:
        return "month"
    return "one_time"


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_rate_limit_headers(headers: Any) -> dict[str, Any]:
    """Extract limit/remaining/reset from a response's headers.

    Returns a dict with any of ``limit``, ``remaining``, ``reset_at``,
    ``retry_after_s`` and ``window``. Unknown headers yield an empty dict, which
    callers must treat as "no information", never as zero quota.
    """
    if not headers:
        return {}
    try:
        lowered = {str(k).lower(): v for k, v in dict(headers).items()}
    except Exception:
        return {}

    out: dict[str, Any] = {}

    for name in _LIMIT_HEADERS:
        value = _as_float(lowered.get(name))
        if value is not None:
            out["limit"] = int(value)
            break

    for name in _REMAINING_HEADERS:
        value = _as_float(lowered.get(name))
        if value is not None:
            out["remaining"] = int(value)
            break

    reset_at = None
    for name in _RESET_HEADERS:
        raw = lowered.get(name)
        value = _as_float(raw)
        if value is None:
            continue
        # Values arrive as delta-seconds, a unix timestamp, or a duration string
        # like "1m30s". A number below 1e9 is a delta.
        if value > 1e9:
            reset_at = value
        else:
            reset_at = _now() + value
        break
    if reset_at is not None:
        out["reset_at"] = reset_at

    for name in _RETRY_AFTER_HEADERS:
        value = _as_float(lowered.get(name))
        if value is not None:
            out["retry_after_s"] = value / 1000.0 if name.endswith("-ms") else value
            break

    seconds_out = None
    if "reset_at" in out:
        seconds_out = out["reset_at"] - _now()
    elif "retry_after_s" in out:
        seconds_out = out["retry_after_s"]
    if seconds_out is not None:
        out["window"] = _window_for_reset(seconds_out)
    elif "limit" in out or "remaining" in out:
        # Limit/remaining headers are per-minute buckets in every provider that
        # sends them. The window is an assumption, so confidence stays medium.
        out["window"] = "minute"
        out["window_assumed"] = True
    return out


class QuotaLedger:
    """SQLite-backed quota state. Thread-safe; safe to call when disabled."""

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        *,
        short_wait_ceiling_s: float = DEFAULT_SHORT_WAIT_CEILING_S,
        max_total_wait_s: float = DEFAULT_MAX_TOTAL_WAIT_S,
        enabled: bool = True,
    ) -> None:
        self.db_path = Path(
            db_path
            or os.getenv("QUOTA_LEDGER_DB")
            or "/tmp/llm_proxy_quota_ledger.db"
        )
        self.short_wait_ceiling_s = short_wait_ceiling_s
        self.max_total_wait_s = max_total_wait_s
        self.enabled = enabled and os.getenv("QUOTA_LEDGER_ENABLED", "1") != "0"
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._degraded = not self.enabled

    # -- plumbing -----------------------------------------------------------
    def _connect(self) -> Optional[sqlite3.Connection]:
        if self._degraded:
            return None
        if self._conn is not None:
            return self._conn
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), timeout=5.0, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quota_state (
                    provider TEXT NOT NULL,
                    credential_id TEXT NOT NULL DEFAULT 'default',
                    model TEXT NOT NULL,
                    window TEXT NOT NULL,
                    quota_limit INTEGER,
                    used INTEGER NOT NULL DEFAULT 0,
                    reset_at REAL,
                    source TEXT NOT NULL DEFAULT 'assumed',
                    confidence TEXT NOT NULL DEFAULT 'low',
                    last_429_at REAL,
                    last_ok_at REAL,
                    status TEXT NOT NULL DEFAULT 'unknown',
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (provider, credential_id, model, window)
                )
                """
            )
            conn.commit()
            self._conn = conn
            return conn
        except Exception as exc:
            # Degrade, never fail a request because the ledger is unusable.
            logger.warning(f"quota_ledger: disabled after connect failure: {exc}")
            self._degraded = True
            return None

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    # -- writes -------------------------------------------------------------
    def observe_headers(
        self,
        provider: str,
        model: str,
        headers: Any,
        *,
        credential_id: str = "default",
        now: Optional[float] = None,
    ) -> bool:
        """Record header-derived quota. Returns True when something was learned."""
        parsed = parse_rate_limit_headers(headers)
        if not parsed:
            return False
        window = parsed.get("window") or "unknown"
        if window == "unknown":
            return False
        now = _now(now)
        remaining = parsed.get("remaining")
        limit = parsed.get("limit")
        used = None
        if limit is not None and remaining is not None:
            used = max(0, limit - remaining)
        reset_at = parsed.get("reset_at")
        if reset_at is None and parsed.get("retry_after_s") is not None:
            reset_at = now + float(parsed["retry_after_s"])

        status = "healthy"
        if remaining is not None:
            if remaining <= 0:
                status = "exhausted"
            elif limit and remaining <= max(1, limit * 0.1):
                status = "throttled"
        self._upsert(
            provider,
            credential_id,
            model,
            window,
            limit=limit,
            used=used,
            reset_at=reset_at,
            source="header",
            confidence="medium" if parsed.get("window_assumed") else "high",
            status=status,
            now=now,
        )
        return True

    def record_ok(
        self,
        provider: str,
        model: str,
        *,
        credential_id: str = "default",
        window: str = "minute",
        now: Optional[float] = None,
    ) -> None:
        now = _now(now)
        row = self._get(provider, credential_id, model, window)
        used = (row["used"] if row else 0) + 1
        self._upsert(
            provider,
            credential_id,
            model,
            window,
            limit=row["quota_limit"] if row else None,
            used=used,
            reset_at=row["reset_at"] if row else None,
            source=row["source"] if row else "observed_429",
            confidence=row["confidence"] if row else "low",
            status="healthy",
            now=now,
            last_ok_at=now,
        )

    def record_429(
        self,
        provider: str,
        model: str,
        *,
        retry_after_s: Optional[float] = None,
        reset_at: Optional[float] = None,
        credential_id: str = "default",
        now: Optional[float] = None,
    ) -> dict[str, Any]:
        """Record a throttle. Chooses the window from the observed reset.

        A 429 with a short reset goes to the minute window (waitable). A 429 with
        a long or unknown reset goes to the day window and marks it exhausted, so
        the next request falls back immediately instead of retrying.
        """
        now = _now(now)
        if reset_at is None and retry_after_s:
            reset_at = now + float(retry_after_s)
        seconds_out = (reset_at - now) if reset_at else None
        window = _window_for_reset(seconds_out)
        if window == "unknown":
            window = "day"  # fail towards fallback, not towards waiting forever
        status = "exhausted" if window not in SHORT_WINDOWS else "throttled"
        self._upsert(
            provider,
            credential_id,
            model,
            window,
            limit=None,
            used=None,
            reset_at=reset_at,
            source="observed_429",
            confidence="medium",
            status=status,
            now=now,
            last_429_at=now,
        )
        return {
            "provider": provider,
            "model": model,
            "window": window,
            "reset_at": reset_at,
            "status": status,
        }

    def _upsert(
        self,
        provider: str,
        credential_id: str,
        model: str,
        window: str,
        *,
        limit: Optional[int],
        used: Optional[int],
        reset_at: Optional[float],
        source: str,
        confidence: str,
        status: str,
        now: float,
        last_429_at: Optional[float] = None,
        last_ok_at: Optional[float] = None,
    ) -> None:
        if self._degraded:
            return
        with self._lock:
            conn = self._connect()
            if conn is None:
                return
            try:
                existing = conn.execute(
                    "SELECT * FROM quota_state WHERE provider=? AND credential_id=?"
                    " AND model=? AND window=?",
                    (provider, credential_id, model, window),
                ).fetchone()
                if existing is None:
                    conn.execute(
                        "INSERT INTO quota_state(provider, credential_id, model, window,"
                        " quota_limit, used, reset_at, source, confidence, last_429_at,"
                        " last_ok_at, status, updated_at)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            provider,
                            credential_id,
                            model,
                            window,
                            limit,
                            used or 0,
                            reset_at,
                            source,
                            confidence,
                            last_429_at,
                            last_ok_at,
                            status,
                            now,
                        ),
                    )
                else:
                    conn.execute(
                        "UPDATE quota_state SET quota_limit=COALESCE(?, quota_limit),"
                        " used=COALESCE(?, used), reset_at=COALESCE(?, reset_at),"
                        " source=?, confidence=?, last_429_at=COALESCE(?, last_429_at),"
                        " last_ok_at=COALESCE(?, last_ok_at), status=?, updated_at=?"
                        " WHERE provider=? AND credential_id=? AND model=? AND window=?",
                        (
                            limit,
                            used,
                            reset_at,
                            source,
                            confidence,
                            last_429_at,
                            last_ok_at,
                            status,
                            now,
                            provider,
                            credential_id,
                            model,
                            window,
                        ),
                    )
                conn.commit()
            except Exception as exc:
                logger.warning(f"quota_ledger: write failed for {provider}/{model}: {exc}")
                self._degraded = True

    # -- reads --------------------------------------------------------------
    def _get(
        self, provider: str, credential_id: str, model: str, window: str
    ) -> Optional[dict[str, Any]]:
        if self._degraded:
            return None
        with self._lock:
            conn = self._connect()
            if conn is None:
                return None
            try:
                cur = conn.execute(
                    "SELECT * FROM quota_state WHERE provider=? AND credential_id=?"
                    " AND model=? AND window=?",
                    (provider, credential_id, model, window),
                )
                cur.row_factory = sqlite3.Row
                row = cur.fetchone()
                return dict(row) if row else None
            except Exception:
                return None

    def snapshot(
        self,
        provider: str,
        model: str,
        *,
        credential_id: str = "default",
        now: Optional[float] = None,
    ) -> dict[str, Any]:
        """Current state with expired windows reset lazily on read."""
        now = _now(now)
        out: dict[str, Any] = {"windows": {}, "status": "unknown"}
        worst = "healthy"
        for window in WINDOWS:
            row = self._get(provider, credential_id, model, window)
            if not row:
                continue
            reset_at = row.get("reset_at")
            status = row.get("status") or "unknown"
            if reset_at and now >= reset_at:
                status = "healthy"  # window rolled over
                reset_at = None
            limit = row.get("quota_limit")
            used = row.get("used") or 0
            entry = {
                "limit": limit,
                "used": used,
                "remaining": (limit - used) if limit is not None else None,
                "reset_at": reset_at,
                "source": row.get("source"),
                "confidence": row.get("confidence"),
                "last_429_at": row.get("last_429_at"),
                "last_ok_at": row.get("last_ok_at"),
                "status": status,
            }
            out["windows"][window] = entry
            if status == "exhausted":
                worst = "exhausted"
            elif status == "throttled" and worst != "exhausted":
                worst = "throttled"
        out["status"] = worst if out["windows"] else "unknown"
        return out

    def decide(
        self,
        provider: str,
        model: str,
        *,
        credential_id: str = "default",
        now: Optional[float] = None,
        is_one_time: bool = False,
    ) -> Decision:
        """WAIT, FALLBACK or PROCEED for one (provider, model)."""
        if self._degraded:
            return Decision(PROCEED, 0.0, "unknown", "ledger disabled")
        now = _now(now)
        state = self.snapshot(provider, model, credential_id=credential_id, now=now)
        if self._degraded:
            # The read itself failed; say so instead of pretending there is no
            # quota pressure.
            return Decision(PROCEED, 0.0, "unknown", "ledger disabled")

        for window, entry in state["windows"].items():
            reset_at = entry.get("reset_at")
            seconds_out = (reset_at - now) if reset_at else None
            if entry["status"] == "exhausted":
                if is_one_time or window not in SHORT_WINDOWS:
                    return Decision(
                        FALLBACK,
                        0.0,
                        window,
                        f"{window} window exhausted",
                        reset_at,
                    )
                if seconds_out is not None and seconds_out <= self.short_wait_ceiling_s:
                    return Decision(
                        WAIT,
                        min(seconds_out, self.max_total_wait_s),
                        window,
                        "short-window exhaustion, bounded wait",
                        reset_at,
                    )
                return Decision(FALLBACK, 0.0, window, "reset beyond wait ceiling", reset_at)

            if entry["status"] == "throttled":
                if is_one_time:
                    return Decision(FALLBACK, 0.0, window, "one-time credit never waits", reset_at)
                if seconds_out is not None and seconds_out <= self.short_wait_ceiling_s:
                    return Decision(
                        WAIT,
                        min(seconds_out, self.max_total_wait_s),
                        window,
                        "throttled with a short reset",
                        reset_at,
                    )
                return Decision(FALLBACK, 0.0, window, "throttled, reset too far out", reset_at)

        return Decision(PROCEED, 0.0, "none", "no quota pressure recorded")

    def exhausted_providers(
        self, *, now: Optional[float] = None
    ) -> set[str]:
        """Providers whose day/one_time windows are currently exhausted."""
        now = _now(now)
        if self._degraded:
            return set()
        with self._lock:
            conn = self._connect()
            if conn is None:
                return set()
            try:
                rows = conn.execute(
                    "SELECT provider FROM quota_state WHERE status='exhausted'"
                    " AND window IN ('day','month','one_time')"
                    " AND (reset_at IS NULL OR reset_at > ?)",
                    (now,),
                ).fetchall()
            except Exception:
                return set()
        return {str(r[0]) for r in rows}

    def filter_candidates(
        self,
        candidates: Iterable[tuple[str, str]],
        *,
        now: Optional[float] = None,
    ) -> list[tuple[str, str]]:
        """Keep candidates that are not currently exhausted."""
        now = _now(now)
        kept: list[tuple[str, str]] = []
        for provider, model in candidates:
            decision = self.decide(provider, model, now=now)
            if decision.should_fallback:
                logger.debug(
                    f"quota_ledger: skip {provider}/{model} ({decision.reason})"
                )
                continue
            kept.append((provider, model))
        return kept


_DEFAULT_LEDGER: Optional[QuotaLedger] = None


def get_ledger() -> QuotaLedger:
    """Process-wide ledger. Lazy so importing this module costs nothing."""
    global _DEFAULT_LEDGER
    if _DEFAULT_LEDGER is None:
        _DEFAULT_LEDGER = QuotaLedger()
    return _DEFAULT_LEDGER
