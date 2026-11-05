"""Read-only provider health snapshots for the sidecar compatibility surface.

The legacy gateway owns the live probes and writes `provider_status.db`. The
sidecar can expose those rows while it is running beside the gateway, but it
must label stale data rather than presenting a historical snapshot as current.
"""

from __future__ import annotations

from contextlib import closing
import csv
from datetime import datetime, timedelta, timezone
import io
import math
from pathlib import Path
import sqlite3


class ProviderStatusError(RuntimeError):
    """A bounded status-read failure without provider secrets or row contents."""


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _age_seconds(timestamp: datetime | None, now: datetime) -> float | None:
    if timestamp is None:
        return None
    return max(0.0, (now - timestamp).total_seconds())


class ProviderStatusStore:
    """Read provider health rows without opening the gateway or mutating its DB."""

    def __init__(self, path: str | Path, *, stale_after_seconds: int = 900) -> None:
        database = Path(path).expanduser().resolve()
        if not database.is_file():
            raise FileNotFoundError(database)
        if (
            isinstance(stale_after_seconds, bool)
            or not isinstance(stale_after_seconds, int)
            or stale_after_seconds < 1
        ):
            raise ValueError("stale_after_seconds must be a positive integer")
        self.path = database
        self.stale_after_seconds = stale_after_seconds
        self.source = "provider_status.db"

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def _latest_rows(self) -> list[sqlite3.Row]:
        try:
            with closing(self._connect()) as connection:
                return connection.execute("""SELECT h.* FROM provider_health_checks AS h
                       WHERE h.id = (
                         SELECT h2.id FROM provider_health_checks AS h2
                         WHERE h2.provider_name = h.provider_name
                         ORDER BY h2.last_check_timestamp DESC, h2.id DESC
                         LIMIT 1
                       )
                       ORDER BY h.provider_name""").fetchall()
        except sqlite3.Error as exc:
            raise ProviderStatusError(
                "provider status database is unavailable"
            ) from exc

    @staticmethod
    def _provider(row: sqlite3.Row, now: datetime, stale_after_seconds: int) -> dict:
        checked = _timestamp(row["last_check_timestamp"])
        age = _age_seconds(checked, now)
        stale = age is None or age > stale_after_seconds
        return {
            "provider_name": row["provider_name"],
            "status": row["status"],
            "response_time_ms": row["response_time_ms"],
            "uptime_percent": row["uptime_percent"],
            "rate_limit_percent": row["rate_limit_percent"],
            "last_check": row["last_check_timestamp"],
            "error_message": row["error_message"],
            "consecutive_failures": row["consecutive_failures"],
            "age_seconds": None if age is None else round(age, 1),
            "stale": stale,
        }

    def current_status(self, now: datetime | None = None) -> dict:
        current = now or datetime.now(timezone.utc)
        rows = self._latest_rows()
        providers = {
            row["provider_name"]: self._provider(row, current, self.stale_after_seconds)
            for row in rows
        }
        return {
            "timestamp": current.isoformat(),
            "source": self.source,
            "read_only": True,
            "stale_after_seconds": self.stale_after_seconds,
            "stale": not providers or all(item["stale"] for item in providers.values()),
            "providers": providers,
        }

    def single(self, provider_name: str, now: datetime | None = None) -> dict | None:
        if (
            not isinstance(provider_name, str)
            or not provider_name
            or len(provider_name) > 512
        ):
            return None
        status = self.current_status(now)
        provider = status["providers"].get(provider_name)
        if provider is None:
            return None
        return {**provider, "provider_name": provider_name}

    def history(
        self, provider_name: str, hours: int = 24, now: datetime | None = None
    ) -> list[dict] | None:
        if (
            not isinstance(provider_name, str)
            or not provider_name
            or len(provider_name) > 512
        ):
            return None
        if (
            isinstance(hours, bool)
            or not isinstance(hours, int)
            or not 1 <= hours <= 168
        ):
            raise ValueError("hours must be an integer between 1 and 168")
        current = now or datetime.now(timezone.utc)
        cutoff = current - timedelta(hours=hours)
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    """SELECT provider_name, status, response_time_ms,
                              last_check_timestamp
                       FROM provider_health_checks WHERE provider_name=?
                       ORDER BY last_check_timestamp ASC""",
                    (provider_name,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise ProviderStatusError(
                "provider status database is unavailable"
            ) from exc
        result = []
        for row in rows:
            checked = _timestamp(row["last_check_timestamp"])
            if checked is None or checked < cutoff:
                continue
            result.append(
                {
                    "provider_name": row["provider_name"],
                    "status": row["status"],
                    "response_time_ms": row["response_time_ms"],
                    "timestamp": row["last_check_timestamp"],
                }
            )
        return result

    def all_history(
        self, hours: int = 24, now: datetime | None = None
    ) -> dict[str, list[dict]]:
        current = self.current_status(now)
        result = {}
        for provider in sorted(current["providers"]):
            rows = self.history(provider, hours, now)
            if rows:
                result[provider] = rows
        return result

    def best(self, now: datetime | None = None) -> dict:
        status = self.current_status(now)
        providers = status["providers"]
        fresh = [
            item
            for item in providers.values()
            if not item["stale"] and item["status"] == "healthy"
        ]
        stale_healthy = [
            item
            for item in providers.values()
            if item["stale"] and item["status"] == "healthy"
        ]
        degraded = [item for item in providers.values() if item["status"] == "degraded"]
        candidates = fresh or stale_healthy or degraded
        if not candidates:
            return {
                "best_provider": None,
                "reason": "All providers are down",
                "alternatives": [],
                "source": self.source,
                "stale": status["stale"],
                "as_of": status["timestamp"],
            }
        candidates.sort(
            key=lambda item: (
                item["stale"],
                item["response_time_ms"] or math.inf,
                item["provider_name"],
            )
        )
        chosen = candidates[0]
        alternatives = [item["provider_name"] for item in candidates[1:3]]
        fresh_label = "stale snapshot" if chosen["stale"] else "current"
        latency = chosen["response_time_ms"] or 0
        return {
            "best_provider": chosen["provider_name"],
            "reason": (f"{fresh_label} {chosen['status']} provider ({latency:.1f}ms)"),
            "alternatives": alternatives,
            "source": self.source,
            "stale": chosen["stale"],
            "as_of": status["timestamp"],
        }

    def health(self, now: datetime | None = None) -> dict:
        status = self.current_status(now)
        rows = list(status["providers"].values())
        latest = max(
            (_timestamp(item["last_check"]) for item in rows),
            key=lambda value: value or datetime.min.replace(tzinfo=timezone.utc),
            default=None,
        )
        age = _age_seconds(latest, now or datetime.now(timezone.utc))
        stale = latest is None or age is None or age > self.stale_after_seconds
        return {
            "status": (
                "unknown" if latest is None else ("degraded" if stale else "healthy")
            ),
            "providers_monitored": len(rows),
            "check_interval_minutes": None,
            "running": False,
            "read_only": True,
            "stale": stale,
            "last_check": latest.isoformat() if latest else None,
            "source": self.source,
        }

    def export_csv(self, now: datetime | None = None) -> str:
        status = self.current_status(now)
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            [
                "provider_name",
                "status",
                "response_time_ms",
                "uptime_percent",
                "rate_limit_percent",
                "last_check_timestamp",
                "consecutive_failures",
            ]
        )
        for provider, item in sorted(status["providers"].items()):
            writer.writerow(
                [
                    provider,
                    item["status"],
                    item["response_time_ms"],
                    item["uptime_percent"],
                    item["rate_limit_percent"],
                    item["last_check"],
                    item["consecutive_failures"],
                ]
            )
        return output.getvalue()
