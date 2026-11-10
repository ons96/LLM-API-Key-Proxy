"""Small SQLite-backed session and decision state for the P0 router."""

import hashlib
import math
import secrets
import sqlite3
import threading
import time
from pathlib import Path

from router_core import CacheHint

VERDICTS = frozenset({
    "pass", "fail_build", "fail_lint", "fail_test",
    "fail_scope", "fail_tamper", "timeout", "unknown",
})


class RouterState:
    """Persist only hashes, route metadata, and bounded outcomes."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY,
                session_hash TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                capability TEXT NOT NULL,
                deployment TEXT NOT NULL,
                provider TEXT NOT NULL,
                config_fingerprint TEXT NOT NULL,
                created_at REAL NOT NULL
            )"""
        )
        decision_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(decisions)")}
        if "provider" not in decision_columns:
            self.connection.execute("ALTER TABLE decisions ADD COLUMN provider TEXT NOT NULL DEFAULT 'unconfigured'")
        if "config_fingerprint" not in decision_columns:
            self.connection.execute("ALTER TABLE decisions ADD COLUMN config_fingerprint TEXT NOT NULL DEFAULT 'default'")
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                session_hash TEXT PRIMARY KEY,
                deployment TEXT NOT NULL,
                phase TEXT NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY,
                session_hash TEXT NOT NULL,
                deployment TEXT NOT NULL,
                outcome TEXT NOT NULL,
                error_class TEXT,
                created_at REAL NOT NULL
            )"""
        )
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS cache_observations (
                id INTEGER PRIMARY KEY,
                session_hash TEXT NOT NULL,
                deployment TEXT NOT NULL,
                prefix_hash TEXT NOT NULL,
                cached_tokens INTEGER NOT NULL,
                cache_write_tokens INTEGER NOT NULL,
                ttl_seconds REAL NOT NULL DEFAULT 600,
                observed_at REAL NOT NULL
            )"""
        )
        cache_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(cache_observations)")}
        if "ttl_seconds" not in cache_columns:
            self.connection.execute(
                "ALTER TABLE cache_observations ADD COLUMN ttl_seconds REAL NOT NULL DEFAULT 600"
            )
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                session_hash TEXT NOT NULL,
                deployment TEXT NOT NULL,
                created_at REAL NOT NULL,
                graded_at REAL,
                grade INTEGER,
                verdict TEXT
            )"""
        )
        self.connection.commit()

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def record(self, session_id: str, prompt: str, capability: str, deployment: str, provider: str = "unconfigured", config_fingerprint: str = "default") -> None:
        with self.lock:
            self.connection.execute(
                "INSERT INTO decisions(session_hash, prompt_hash, capability, deployment, provider, config_fingerprint, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self._hash(session_id), self._hash(prompt), capability, deployment, provider, config_fingerprint, time.time()),
            )
            self.connection.commit()

    def count(self) -> int:
        with self.lock:
            return int(self.connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0])

    def preferred(self, session_id: str, phase: str) -> str | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT deployment FROM sessions WHERE session_hash = ? AND phase = ?",
                (self._hash(session_id), phase),
            ).fetchone()
        return None if row is None else str(row[0])

    def set_preferred(self, session_id: str, phase: str, deployment: str) -> None:
        with self.lock:
            self.connection.execute(
                "INSERT OR REPLACE INTO sessions(session_hash, deployment, phase, updated_at) VALUES (?, ?, ?, ?)",
                (self._hash(session_id), deployment, phase, time.time()),
            )
            self.connection.commit()

    def record_outcome(
        self,
        session_id: str,
        deployment: str,
        outcome: str,
        error_class: str | None = None,
    ) -> None:
        """Record bounded outcome categories without raw request data."""
        allowed = {"success", "operational_failure", "semantic_failure", "escalated"}
        if outcome not in allowed:
            raise ValueError(f"unsupported outcome: {outcome}")
        with self.lock:
            self.connection.execute(
                "INSERT INTO outcomes(session_hash, deployment, outcome, error_class, created_at) VALUES (?, ?, ?, ?, ?)",
                (self._hash(session_id), deployment, outcome, error_class, time.time()),
            )
            self.connection.commit()

    def outcome_count(self, outcome: str) -> int:
        with self.lock:
            return int(
                self.connection.execute("SELECT COUNT(*) FROM outcomes WHERE outcome = ?", (outcome,)).fetchone()[0]
            )

    def open_attempt(self, session_id: str, deployment: str) -> str:
        """Issue an attempt id for the client to grade later.

        Dispatching a request no longer implies success: the attempt stays
        ungraded until the client reports a verdict. A router that grades on
        HTTP 200 learns only which providers avoid rate limits.
        """
        attempt_id = secrets.token_hex(16)
        with self.lock:
            self.connection.execute(
                "INSERT INTO attempts(attempt_id, session_hash, deployment, created_at) VALUES (?, ?, ?, ?)",
                (attempt_id, self._hash(session_id), deployment, time.time()),
            )
            self.connection.commit()
        return attempt_id

    def grade_attempt(self, attempt_id: str, grade: bool, verdict: str) -> None:
        """Store a client verdict once. Unknown verdicts and regrades are rejected."""
        if not isinstance(grade, bool):
            raise ValueError("grade must be a boolean")
        if not isinstance(verdict, str) or verdict not in VERDICTS:
            raise ValueError(f"unsupported verdict: {verdict}")
        # Unknown is not success. Only a verified `pass` may carry a true grade;
        # otherwise a client without an oracle could report `unknown` as a win.
        if grade and verdict != "pass":
            raise ValueError("grade must be false unless verdict is 'pass'")
        if not grade and verdict == "pass":
            raise ValueError("verdict 'pass' requires grade true")
        with self.lock:
            row = self.connection.execute(
                "SELECT graded_at FROM attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise ValueError("unknown attempt_id")
            if row[0] is not None:
                raise ValueError("attempt already graded")
            self.connection.execute(
                "UPDATE attempts SET graded_at = ?, grade = ?, verdict = ? WHERE attempt_id = ?",
                (time.time(), int(grade), verdict, attempt_id),
            )
            self.connection.commit()

    def attempt(self, attempt_id: str) -> dict[str, object] | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT attempt_id, deployment, created_at, graded_at, grade, verdict FROM attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "attempt_id": row[0], "deployment": row[1], "created_at": row[2],
            "graded_at": row[3], "grade": row[4], "verdict": row[5],
        }

    def pending_attempts(self) -> int:
        with self.lock:
            return int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM attempts WHERE graded_at IS NULL"
                ).fetchone()[0]
            )

    def graded_attempts(self, passed: bool | None = None) -> int:
        query = "SELECT COUNT(*) FROM attempts WHERE graded_at IS NOT NULL"
        params: tuple[object, ...] = ()
        if passed is not None:
            query += " AND grade = ?"
            params = (int(passed),)
        with self.lock:
            return int(self.connection.execute(query, params).fetchone()[0])

    def record_cache(
        self,
        session_id: str,
        deployment: str,
        prefix: str,
        cached_tokens: int = 0,
        cache_write_tokens: int = 0,
        ttl_seconds: float = 600.0,
    ) -> None:
        """Store provider-neutral cache observations keyed by deployment."""
        cached_count = self._cache_count(cached_tokens)
        write_count = self._cache_count(cache_write_tokens)
        ttl = self._strict_nonnegative(ttl_seconds)
        with self.lock:
            self.connection.execute(
                "INSERT INTO cache_observations(session_hash, deployment, prefix_hash, cached_tokens, cache_write_tokens, ttl_seconds, observed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self._hash(session_id), deployment, self._hash(prefix), cached_count, write_count, ttl, time.time()),
            )
            self.connection.commit()

    @staticmethod
    def _cache_count(value: object) -> int:
        parsed = RouterState._strict_nonnegative(value)
        if not parsed.is_integer():
            raise ValueError("cache counts must be finite non-negative integers")
        return int(parsed)

    @staticmethod
    def _strict_nonnegative(value: object) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("cache values must be finite and non-negative") from exc
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError("cache values must be finite and non-negative")
        return parsed

    def cache_hints(self, session_id: str, prefix: str) -> dict[str, CacheHint]:
        """Return the newest per-deployment cache observations for a prefix."""
        with self.lock:
            rows = self.connection.execute(
                """SELECT deployment, cached_tokens, observed_at, ttl_seconds
                   FROM cache_observations
                   WHERE session_hash = ? AND prefix_hash = ?
                   ORDER BY observed_at DESC""",
                (self._hash(session_id), self._hash(prefix)),
            ).fetchall()
        hints: dict[str, CacheHint] = {}
        for deployment, cached_tokens, observed_at, ttl_seconds in rows:
            if deployment not in hints:
                try:
                    cached_count = self._cache_count(cached_tokens)
                    observed = self._strict_nonnegative(observed_at)
                    ttl = self._strict_nonnegative(ttl_seconds)
                    hints[str(deployment)] = CacheHint(
                        deployment_id=str(deployment),
                        cached_tokens=cached_count,
                        observed_at=observed,
                        ttl_seconds=ttl,
                    )
                except (TypeError, ValueError, OverflowError):
                    continue
        return hints

    def cache_count(self, deployment: str) -> int:
        with self.lock:
            return int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM cache_observations WHERE deployment = ? AND cached_tokens > 0",
                    (deployment,),
                ).fetchone()[0]
            )

    def total_cache_hits(self) -> int:
        with self.lock:
            return int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM cache_observations WHERE cached_tokens > 0"
                ).fetchone()[0]
            )

    def close(self) -> None:
        self.connection.close()
