"""Small SQLite-backed session and decision state for the P0 router."""

import hashlib
import sqlite3
import threading
import time
from pathlib import Path


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
                observed_at REAL NOT NULL
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

    def record_cache(
        self,
        session_id: str,
        deployment: str,
        prefix: str,
        cached_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        """Store provider-neutral cache observations keyed by deployment."""
        if cached_tokens < 0 or cache_write_tokens < 0:
            raise ValueError("cache token counts cannot be negative")
        with self.lock:
            self.connection.execute(
                "INSERT INTO cache_observations(session_hash, deployment, prefix_hash, cached_tokens, cache_write_tokens, observed_at) VALUES (?, ?, ?, ?, ?, ?)",
                (self._hash(session_id), deployment, self._hash(prefix), cached_tokens, cache_write_tokens, time.time()),
            )
            self.connection.commit()

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
