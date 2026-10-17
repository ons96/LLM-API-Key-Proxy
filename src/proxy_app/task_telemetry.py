"""
Task-class telemetry + outcome buffer for the quota-aware routing brain.

Phase 3 of the routing brain (ons96/llm-leaderboard-aggregate): the gateway
records which task class served each request (X-Task-Class / X-Task-ID
headers on /v1/chat/completions) and buffers task outcomes reported by
harnesses and scripts (POST /api/task-outcome). The brain pulls exports
nightly and commits them to git — this DB is a buffer; git is the durable
store, so a wiped /tmp costs at most one day of labels.

All helpers are best-effort by design: telemetry must never break the
request hot path.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("TASK_TELEMETRY_DB_PATH", "/tmp/llm_task_telemetry.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS gateway_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT,
    ts REAL NOT NULL,
    virtual_model TEXT,
    task_class TEXT,
    task_id TEXT,
    concrete_provider TEXT,
    concrete_model TEXT,
    stream INTEGER DEFAULT 0,
    status TEXT NOT NULL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_gw_requests_class ON gateway_requests(task_class, ts);

CREATE TABLE IF NOT EXISTS task_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT,
    task_class TEXT NOT NULL,
    model_id TEXT,
    provider TEXT,
    success INTEGER NOT NULL,
    total_tokens INTEGER,
    turns INTEGER,
    source TEXT NOT NULL,
    notes TEXT,
    observed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task_outcomes_class ON task_outcomes(task_class, model_id);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db() -> None:
    try:
        conn = _connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()
    except Exception:
        logger.exception("task telemetry DB init failed (path=%s)", DB_PATH)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_since(since: Optional[str]) -> Optional[float]:
    """Parse an ISO-8601 'since' bound to a unix ts. Returns None if unset/bad."""
    if not since:
        return None
    try:
        dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def record_gateway_request(
    *,
    request_id: Optional[str],
    virtual_model: Optional[str],
    task_class: Optional[str],
    task_id: Optional[str],
    concrete_provider: Optional[str],
    concrete_model: Optional[str],
    stream: bool,
    status: str,
    error: Optional[str] = None,
) -> None:
    """Best-effort per-request task log. Never raises."""
    try:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO gateway_requests
                    (request_id, ts, virtual_model, task_class, task_id,
                     concrete_provider, concrete_model, stream, status, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    time.time(),
                    virtual_model,
                    task_class,
                    task_id,
                    concrete_provider,
                    concrete_model,
                    int(bool(stream)),
                    status,
                    (error or "")[:500] or None,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        logger.debug("record_gateway_request failed", exc_info=True)


def record_outcome(
    *,
    task_class: str,
    success: bool,
    model_id: Optional[str] = None,
    provider: Optional[str] = None,
    task_id: Optional[str] = None,
    total_tokens: Optional[int] = None,
    turns: Optional[int] = None,
    source: str = "manual",
    notes: Optional[str] = None,
    observed_at: Optional[str] = None,
) -> int:
    """Insert a reported task outcome. Raises ValueError on missing fields."""
    if not task_class or not str(task_class).strip():
        raise ValueError("task_class is required")
    if model_id is not None and not str(model_id).strip():
        model_id = None
    if not isinstance(success, bool):
        raise ValueError("success must be a boolean")
    observed = observed_at or _utc_now_iso()
    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO task_outcomes
                (task_id, task_class, model_id, provider, success,
                 total_tokens, turns, source, notes, observed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                str(task_class).strip(),
                model_id,
                provider,
                int(success),
                total_tokens,
                turns,
                source,
                notes,
                observed,
            ),
        )
        conn.commit()
        return int(cur.lastrowid or 0)
    finally:
        conn.close()


def export_outcomes(since: Optional[str] = None, limit: int = 50000) -> List[Dict[str, Any]]:
    """Outcomes for the brain to pull, oldest first."""
    since_ts = _parse_since(since)
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM task_outcomes ORDER BY id LIMIT ?", (int(limit),)
        ).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        d = dict(r)
        if since_ts is not None:
            try:
                ts = datetime.fromisoformat(d["observed_at"].replace("Z", "+00:00")).timestamp()
                if ts <= since_ts:
                    continue
            except (ValueError, AttributeError):
                pass
        d["success"] = bool(d["success"])
        out.append(d)
    return out


def export_gateway_requests(since: Optional[str] = None, limit: int = 50000) -> List[Dict[str, Any]]:
    """Per-request task log for the brain (volume/reliability signals)."""
    since_ts = _parse_since(since)
    conn = _connect()
    try:
        if since_ts is not None:
            rows = conn.execute(
                "SELECT * FROM gateway_requests WHERE ts > ? ORDER BY id LIMIT ?",
                (since_ts, int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM gateway_requests ORDER BY id LIMIT ?", (int(limit),)
            ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]
