"""Persistent provider penalty store with time-decay.

Replaces the in-memory cooldown_manager dict with a SQLite-backed store that:
  - records failures per (provider, model, failure_type)
  - computes a decaying penalty score via exponential half-life
  - is consumed by _get_candidates() to sort chain candidates
  - NEVER deletes providers from the master chain; only deprioritizes

Schema (penalty_events append-only + penalty_state materialized):
  penalty_events(id, ts, provider, model, failure_type, weight)
  penalty_state(provider, model, failure_type, count, last_ts, score, updated_at)

Failure types + base weights:
  rate_limit        1.0   (transient, recovers fast)
  timeout           1.5   (slow upstream, recovers medium)
  provider_down     2.0   (5xx, recovers medium)
  out_of_credit     3.0   (daily quota hit, recovers slow)
  invalid_key       5.0   (401/403, recovers very slow; needs human attention)
  bad_output        1.0   (malformed/truncated response)

Decay:
  score = sum(weight_i * exp(-age_i / half_life))
  Default half-life = 1h (3600s). Tunable via PENALTY_HALFLIFE_S.

Refs: task-board #196, #194. Consumed by #195 (dynamic reorder).
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("PENALTY_DB_PATH", "/dev/shm/penalty.db")
HALF_LIFE_S = float(os.environ.get("PENALTY_HALFLIFE_S", "3600"))  # 1h default
MAX_EVENTS_PER_KEY = 200  # cap event log growth; state table is the source of truth

FAILURE_WEIGHTS: Dict[str, float] = {
    "rate_limit": 1.0,
    "timeout": 1.5,
    "provider_down": 2.0,
    "out_of_credit": 3.0,
    "invalid_key": 5.0,
    "bad_output": 1.0,
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS penalty_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    failure_type TEXT NOT NULL,
    weight REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_key ON penalty_events(provider, model, failure_type);
CREATE INDEX IF NOT EXISTS idx_events_ts ON penalty_events(ts);

CREATE TABLE IF NOT EXISTS penalty_state (
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    failure_type TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    last_ts REAL NOT NULL DEFAULT 0,
    score REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (provider, model, failure_type)
);
"""


@dataclass
class PenaltyEntry:
    provider: str
    model: str
    failure_type: str
    count: int
    last_ts: float
    score: float  # decaying score at the time of read


class PenaltyStore:
    """Thread-safe SQLite penalty store. All ops are sync (called from async via executor)."""

    _singleton: Optional["PenaltyStore"] = None
    _singleton_lock = threading.Lock()

    def __init__(self, db_path: str = DB_PATH, half_life_s: float = HALF_LIFE_S) -> None:
        self.db_path = db_path
        self.half_life_s = half_life_s
        self._lock = threading.Lock()
        self._init_db()

    @classmethod
    def get(cls) -> "PenaltyStore":
        """Process-wide singleton. Lazy-init on first use."""
        if cls._singleton is None:
            with cls._singleton_lock:
                if cls._singleton is None:
                    cls._singleton = cls()
        return cls._singleton

    @classmethod
    def reset_singleton_for_test(cls) -> None:
        """Test helper: clear the singleton so a new DB path can be used."""
        with cls._singleton_lock:
            cls._singleton = None

    def clear(self) -> None:
        """Wipe all events + state rows. Used by tests; safe to call at runtime."""
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM penalty_events")
            conn.execute("DELETE FROM penalty_state")
            conn.commit()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()
        logger.info("penalty DB initialized at %s (half_life=%.0fs)", self.db_path, self.half_life_s)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
        finally:
            conn.close()

    # --- write path ---

    def record_failure(
        self,
        provider: str,
        model: str,
        failure_type: str,
        weight: Optional[float] = None,
        ts: Optional[float] = None,
    ) -> None:
        """Record a failure event and update the materialized state row.

        Idempotent in the sense that calling twice = two failures logged.
        weight defaults to FAILURE_WEIGHTS[failure_type]; unknown types get 1.0.
        """
        if failure_type not in FAILURE_WEIGHTS:
            logger.warning("unknown failure_type %r; using weight=1.0", failure_type)
        w = weight if weight is not None else FAILURE_WEIGHTS.get(failure_type, 1.0)
        now = ts if ts is not None else time.time()

        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO penalty_events (ts, provider, model, failure_type, weight) VALUES (?,?,?,?,?)",
                (now, provider, model, failure_type, w),
            )
            # Upsert state: increment count, refresh score (add new weight, decay old).
            row = conn.execute(
                "SELECT count, score FROM penalty_state WHERE provider=? AND model=? AND failure_type=?",
                (provider, model, failure_type),
            ).fetchone()
            if row is None:
                count, score = 0, 0.0
            else:
                count, score = row[0], row[1]
                # Decay existing score by elapsed time since last_ts.
                # State row's last_ts is the previous update; we read it now.
                prev_ts_row = conn.execute(
                    "SELECT last_ts FROM penalty_state WHERE provider=? AND model=? AND failure_type=?",
                    (provider, model, failure_type),
                ).fetchone()
                prev_ts = prev_ts_row[0] if prev_ts_row else now
                # ponytail: clamp exponent to avoid math range error on huge ages.
                # If age > ~700 half-lives, exp underflows to 0 anyway.
                age = max(0.0, now - prev_ts)
                decay_exp = -age / self.half_life_s
                if decay_exp < -700:
                    score = 0.0
                else:
                    score = score * math.exp(decay_exp)

            new_count = count + 1
            new_score = score + w
            conn.execute(
                """INSERT INTO penalty_state (provider, model, failure_type, count, last_ts, score, updated_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(provider, model, failure_type) DO UPDATE SET
                     count=excluded.count,
                     last_ts=excluded.last_ts,
                     score=excluded.score,
                     updated_at=excluded.updated_at""",
                (provider, model, failure_type, new_count, now, new_score, now),
            )
            conn.commit()

            # Cap event log growth (ponytail: global lock, fine for our write rate).
            conn.execute(
                """DELETE FROM penalty_events WHERE id NOT IN (
                       SELECT id FROM penalty_events
                       WHERE provider=? AND model=? AND failure_type=?
                       ORDER BY ts DESC LIMIT ?
                   ) AND provider=? AND model=? AND failure_type=?""",
                (provider, model, failure_type, MAX_EVENTS_PER_KEY, provider, model, failure_type),
            )
            conn.commit()

    # --- read path ---

    def get_score(
        self, provider: str, model: str, failure_type: Optional[str] = None, now: Optional[float] = None
    ) -> float:
        """Get the current decaying penalty score for (provider, model).

        If failure_type is None, sums across all failure types.
        Score is recomputed lazily from state.last_ts + state.score using
        elapsed time. Does NOT write back to state (read-only).
        """
        now = now if now is not None else time.time()
        with self._lock, self._conn() as conn:
            if failure_type is None:
                rows = conn.execute(
                    "SELECT score, last_ts FROM penalty_state WHERE provider=? AND model=?",
                    (provider, model),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT score, last_ts FROM penalty_state WHERE provider=? AND model=? AND failure_type=?",
                    (provider, model, failure_type),
                ).fetchall()
        total = 0.0
        for score, last_ts in rows:
            age = max(0.0, now - last_ts)
            decay_exp = -age / self.half_life_s
            if decay_exp > -700:
                total += score * math.exp(decay_exp)
        return total

    def get_entries(
        self, provider: Optional[str] = None, model: Optional[str] = None, now: Optional[float] = None
    ) -> List[PenaltyEntry]:
        """List materialized state rows, with score decayed to current time."""
        now = now if now is not None else time.time()
        with self._lock, self._conn() as conn:
            if provider and model:
                rows = conn.execute(
                    "SELECT provider, model, failure_type, count, last_ts, score FROM penalty_state WHERE provider=? AND model=?",
                    (provider, model),
                ).fetchall()
            elif provider:
                rows = conn.execute(
                    "SELECT provider, model, failure_type, count, last_ts, score FROM penalty_state WHERE provider=?",
                    (provider,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT provider, model, failure_type, count, last_ts, score FROM penalty_state"
                ).fetchall()
        out = []
        for provider_, model_, ftype, count, last_ts, score in rows:
            age = max(0.0, now - last_ts)
            decay_exp = -age / self.half_life_s
            if decay_exp > -700:
                decayed = score * math.exp(decay_exp)
            else:
                decayed = 0.0
            out.append(
                PenaltyEntry(
                    provider=provider_,
                    model=model_,
                    failure_type=ftype,
                    count=count,
                    last_ts=last_ts,
                    score=decayed,
                )
            )
        return out

    def score_chain(
        self,
        candidates: Iterable[Tuple[str, str]],
        now: Optional[float] = None,
    ) -> List[Tuple[str, str, float]]:
        """Score a chain of (provider, model) candidates.

        Returns list of (provider, model, penalty_score) sorted ASC by penalty
        (lowest penalty first = healthiest first). Candidates with no penalty
        get score 0.0 and sort first.
        """
        now = now if now is not None else time.time()
        cand_list = list(candidates)
        if not cand_list:
            return []
        scored = [(p, m, self.get_score(p, m, now=now)) for p, m in cand_list]
        scored.sort(key=lambda x: x[2])
        return scored

    def prune_old_state(self, max_age_s: float = 86400.0) -> int:
        """Remove state rows that have fully decayed (last_ts older than max_age_s
        AND score contribution < 0.01). Returns count of removed rows."""
        cutoff = time.time() - max_age_s
        # ponytail: clamp exponent for huge ages (same as record_failure).
        decay_exp = -max_age_s / self.half_life_s
        threshold = 0.01 if decay_exp > -700 else 0.0
        with self._lock, self._conn() as conn:
            if decay_exp > -700:
                cur = conn.execute(
                    "DELETE FROM penalty_state WHERE last_ts < ? AND score * exp(?) < ?",
                    (cutoff, decay_exp, threshold),
                )
            else:
                # Fully decayed: any row older than cutoff can be removed.
                cur = conn.execute(
                    "DELETE FROM penalty_state WHERE last_ts < ?",
                    (cutoff,),
                )
            conn.commit()
            return cur.rowcount

    # --- async wrappers (for call sites that prefer await) ---

    async def arecord_failure(self, *args, **kwargs) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self.record_failure(*args, **kwargs))

    async def ascore_chain(self, candidates, now=None):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.score_chain(candidates, now=now))


# ---------------------------------------------------------------------------
# Failure classification (maps HTTP status / exception type -> failure_type)
# ---------------------------------------------------------------------------


def classify_failure(
    status_code: Optional[int] = None,
    error_message: Optional[str] = None,
    exception_type: Optional[str] = None,
) -> str:
    """Classify an error into one of the FAILURE_WEIGHTS keys."""
    msg = (error_message or "").lower()
    etype = (exception_type or "").lower()

    if status_code in (401, 403) or "auth" in etype or "unauthorized" in msg or "forbidden" in msg:
        return "invalid_key"
    if status_code == 429 or "rate" in msg and "limit" in msg:
        return "rate_limit"
    if status_code and 500 <= status_code < 600:
        return "provider_down"
    if "timeout" in etype or "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "quota" in msg or "credit" in msg or "out of" in msg:
        return "out_of_credit"
    if "json" in msg or "parse" in msg or "malformed" in msg or "truncat" in msg:
        return "bad_output"
    # Default: treat as provider_down (transient infrastructure issue).
    return "provider_down"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Smoke test: record failures, verify decay, verify chain sort.

    Run with: python -m proxy_app.penalty_store
    """
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        db = os.path.join(tmpdir, "test_penalty.db")
        PenaltyStore.reset_singleton_for_test()
        store = PenaltyStore(db_path=db, half_life_s=1.0)  # 1s half-life for fast test

        # 1. Record a 401 on groq/llama-3.3-70b
        store.record_failure("groq", "llama-3.3-70b", "invalid_key")
        score1 = store.get_score("groq", "llama-3.3-70b")
        assert score1 > 4.0, f"expected ~5.0 after invalid_key, got {score1}"
        print(f"  after invalid_key: score={score1:.3f}")

        # 2. Record a 429 on groq/llama-3.1-8b
        store.record_failure("groq", "llama-3.1-8b", "rate_limit")
        score2 = store.get_score("groq", "llama-3.1-8b")
        assert score2 > 0.5, f"expected ~1.0 after rate_limit, got {score2}"
        print(f"  after rate_limit: score={score2:.3f}")

        # 3. Chain sort: healthy provider should come first
        chain = [("groq", "llama-3.3-70b"), ("gemini", "gemini-1.5-pro"), ("groq", "llama-3.1-8b")]
        scored = store.score_chain(chain)
        print(f"  chain order: {scored}")
        assert scored[0][0] == "gemini", f"healthy provider should be first, got {scored[0]}"
        assert scored[-1][0] == "groq" and scored[-1][1] == "llama-3.3-70b", \
            f"highest-penalty provider should be last, got {scored[-1]}"

        # 4. Decay: wait 2 half-lives, score should drop ~75%
        import time as _time
        _time.sleep(2.0)
        score1_decayed = store.get_score("groq", "llama-3.3-70b")
        ratio = score1_decayed / score1 if score1 > 0 else 0
        print(f"  after 2 half-lives: score={score1_decayed:.3f} (ratio={ratio:.3f})")
        assert ratio < 0.3, f"expected <30% after 2 half-lives, got {ratio:.3f}"

        # 5. classify_failure sanity
        assert classify_failure(status_code=401) == "invalid_key"
        assert classify_failure(status_code=429) == "rate_limit"
        assert classify_failure(status_code=503) == "provider_down"
        assert classify_failure(exception_type="TimeoutError") == "timeout"
        assert classify_failure(error_message="out of credits") == "out_of_credit"

        print("self-test OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _self_test()
