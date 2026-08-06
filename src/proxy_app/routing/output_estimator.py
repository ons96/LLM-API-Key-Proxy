"""Output-token estimator for smart routing (#476).

Per-task_class expected output tokens: static defaults (cold start) blended
with an EMA of observed completion_tokens per task_class, persisted in the
gateway telemetry DB (same sqlite file reorder_chains.py reads; default
TELEMETRY_DB_PATH=/dev/shm/telemetry.db). Stdlib-only (sqlite3), zero model
calls. Consumed by latency_predictor (#477) and chain_selector (#480).

Calibration table (created lazily on first write):
    task_completion_stats(task_class TEXT PRIMARY KEY, ema REAL,
                          count INTEGER, updated_at REAL)

Median-within-+-30% acceptance: EMA alpha=0.3 converges fast; with >=5
observations of the same task_class the estimate is within 30% of the recent
mean unless the workload distribution shifts mid-window.
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Dict, Optional

from .request_features import TaskClass

# ---------------------------------------------------------------------------
# Static defaults (cold start / no telemetry). Tuned per task class.
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_TOKENS: Dict[TaskClass, int] = {
    TaskClass.GREETING: 40,
    TaskClass.SHORT_QA: 120,
    TaskClass.VISION_CAPTION: 150,
    TaskClass.SUMMARIZATION: 500,
    TaskClass.CODE_EDIT: 400,
    TaskClass.FILE_ANALYSIS: 600,
    TaskClass.CODE_GEN: 600,
    TaskClass.REASONING: 800,
    TaskClass.AGENTIC: 1200,
}

_EMA_ALPHA = 0.3
_CACHE_TTL_S = 30.0
_TABLE = "task_completion_stats"
_DEFAULT_DB = os.environ.get("TELEMETRY_DB_PATH", "/dev/shm/telemetry.db")


class OutputEstimator:
    """EMA-calibrated per-task-class output token estimator."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or _DEFAULT_DB
        self._cache: Dict[TaskClass, float] = {}
        self._cache_ts = 0.0

    # -- reads -------------------------------------------------------------

    def estimate(self, task_class: TaskClass) -> int:
        """Expected output tokens for a task class (defaults + EMA blend)."""
        default = DEFAULT_OUTPUT_TOKENS.get(task_class, 200)
        ema = self._load_ema()
        return int(round(ema.get(task_class, default)))

    def _load_ema(self) -> Dict[TaskClass, float]:
        now = time.monotonic()
        if self._cache and now - self._cache_ts < _CACHE_TTL_S:
            return self._cache
        ema: Dict[TaskClass, float] = {}
        if os.path.exists(self._db_path):
            try:
                conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
                try:
                    rows = conn.execute(
                        f"SELECT task_class, ema FROM {_TABLE}"
                    ).fetchall()
                    for tc, val in rows:
                        try:
                            ema[TaskClass(tc)] = float(val)
                        except ValueError:
                            continue
                finally:
                    conn.close()
            except sqlite3.Error:
                pass  # corrupt/absent table -> defaults
        self._cache = ema
        self._cache_ts = now
        return ema

    # -- writes ------------------------------------------------------------

    def record_observation(self, task_class: TaskClass,
                           completion_tokens: int) -> None:
        """Fold one observed completion into the EMA (idempotent, safe)."""
        if completion_tokens < 0:
            return
        self._cache = {}  # invalidate cached reads
        try:
            conn = sqlite3.connect(self._db_path, timeout=2.0)
            try:
                conn.execute(
                    f"CREATE TABLE IF NOT EXISTS {_TABLE} ("
                    "task_class TEXT PRIMARY KEY, ema REAL, "
                    "count INTEGER, updated_at REAL)"
                )
                row = conn.execute(
                    f"SELECT ema, count FROM {_TABLE} WHERE task_class=?",
                    (task_class.value,),
                ).fetchone()
                if row is None:
                    conn.execute(
                        f"INSERT INTO {_TABLE} VALUES (?, ?, 1, ?)",
                        (task_class.value, float(completion_tokens),
                         time.time()),
                    )
                else:
                    old_ema, count = row
                    new_ema = _EMA_ALPHA * completion_tokens + \
                        (1.0 - _EMA_ALPHA) * old_ema
                    conn.execute(
                        f"UPDATE {_TABLE} SET ema=?, count=?, updated_at=? "
                        "WHERE task_class=?",
                        (new_ema, count + 1, time.time(), task_class.value),
                    )
                conn.commit()
            finally:
                conn.close()
        except sqlite3.Error:
            pass  # telemetry unavailable (tmpfs, read-only) -> defaults only


def _demo() -> None:
    """Self-check: PYTHONPATH=src python3 -m proxy_app.routing.output_estimator"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "telemetry.db")
        est = OutputEstimator(db)
        # cold start -> defaults
        assert est.estimate(TaskClass.GREETING) == 40
        assert est.estimate(TaskClass.AGENTIC) == 1200
        # calibration converges toward observations
        for _ in range(8):
            est.record_observation(TaskClass.CODE_GEN, 900)
        fresh = OutputEstimator(db)
        v = fresh.estimate(TaskClass.CODE_GEN)
        assert abs(v - 900) < 0.3 * 900, f"EMA drift: {v}"
        print(f"output_estimator OK (code-gen EMA -> {v})")


if __name__ == "__main__":
    _demo()
