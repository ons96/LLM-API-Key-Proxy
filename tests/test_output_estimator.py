"""Output estimator tests (#476).

Covers cold-start defaults, EMA convergence (median within +-30%), persistence
across instances, and degradation when telemetry is missing/corrupt.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proxy_app.routing.output_estimator import (
    DEFAULT_OUTPUT_TOKENS,
    OutputEstimator,
)
from proxy_app.routing.request_features import TaskClass


class TestDefaults:
    def test_defaults_are_sane(self):
        assert DEFAULT_OUTPUT_TOKENS[TaskClass.GREETING] < \
            DEFAULT_OUTPUT_TOKENS[TaskClass.SHORT_QA] < \
            DEFAULT_OUTPUT_TOKENS[TaskClass.REASONING] < \
            DEFAULT_OUTPUT_TOKENS[TaskClass.AGENTIC]

    def test_cold_start_returns_defaults(self, tmp_path):
        est = OutputEstimator(str(tmp_path / "missing.db"))
        assert est.estimate(TaskClass.GREETING) == \
            DEFAULT_OUTPUT_TOKENS[TaskClass.GREETING]
        assert est.estimate(TaskClass.AGENTIC) == \
            DEFAULT_OUTPUT_TOKENS[TaskClass.AGENTIC]

    def test_unknown_task_class_falls_back(self, tmp_path):
        est = OutputEstimator(str(tmp_path / "missing.db"))
        assert est.estimate(None) == 200  # type: ignore[arg-type]


class TestEmaCalibration:
    def test_converges_within_30_percent(self, tmp_path):
        db = str(tmp_path / "telemetry.db")
        est = OutputEstimator(db)
        for _ in range(8):
            est.record_observation(TaskClass.CODE_GEN, 900)
        fresh = OutputEstimator(db)
        v = fresh.estimate(TaskClass.CODE_GEN)
        assert abs(v - 900) < 0.30 * 900, f"EMA drifted: {v}"

    def test_persists_across_instances(self, tmp_path):
        db = str(tmp_path / "telemetry.db")
        OutputEstimator(db).record_observation(TaskClass.SHORT_QA, 250)
        OutputEstimator(db).record_observation(TaskClass.SHORT_QA, 350)
        v = OutputEstimator(db).estimate(TaskClass.SHORT_QA)
        assert 200 < v < 400

    def test_other_task_classes_untouched(self, tmp_path):
        db = str(tmp_path / "telemetry.db")
        est = OutputEstimator(db)
        est.record_observation(TaskClass.AGENTIC, 900)
        # calibrating AGENTIC must not move CODE_GEN
        assert est.estimate(TaskClass.CODE_GEN) == \
            DEFAULT_OUTPUT_TOKENS[TaskClass.CODE_GEN]

    def test_negative_observation_ignored(self, tmp_path):
        db = str(tmp_path / "telemetry.db")
        est = OutputEstimator(db)
        est.record_observation(TaskClass.CODE_GEN, -5)
        assert est.estimate(TaskClass.CODE_GEN) == \
            DEFAULT_OUTPUT_TOKENS[TaskClass.CODE_GEN]


class TestDegradation:
    def test_empty_db_file(self, tmp_path):
        db = str(tmp_path / "empty.db")
        db_path = Path(db)
        db_path.touch()
        est = OutputEstimator(db)
        assert est.estimate(TaskClass.REASONING) == \
            DEFAULT_OUTPUT_TOKENS[TaskClass.REASONING]

    def test_corrupt_db_degrades_to_defaults(self, tmp_path):
        db = str(tmp_path / "corrupt.db")
        Path(db).write_bytes(b"this is not sqlite")
        est = OutputEstimator(db)
        # read path must not raise
        assert est.estimate(TaskClass.SUMMARIZATION) == \
            DEFAULT_OUTPUT_TOKENS[TaskClass.SUMMARIZATION]

    def test_unknown_task_class_row_ignored(self, tmp_path):
        import sqlite3

        db = str(tmp_path / "telemetry.db")
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE task_completion_stats (task_class TEXT PRIMARY KEY,"
            " ema REAL, count INTEGER, updated_at REAL)"
        )
        conn.execute(
            "INSERT INTO task_completion_stats VALUES ('made-up-class', 999, 1, 0)"
        )
        conn.commit()
        conn.close()
        est = OutputEstimator(db)
        assert est.estimate(TaskClass.GREETING) == \
            DEFAULT_OUTPUT_TOKENS[TaskClass.GREETING]
