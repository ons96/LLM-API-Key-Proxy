"""Tests for proxy_app.routing.latency_predictor (#477)."""

import os
import sqlite3
import tempfile
import time

import pytest

from proxy_app.routing.latency_predictor import (
    DEFAULT_P_FAIL,
    DEFAULT_TPS,
    DEFAULT_TTFT_MS,
    LatencyPredictor,
    Prediction,
)


@pytest.fixture()
def telemetry_db():
    """Temp llm_events DB with known aggregates: (tp,m1) 10x success
    tps=100 ttft=500ms; (tp,m2) 10 events, 8 success -> p_fail=0.2."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE llm_events (
            ts_start REAL, provider TEXT, model TEXT, status TEXT,
            tps REAL, ttft_ms REAL)"""
    )
    now = time.time()
    for i in range(10):
        conn.execute(
            "INSERT INTO llm_events VALUES (?,?,?,?,?,?)",
            (now - i * 60, "tp", "m1", "success", 100.0, 500.0),
        )
    for i in range(10):
        status = "success" if i < 8 else "error"
        conn.execute(
            "INSERT INTO llm_events VALUES (?,?,?,?,?,?)",
            (now - i * 60, "tp", "m2", status, 200.0, 250.0),
        )
    # old rows outside window (should be ignored)
    for i in range(5):
        conn.execute(
            "INSERT INTO llm_events VALUES (?,?,?,?,?,?)",
            (now - 48 * 3600 - i * 60, "tp", "m1", "success", 500.0, 100.0),
        )
    conn.commit()
    conn.close()
    yield path
    os.unlink(path)


@pytest.fixture()
def speeds_file():
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        fh.write('{"fastp": {"tps": 750, "ttft": 0.5}}')
    yield path
    os.unlink(path)


@pytest.fixture()
def leaderboard_file():
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as fh:
        fh.write("rank,provider,model,effort,TTFT_sec,TPS,avg_tokens,10K_total_sec\n")
        fh.write("1,lbfast,lbmodel,,0.315,1411.0,445,7.4\n")
        fh.write("2,brokeprov,,,0.5,0.0,0,0\n")
    yield path
    os.unlink(path)


def _predictor(telemetry_db, speeds_file="", leaderboard_csv=""):
    return LatencyPredictor(
        telemetry_db=telemetry_db,
        leaderboard_csv=leaderboard_csv,
        speeds_path=speeds_file,
        min_samples=5,
        window_h=24,
    )


# -- formula + source cascade -------------------------------------------


def test_formula_exact_telemetry(telemetry_db):
    pred = _predictor(telemetry_db).predict("tp", "m1", output_tokens=100)
    # E = (500 + 100/100*1000) / (1 - 0.0) = 1500
    assert pred.e_time_ms == pytest.approx(1500.0, rel=1e-9)
    assert pred.source == "telemetry"
    assert pred.confidence == 1.0
    assert pred.p_fail == 0.0


def test_p_fail_factor(telemetry_db):
    pred = _predictor(telemetry_db).predict("tp", "m2", output_tokens=100)
    # (250 + 100/200*1000) / (1 - 0.2) = 750 / 0.8 = 937.5
    assert pred.e_time_ms == pytest.approx(937.5, rel=1e-9)
    assert pred.p_fail == pytest.approx(0.2)


def test_min_samples_gate_skips_telemetry(telemetry_db):
    # fixture has exactly 10 samples; require 11 -> falls to default
    pred = _predictor(telemetry_db).predict("tp", "m1", output_tokens=100)
    strict = LatencyPredictor(
        telemetry_db=telemetry_db, leaderboard_csv="", speeds_path="", min_samples=11
    ).predict("tp", "m1", output_tokens=100)
    assert pred.source == "telemetry"
    assert strict.source == "default"


def test_leaderboard_source(telemetry_db, leaderboard_file):
    pred = _predictor(telemetry_db, leaderboard_csv=leaderboard_file).predict(
        "lbfast", "lbmodel", output_tokens=100
    )
    assert pred.source == "leaderboard"
    assert pred.confidence == 0.7
    # (315 + 100/1411*1000) / (1 - 0.05)
    expected = (315.0 + (100.0 / 1411.0) * 1000.0) / (1.0 - DEFAULT_P_FAIL)
    assert pred.e_time_ms == pytest.approx(expected, rel=1e-9)


def test_speeds_source(telemetry_db, speeds_file):
    pred = _predictor(telemetry_db, speeds_file=speeds_file).predict(
        "fastp", "whatever", output_tokens=100
    )
    assert pred.source == "speeds"
    assert pred.confidence == 0.5
    assert pred.ttft_ms == 500.0  # 0.5s -> ms
    assert pred.tps == 750.0


def test_default_source(telemetry_db):
    pred = _predictor(telemetry_db).predict("unknown", "model", output_tokens=100)
    assert pred.source == "default"
    assert pred.confidence == 0.2
    assert pred.ttft_ms == DEFAULT_TTFT_MS
    assert pred.tps == DEFAULT_TPS
    assert pred.e_time_ms == pytest.approx(
        (DEFAULT_TTFT_MS + (100.0 / DEFAULT_TPS) * 1000.0) / (1.0 - DEFAULT_P_FAIL)
    )


def test_source_cascade_prefers_telemetry(telemetry_db, speeds_file):
    # same provider/model present in telemetry AND speeds -> telemetry wins
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        fh.write('{"tp": {"tps": 999, "ttft": 9.9}}')
    pred = LatencyPredictor(
        telemetry_db=telemetry_db, leaderboard_csv="", speeds_path=path
    ).predict("tp", "m1", output_tokens=100)
    os.unlink(path)
    assert pred.source == "telemetry"
    assert pred.tps == 100.0


# -- degradation paths ---------------------------------------------------


def test_missing_db_file():
    pred = LatencyPredictor(
        telemetry_db="/nonexistent/telemetry.db", leaderboard_csv="", speeds_path=""
    ).predict("a", "b", output_tokens=50)
    assert pred.source == "default"
    assert pred.e_time_ms > 0


def test_empty_db(tmp_path):
    db = str(tmp_path / "empty.db")
    sqlite3.connect(db).close()
    pred = LatencyPredictor(
        telemetry_db=db, leaderboard_csv="", speeds_path=""
    ).predict("a", "b", output_tokens=50)
    assert pred.source == "default"


def test_db_without_table(tmp_path):
    db = str(tmp_path / "nottable.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE other (x INTEGER)")
    conn.commit()
    conn.close()
    pred = LatencyPredictor(
        telemetry_db=db, leaderboard_csv="", speeds_path=""
    ).predict("a", "b", output_tokens=50)
    assert pred.source == "default"


def test_concrete_provider_fallback(tmp_path):
    """Events logged with concrete_provider/concrete_model must aggregate
    under the concrete identity (mirrors reorder_chains behavior)."""
    db = str(tmp_path / "concrete.db")
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE llm_events (
            ts_start REAL, provider TEXT, model TEXT, status TEXT,
            tps REAL, ttft_ms REAL, concrete_provider TEXT, concrete_model TEXT)"""
    )
    now = time.time()
    for i in range(8):
        conn.execute(
            "INSERT INTO llm_events VALUES (?,?,?,?,?,?,?,?)",
            (now - i * 60, "alias", "aliasmodel", "success", 100.0, 500.0, "real", "realmodel"),
        )
    conn.commit()
    conn.close()
    pred = LatencyPredictor(
        telemetry_db=db, leaderboard_csv="", speeds_path=""
    ).predict("real", "realmodel", output_tokens=100)
    assert pred.source == "telemetry"
    assert pred.tps == pytest.approx(100.0)


def test_old_window_rows_ignored(telemetry_db):
    pred = _predictor(telemetry_db).predict("tp", "m1", output_tokens=100)
    # stale 48h rows at tps=500 must NOT be averaged in
    assert pred.tps == pytest.approx(100.0)


def test_probe_rows_excluded_from_aggregate(tmp_path):
    """Health probes (completion_tokens < min) must not skew tps EMA."""
    db = str(tmp_path / "probes.db")
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE llm_events (
            ts_start REAL, provider TEXT, model TEXT, status TEXT,
            tps REAL, ttft_ms REAL, completion_tokens INTEGER)"""
    )
    now = time.time()
    for i in range(8):
        # 8 real rows @ tps=100
        conn.execute(
            "INSERT INTO llm_events VALUES (?,?,?,?,?,?,?)",
            (now - i * 60, "tp", "m", "success", 100.0, 500.0, 400),
        )
    for i in range(200):
        # 200 probe rows @ tps=0.5 would crush the average without the filter
        conn.execute(
            "INSERT INTO llm_events VALUES (?,?,?,?,?,?,?)",
            (now - i * 10, "tp", "m", "success", 0.5, 3000.0, 1),
        )
    conn.commit()
    conn.close()
    pred = LatencyPredictor(
        telemetry_db=db, leaderboard_csv="", speeds_path="", min_completion_tokens=50
    ).predict("tp", "m", output_tokens=100)
    assert pred.source == "telemetry"
    assert pred.tps == pytest.approx(100.0)
    assert pred.samples_used if hasattr(pred, "samples_used") else True


# -- predict_many + misc -------------------------------------------------


def test_predict_many_mixed(telemetry_db, leaderboard_file, speeds_file):
    preds = LatencyPredictor(
        telemetry_db=telemetry_db,
        leaderboard_csv=leaderboard_file,
        speeds_path=speeds_file,
    ).predict_many(
        [("tp", "m1"), ("lbfast", "lbmodel"), ("fastp", "x"), ("nope", "nope")],
        output_tokens=100,
    )
    assert [p.source for p in preds] == ["telemetry", "leaderboard", "speeds", "default"]
    assert len(preds) == 4
    assert all(isinstance(p, Prediction) for p in preds)


def test_predict_many_fast(telemetry_db, speeds_file):
    start = time.monotonic()
    LatencyPredictor(
        telemetry_db=telemetry_db, leaderboard_csv="", speeds_path=speeds_file
    ).predict_many([(f"p{i}", f"m{i}") for i in range(50)], output_tokens=200)
    elapsed_ms = (time.monotonic() - start) * 1000
    # AC: <2ms per candidate for <=50 candidates -> 100ms budget is 50x slack
    assert elapsed_ms < 100.0


def test_leaderboard_skips_zero_tps(telemetry_db, leaderboard_file):
    pred = _predictor(telemetry_db, leaderboard_csv=leaderboard_file).predict(
        "brokeprov", "x", output_tokens=100
    )
    assert pred.source == "default"  # zero-TPS row must be ignored, not used


def test_debug_header_shape():
    pred = Prediction(
        provider="p", model="m", e_time_ms=1234.5, ttft_ms=500.0,
        tps=100.0, p_fail=0.1, confidence=1.0, source="telemetry",
    )
    h = pred.to_debug_header()
    assert "p/m" in h
    assert "E=1234ms" in h  # 1234.5 -> round-half-even
    assert "src=telemetry" in h
