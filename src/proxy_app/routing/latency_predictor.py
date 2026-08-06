"""Latency prediction for smart routing (#477).

Pure-stdlib, deterministic, zero network. Predicts expected end-to-end
latency for a (provider, model) candidate serving a request:

    E[time] = (TTFT + output_tokens / TPS) / (1 - p_fail)

Data sources, best-first (per issue #477):
  1. telemetry.db  -- EMA of llm_events over a rolling window
     (min_samples gate; real p_fail from success rate)
  2. leaderboard.csv -- nightly speedrun sweep (VPS: ~/llm-speedrun/data/)
  3. config/provider_speeds.json -- static per-provider defaults
  4. global defaults (conservative: TTFT 800ms, TPS 50)

Aggregates are loaded once per call window and cached in-memory (TTL 30s);
`predict_many` reuses one load for up to N candidates so per-candidate cost
is a dict lookup (<2ms for <=50 candidates).

Composition: consumes RequestFeatures.estimated_input_tokens via callers and
the #476 OutputEstimator for output_tokens; this module only needs the
output-token count as an argument.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_TTFT_MS = 800.0
DEFAULT_TPS = 50.0
DEFAULT_P_FAIL = 0.05  # applied when no success-rate data exists
DEFAULT_CONFIDENCE = 0.2

DEFAULT_TELEMETRY_DB = os.environ.get("TELEMETRY_DB_PATH", "/dev/shm/telemetry.db")
DEFAULT_LEADERBOARD_CSV = os.environ.get("LATENCY_LEADERBOARD_CSV", "")
DEFAULT_SPEEDS_PATH = os.environ.get("LATENCY_SPEEDS_PATH", "config/provider_speeds.json")
DEFAULT_MIN_SAMPLES = int(os.environ.get("LATENCY_MIN_SAMPLES", "5"))
DEFAULT_WINDOW_H = int(os.environ.get("LATENCY_WINDOW_H", "24"))
DEFAULT_CACHE_TTL = float(os.environ.get("LATENCY_CACHE_TTL", "30.0"))
# health probes log tiny completions (<=46 tok) whose tps measures mostly
# connection overhead; excluding them keeps throughput aggregates honest.
# No-op when the column is absent (older schemas).
DEFAULT_MIN_COMPLETION_TOKENS = int(os.environ.get("LATENCY_MIN_COMPLETION_TOKENS", "50"))

# confidence per data source (telemetry >= leaderboard >= speeds >= default)
_CONFIDENCE = {"telemetry": 1.0, "leaderboard": 0.7, "speeds": 0.5, "default": DEFAULT_CONFIDENCE}


@dataclass(frozen=True)
class Prediction:
    """Predicted latency for one candidate."""

    provider: str
    model: str
    e_time_ms: float
    ttft_ms: float
    tps: float
    p_fail: float
    confidence: float
    source: str  # telemetry | leaderboard | speeds | default

    def to_debug_header(self) -> str:
        """Single-line value for the X-Route-Prediction debug header.

        Header injection into the request pipeline lands with #480/#481;
        this helper only formats the payload.
        """
        return (
            f"{self.provider}/{self.model} "
            f"E={self.e_time_ms:.0f}ms "
            f"ttft={self.ttft_ms:.0f}ms "
            f"tps={self.tps:.1f} "
            f"fail={self.p_fail:.2f} "
            f"conf={self.confidence:.1f} "
            f"src={self.source}"
        )


class LatencyPredictor:
    """Best-source latency predictor with in-memory aggregate cache."""

    def __init__(
        self,
        telemetry_db: Optional[str] = None,
        leaderboard_csv: Optional[str] = None,
        speeds_path: Optional[str] = None,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        window_h: int = DEFAULT_WINDOW_H,
        cache_ttl: float = DEFAULT_CACHE_TTL,
        min_completion_tokens: int = DEFAULT_MIN_COMPLETION_TOKENS,
    ) -> None:
        self.telemetry_db = telemetry_db if telemetry_db is not None else DEFAULT_TELEMETRY_DB
        self.leaderboard_csv = (
            leaderboard_csv if leaderboard_csv is not None else DEFAULT_LEADERBOARD_CSV
        )
        self.speeds_path = speeds_path if speeds_path is not None else DEFAULT_SPEEDS_PATH
        self.min_samples = min_samples
        self.window_h = window_h
        self.cache_ttl = cache_ttl
        self.min_completion_tokens = min_completion_tokens
        self._cache: Dict[str, Tuple[float, object]] = {}

    # -- aggregate loaders (cached) -------------------------------------

    def _cached(self, key: str, loader) -> object:
        now = time.monotonic()
        hit = self._cache.get(key)
        if hit is not None and now - hit[0] < self.cache_ttl:
            return hit[1]
        value = loader()
        self._cache[key] = (now, value)
        return value

    def _load_telemetry(self) -> Dict[Tuple[str, str], Tuple[float, float, float, int]]:
        """(provider, model) -> (avg_tps, avg_ttft_ms, p_fail, samples).

        Mirrors reorder_chains.load_telemetry: concrete_* columns preferred,
        alias fallback; AVG over the window; success rate -> p_fail.
        """

        def load() -> Dict[Tuple[str, str], Tuple[float, float, float, int]]:
            out: Dict[Tuple[str, str], Tuple[float, float, float, int]] = {}
            if not os.path.exists(self.telemetry_db):
                return out
            try:
                conn = sqlite3.connect(f"file:{self.telemetry_db}?mode=ro", uri=True)
                conn.row_factory = sqlite3.Row
            except sqlite3.Error:
                return out
            try:
                cols = {
                    r["name"]
                    for r in conn.execute("PRAGMA table_info(llm_events)").fetchall()
                }
                required = {"provider", "model", "status", "tps", "ttft_ms", "ts_start"}
                if not required.issubset(cols):
                    return out
                has_concrete = {"concrete_provider", "concrete_model"}.issubset(cols)
                has_completion = "completion_tokens" in cols
                prov_expr = (
                    "COALESCE(NULLIF(concrete_provider, ''), provider)"
                    if has_concrete
                    else "provider"
                )
                model_expr = (
                    "COALESCE(NULLIF(concrete_model, ''), model)" if has_concrete else "model"
                )
                cutoff = time.time() - (self.window_h * 3600)
                probe_filter = (
                    f" AND completion_tokens >= {int(self.min_completion_tokens)}"
                    if has_completion
                    else ""
                )
                rows = conn.execute(
                    f"""
                    SELECT {prov_expr} AS provider, {model_expr} AS model,
                           COUNT(*) AS samples,
                           SUM(CASE WHEN status='success' THEN 1.0 ELSE 0.0 END) AS successes,
                           AVG(tps) AS avg_tps,
                           AVG(ttft_ms) AS avg_ttft_ms
                    FROM llm_events
                    WHERE ts_start >= ? AND provider IS NOT NULL AND model IS NOT NULL
                    {probe_filter}
                    GROUP BY {prov_expr}, {model_expr}
                    """,
                    (cutoff,),
                ).fetchall()
                for row in rows:
                    samples = int(row["samples"] or 0)
                    successes = float(row["successes"] or 0.0)
                    p_fail = 1.0 - (successes / samples) if samples > 0 else 1.0
                    out[(row["provider"], row["model"])] = (
                        float(row["avg_tps"] or 0.0),
                        float(row["avg_ttft_ms"] or 0.0),
                        p_fail,
                        samples,
                    )
                return out
            except sqlite3.Error:
                return out
            finally:
                conn.close()

        return self._cached("telemetry", load)  # type: ignore[return-value]

    def _load_leaderboard(self) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """(provider, model) -> (ttft_ms, tps) from the speedrun sweep CSV."""

        def load() -> Dict[Tuple[str, str], Tuple[float, float]]:
            out: Dict[Tuple[str, str], Tuple[float, float]] = {}
            if not self.leaderboard_csv or not os.path.exists(self.leaderboard_csv):
                return out
            try:
                with open(self.leaderboard_csv, newline="", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        provider = (row.get("provider") or "").strip()
                        model = (row.get("model") or "").strip()
                        if not provider or not model:
                            continue
                        try:
                            tps = float(row["TPS"])
                            ttft_ms = float(row["TTFT_sec"]) * 1000.0
                        except (KeyError, ValueError):
                            continue
                        if tps <= 0:
                            continue
                        out[(provider, model)] = (ttft_ms, tps)
                return out
            except (OSError, csv.Error):
                return out

        return self._cached("leaderboard", load)  # type: ignore[return-value]

    def _load_speeds(self) -> Dict[str, Tuple[float, float]]:
        """provider -> (ttft_ms, tps) from config/provider_speeds.json (ttft in sec)."""

        def load() -> Dict[str, Tuple[float, float]]:
            out: Dict[str, Tuple[float, float]] = {}
            if not self.speeds_path or not os.path.exists(self.speeds_path):
                return out
            try:
                with open(self.speeds_path, encoding="utf-8") as fh:
                    data = json.load(fh)
                if not isinstance(data, dict):
                    return out
                for provider, entry in data.items():
                    if not isinstance(entry, dict):
                        continue
                    try:
                        tps = float(entry.get("tps") or 0.0)
                        ttft_ms = float(entry.get("ttft") or 0.0) * 1000.0
                    except (TypeError, ValueError):
                        continue
                    if tps <= 0:
                        continue
                    out[provider] = (ttft_ms, tps)
                return out
            except (OSError, ValueError):
                return out

        return self._cached("speeds", load)  # type: ignore[return-value]

    # -- prediction ------------------------------------------------------

    def _e_time(self, ttft_ms: float, tps: float, p_fail: float, output_tokens: int) -> float:
        if tps <= 0:
            tps = DEFAULT_TPS
        base = ttft_ms + (output_tokens / tps) * 1000.0
        return base / (1.0 - min(max(p_fail, 0.0), 0.95))

    def predict(self, provider: str, model: str, output_tokens: int = 200) -> Prediction:
        """Best-source prediction for one candidate."""
        telemetry = self._load_telemetry()
        hit = telemetry.get((provider, model))
        if hit is not None and hit[3] >= self.min_samples and hit[0] > 0:
            tps, ttft_ms, p_fail, _ = hit
            return Prediction(
                provider, model, self._e_time(ttft_ms, tps, p_fail, output_tokens),
                ttft_ms, tps, p_fail, _CONFIDENCE["telemetry"], "telemetry",
            )
        leaderboard = self._load_leaderboard()
        hit = leaderboard.get((provider, model))
        if hit is not None:
            ttft_ms, tps = hit
            return Prediction(
                provider, model, self._e_time(ttft_ms, tps, DEFAULT_P_FAIL, output_tokens),
                ttft_ms, tps, DEFAULT_P_FAIL, _CONFIDENCE["leaderboard"], "leaderboard",
            )
        speeds = self._load_speeds()
        hit = speeds.get(provider)
        if hit is not None:
            ttft_ms, tps = hit
            return Prediction(
                provider, model, self._e_time(ttft_ms, tps, DEFAULT_P_FAIL, output_tokens),
                ttft_ms, tps, DEFAULT_P_FAIL, _CONFIDENCE["speeds"], "speeds",
            )
        return Prediction(
            provider, model,
            self._e_time(DEFAULT_TTFT_MS, DEFAULT_TPS, DEFAULT_P_FAIL, output_tokens),
            DEFAULT_TTFT_MS, DEFAULT_TPS, DEFAULT_P_FAIL,
            DEFAULT_CONFIDENCE, "default",
        )

    def predict_many(
        self, candidates: Iterable[Tuple[str, str]], output_tokens: int = 200
    ) -> List[Prediction]:
        """Predict for many candidates; aggregates load once per window."""
        # touch all three sources once so subsequent predict() calls are cache hits
        self._load_telemetry()
        self._load_leaderboard()
        self._load_speeds()
        return [self.predict(p, m, output_tokens) for p, m in candidates]


def _demo() -> None:
    """Self-check: formula exactness + full source cascade on a temp DB."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db = tf.name
    try:
        conn = sqlite3.connect(db)
        conn.execute(
            """CREATE TABLE llm_events (
                ts_start REAL, provider TEXT, model TEXT, status TEXT,
                tps REAL, ttft_ms REAL)"""
        )
        now = time.time()
        # 10 successes @ tps=100, ttft=500ms, 0 failures -> p_fail=0.0
        for i in range(10):
            conn.execute(
                "INSERT INTO llm_events VALUES (?, ?, ?, ?, ?, ?)",
                (now - i * 60, "tp", "m1", "success", 100.0, 500.0),
            )
        conn.commit()
        conn.close()

        pred = LatencyPredictor(telemetry_db=db, leaderboard_csv="", speeds_path="").predict(
            "tp", "m1", output_tokens=100
        )
        expected = (500.0 + (100 / 100.0) * 1000.0) / 1.0  # 1500.0
        assert abs(pred.e_time_ms - expected) < 1e-6, pred
        assert pred.source == "telemetry" and pred.confidence == 1.0

        # min_samples gate: 10 samples < 11 -> falls through to default
        pred2 = LatencyPredictor(
            telemetry_db=db, leaderboard_csv="", speeds_path="", min_samples=11
        ).predict("tp", "m1", output_tokens=100)
        assert pred2.source == "default", pred2

        # missing db -> default source, sane values
        pred3 = LatencyPredictor(
            telemetry_db="/nonexistent/x.db", leaderboard_csv="", speeds_path=""
        ).predict("x", "y", output_tokens=100)
        assert pred3.source == "default" and pred3.e_time_ms > 0
        print("latency_predictor self-check: OK")
    finally:
        os.unlink(db)


if __name__ == "__main__":
    _demo()
