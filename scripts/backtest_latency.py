"""Backtest latency predictions against recorded llm_events (#477 AC).

Replays every event that has total_ms + provider/model + completion_tokens
and compares the predictor's E[time] (using the event's own completion
tokens as the output estimate) against the actual end-to-end latency.
Reports MAPE. AC gate: MAPE < 40% on VPS-40 before enabling.

Filter: only rows where total_ms != ttft_ms are compared. The telemetry
logger sets total_ms == ttft_ms on streaming rows (stream duration is not
added), so those rows carry no true end-to-end latency; rows where the two
differ do (verified: glm-5.2/groq rows match the predictor to ~1%).

Usage:
    PYTHONPATH=src python3 scripts/backtest_latency.py \
        [--db /dev/shm/telemetry.db] [--window-h 24] [--min-samples 5] \
        [--leaderboard ~/llm-speedrun/data/leaderboard.csv]

Stdlib only. Read-only on the telemetry DB.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proxy_app.routing.latency_predictor import (  # noqa: E402
    DEFAULT_LEADERBOARD_CSV,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_TELEMETRY_DB,
    DEFAULT_WINDOW_H,
    LatencyPredictor,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DEFAULT_TELEMETRY_DB)
    ap.add_argument("--window-h", type=int, default=DEFAULT_WINDOW_H)
    ap.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    ap.add_argument(
        "--leaderboard",
        default=os.environ.get("LATENCY_LEADERBOARD_CSV", DEFAULT_LEADERBOARD_CSV),
        help="speedrun leaderboard.csv (VPS: ~/llm-speedrun/data/leaderboard.csv)",
    )
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"error: telemetry db not found: {args.db}")
        return 1

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT provider, model, completion_tokens, total_ms, status,
               COALESCE(NULLIF(concrete_provider, ''), provider) AS c_provider,
               COALESCE(NULLIF(concrete_model, ''), model) AS c_model
        FROM llm_events
        WHERE total_ms IS NOT NULL AND completion_tokens IS NOT NULL
              AND completion_tokens > 0 AND total_ms > 0
              AND total_ms != ttft_ms
        ORDER BY ts_start DESC
        LIMIT 2000
        """
    ).fetchall()
    conn.close()

    if not rows:
        print("no events with total_ms + completion_tokens found")
        return 2

    predictor = LatencyPredictor(
        telemetry_db=args.db,
        leaderboard_csv=args.leaderboard,
        min_samples=args.min_samples,
        window_h=args.window_h,
    )

    errors = []
    by_source: dict = {}
    for r in rows:
        pred = predictor.predict(r["c_provider"], r["c_model"], output_tokens=int(r["completion_tokens"]))
        actual = float(r["total_ms"])
        err = abs(pred.e_time_ms - actual) / actual
        errors.append(err)
        by_source.setdefault(pred.source, []).append(err)

    mape = (sum(errors) / len(errors)) * 100.0
    print(f"events={len(errors)}  MAPE={mape:.1f}%  (AC gate: <40%)")
    for src, errs in sorted(by_source.items()):
        print(f"  src={src:<10} n={len(errs):<4} MAPE={sum(errs)/len(errs)*100:.1f}%")
    return 0 if mape < 40.0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
