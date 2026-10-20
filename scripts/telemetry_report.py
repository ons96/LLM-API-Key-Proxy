#!/usr/bin/env python3
"""Per-provider failure rate and error taxonomy from the gateway telemetry DB.

Read-only. Used by M0 (measurement 2: actual failure rate per provider) and by
RESULT.md to quote reproducible numbers.

Usage:
  python3 scripts/telemetry_report.py [--db /tmp/llm_proxy_telemetry.db] [--top 20]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="/tmp/llm_proxy_telemetry.db")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args(argv)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)

    window = conn.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM api_calls").fetchone()
    print(f"window: first={window[0]} last={window[1]} calls={window[2]}")
    print()

    header = f"{'provider':26s} {'calls':>6s} {'ok':>6s} {'fail':>5s} {'fail%':>6s}"
    print(header)
    print("-" * len(header))
    rows = conn.execute(
        "SELECT provider, COUNT(*) n, SUM(success) ok, SUM(1 - success) f"
        " FROM api_calls GROUP BY provider ORDER BY n DESC LIMIT ?",
        (args.top,),
    ).fetchall()
    for provider, n, ok, f in rows:
        pct = 100.0 * (f or 0) / n if n else 0.0
        print(f"{str(provider):26s} {n:6d} {(ok or 0):6d} {(f or 0):5d} {pct:6.1f}")

    print()
    print("error_reason histogram (failures only):")
    for reason, n in conn.execute(
        "SELECT COALESCE(error_reason, '(none)'), COUNT(*) FROM api_calls"
        " WHERE success = 0 GROUP BY 1 ORDER BY 2 DESC LIMIT 10"
    ):
        print(f"  {str(reason)[:60]:62s} {n}")

    print()
    print("ttft by provider (successful streaming calls, ms):")
    for provider, n, p50 in conn.execute(
        "SELECT provider, COUNT(*), 0 FROM api_calls WHERE success = 1"
        " GROUP BY provider ORDER BY 2 DESC LIMIT 10"
    ):
        values = [
            r[0]
            for r in conn.execute(
                "SELECT time_to_first_token_ms FROM api_calls"
                " WHERE provider = ? AND success = 1 AND time_to_first_token_ms IS NOT NULL"
                " ORDER BY time_to_first_token_ms",
                (provider,),
            )
        ]
        if not values:
            print(f"  {str(provider):26s} n={n:<5d} no ttft samples")
            continue
        mid = values[len(values) // 2]
        print(f"  {str(provider):26s} n={n:<5d} ttft_p50={mid:.0f}ms samples={len(values)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
