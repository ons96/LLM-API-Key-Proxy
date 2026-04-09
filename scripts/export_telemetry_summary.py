#!/usr/bin/env python3
"""Export telemetry summary from VPS SQLite database to JSON."""

import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = "/tmp/llm_proxy_telemetry.db"
OUTPUT_PATH = Path(__file__).parent.parent / "config" / "telemetry_summary.json"


def export_telemetry_summary(hours: int = 168):
    """Export telemetry summary for the last N hours."""
    if not Path(DB_PATH).exists():
        print(f"Database not found: {DB_PATH}")
        return None

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cutoff = datetime.now() - timedelta(hours=hours)

    cursor.execute(
        """
        SELECT 
            provider,
            model,
            COUNT(*) as total_calls,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_calls,
            AVG(response_time_ms) as avg_latency_ms,
            AVG(tokens_per_second) as avg_tps,
            MIN(response_time_ms) as min_latency_ms,
            MAX(response_time_ms) as max_latency_ms
        FROM api_calls
        WHERE timestamp >= ?
        GROUP BY provider, model
        HAVING total_calls > 0
    """,
        (cutoff.isoformat(),),
    )

    providers = {}
    for row in cursor.fetchall():
        key = f"{row['provider']}/{row['model']}"
        success_rate = (
            row["successful_calls"] / row["total_calls"]
            if row["total_calls"] > 0
            else 0
        )

        providers[key] = {
            "provider": row["provider"],
            "model": row["model"],
            "total_calls": row["total_calls"],
            "successful_calls": row["successful_calls"],
            "success_rate": round(success_rate, 4),
            "avg_latency_ms": round(row["avg_latency_ms"] or 0, 1),
            "avg_tps": round(row["avg_tps"] or 0, 1),
            "min_latency_ms": round(row["min_latency_ms"] or 0, 1),
            "max_latency_ms": round(row["max_latency_ms"] or 0, 1),
        }

    conn.close()

    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "lookback_hours": hours,
        "total_providers": len(set(p.split("/")[0] for p in providers.keys())),
        "total_models": len(providers),
        "providers": providers,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Exported {len(providers)} provider/model combinations to {OUTPUT_PATH}")
    return summary


if __name__ == "__main__":
    hours = int(sys.argv[1]) if len(sys.argv) > 1 else 168
    export_telemetry_summary(hours)
