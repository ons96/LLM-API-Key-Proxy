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

    # Active rate-limit blocks (latest row per provider/model with a future
    # reset_time). Feeds router cooldown decisions + phase-router diagnostics.
    cursor.execute(
        """
        SELECT provider, model, limit_type, current_count, limit_limit,
               reset_time, last_updated
        FROM rate_limits
        WHERE reset_time IS NOT NULL
          AND (provider, model, last_updated) IN (
              SELECT provider, model, MAX(last_updated)
              FROM rate_limits
              GROUP BY provider, model
          )
    """
    )
    rate_limits = {}
    now = datetime.now()
    for row in cursor.fetchall():
        reset_time = row["reset_time"]
        try:
            reset_dt = datetime.fromisoformat(reset_time)
            retry_after_ms = max(0, int((reset_dt - now).total_seconds() * 1000))
        except (ValueError, TypeError):
            retry_after_ms = None
        rate_limits[f"{row['provider']}/{row['model']}"] = {
            "provider": row["provider"],
            "model": row["model"],
            "limit_type": row["limit_type"],
            "current_count": row["current_count"],
            "limit_limit": row["limit_limit"],
            "reset_time": reset_time,
            "retry_after_ms": retry_after_ms,
            "last_updated": row["last_updated"],
        }

    # Latest health snapshot per provider/model (most recent check).
    cursor.execute(
        """
        SELECT provider, model, is_healthy, failure_rate,
               consecutive_failures, last_check_time, last_success_time
        FROM provider_health
        WHERE (provider, model, last_check_time) IN (
            SELECT provider, model, MAX(last_check_time)
            FROM provider_health
            GROUP BY provider, model
        )
    """
    )
    health = {}
    for row in cursor.fetchall():
        health[f"{row['provider']}/{row['model'] or '*'}"] = {
            "provider": row["provider"],
            "model": row["model"],
            "is_healthy": bool(row["is_healthy"]),
            "failure_rate": row["failure_rate"] or 0,
            "consecutive_failures": row["consecutive_failures"] or 0,
            "last_check_time": row["last_check_time"],
            "last_success_time": row["last_success_time"],
        }

    conn.close()

    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "lookback_hours": hours,
        "total_providers": len(set(p.split("/")[0] for p in providers.keys())),
        "total_models": len(providers),
        "providers": providers,
        "rate_limits": rate_limits,
        "provider_health": health,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Exported {len(providers)} provider/model combinations to {OUTPUT_PATH}")
    return summary


if __name__ == "__main__":
    hours = int(sys.argv[1]) if len(sys.argv) > 1 else 168
    export_telemetry_summary(hours)
