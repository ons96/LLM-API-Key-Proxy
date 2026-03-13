"""
TPS (tokens-per-second) self-tracking export capability.

Reads the telemetry SQLite database, aggregates TPS data per (provider, model)
pair over a configurable time window, and exports statistics to JSON.
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_PATH = "/tmp/llm_proxy_telemetry.db"
DEFAULT_EXPORT_PATH = "config/self_tracked_tps.json"

# Minimum number of successful samples required before reporting stats
MIN_SAMPLE_COUNT = 5


def _get_project_root() -> Path:
    """Get the project root directory for resolving relative config paths."""
    import sys

    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    # Walk up from this file: src/rotator_library/tps_export.py -> project root
    return Path(__file__).resolve().parent.parent.parent


def _percentile(sorted_values: list, p: float) -> float:
    """Calculate the p-th percentile from a sorted list of values."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    k = (p / 100.0) * (n - 1)
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_values[-1]
    d = k - f
    return sorted_values[f] + d * (sorted_values[c] - sorted_values[f])


def _median(sorted_values: list) -> float:
    """Calculate the median from a sorted list of values."""
    return _percentile(sorted_values, 50.0)


def aggregate_tps_stats(
    db_path: str = DEFAULT_DB_PATH,
    days: int = 7,
    min_samples: int = MIN_SAMPLE_COUNT,
) -> List[Dict[str, Any]]:
    """
    Aggregate TPS statistics per (provider, model) pair from the telemetry DB.

    Args:
        db_path: Path to the SQLite telemetry database.
        days: Number of days to look back (default: 7).
        min_samples: Minimum sample count required to include a pair (default: 5).

    Returns:
        A list of dicts with aggregated TPS stats per provider/model pair.
    """
    db_file = Path(db_path)
    if not db_file.exists():
        logger.warning(f"Telemetry database not found at {db_path}")
        return []

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Calculate the cutoff timestamp
        cutoff = datetime.now() - timedelta(days=days)

        # Query successful completions that have TPS data within the time window
        cursor.execute(
            """
            SELECT provider, model, tokens_per_second, time_to_first_token_ms
            FROM api_calls
            WHERE success = 1
              AND tokens_per_second IS NOT NULL
              AND tokens_per_second > 0
              AND timestamp >= ?
            ORDER BY provider, model
            """,
            (cutoff.isoformat(),),
        )

        rows = cursor.fetchall()
        conn.close()

    except sqlite3.Error as e:
        logger.error(f"Failed to read telemetry database: {e}")
        return []

    if not rows:
        logger.info("No successful TPS data found in telemetry database")
        return []

    # Group by (provider, model)
    groups: Dict[tuple, Dict[str, list]] = {}
    for row in rows:
        key = (row["provider"], row["model"])
        if key not in groups:
            groups[key] = {"tps_values": [], "ttft_values": []}
        groups[key]["tps_values"].append(row["tokens_per_second"])
        if row["time_to_first_token_ms"] is not None:
            groups[key]["ttft_values"].append(row["time_to_first_token_ms"])

    # Build aggregated results
    now_iso = datetime.now(timezone.utc).isoformat()
    results = []

    for (provider, model), data in sorted(groups.items()):
        tps_values = data["tps_values"]
        ttft_values = data["ttft_values"]

        sample_count = len(tps_values)
        if sample_count < min_samples:
            logger.debug(
                f"Skipping {provider}/{model}: only {sample_count} samples "
                f"(minimum {min_samples})"
            )
            continue

        tps_sorted = sorted(tps_values)
        avg_tps = sum(tps_values) / len(tps_values)
        median_tps = _median(tps_sorted)
        p95_tps = _percentile(tps_sorted, 95.0)

        avg_ttft_ms = None
        if ttft_values:
            avg_ttft_ms = round(sum(ttft_values) / len(ttft_values), 1)

        results.append(
            {
                "provider_name": provider,
                "model": model,
                "median_tps": round(median_tps, 1),
                "avg_tps": round(avg_tps, 1),
                "p95_tps": round(p95_tps, 1),
                "avg_ttft_ms": avg_ttft_ms,
                "sample_count": sample_count,
                "last_updated": now_iso,
            }
        )

    return results


def export_tps_stats(
    db_path: str = DEFAULT_DB_PATH,
    export_path: Optional[str] = None,
    days: int = 7,
    min_samples: int = MIN_SAMPLE_COUNT,
) -> List[Dict[str, Any]]:
    """
    Aggregate TPS stats and export them to a JSON file.

    Args:
        db_path: Path to the SQLite telemetry database.
        export_path: Path for the output JSON file. If None or relative,
                     resolved relative to the project root.
        days: Number of days to look back (default: 7).
        min_samples: Minimum sample count required to include a pair (default: 5).

    Returns:
        The aggregated stats list (same data written to the file).
    """
    stats = aggregate_tps_stats(db_path=db_path, days=days, min_samples=min_samples)

    # Resolve export path
    if export_path is None:
        export_path = DEFAULT_EXPORT_PATH

    out_path = Path(export_path)
    if not out_path.is_absolute():
        out_path = _get_project_root() / out_path

    # Ensure the parent directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Exported TPS stats to {out_path} ({len(stats)} entries)")
    except OSError as e:
        logger.error(f"Failed to write TPS export file: {e}")

    return stats
