"""
Telemetry Analyzer

Reads VPS telemetry (SQLite or JSON) and computes:
- Success rates per provider/model
- Average latency
- TPS metrics
- Composite performance scores
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class TelemetryAnalyzer:
    """Analyzes telemetry data to compute provider/model performance scores."""

    def __init__(self, telemetry_path: Path, hours: int = 168):
        """
        Initialize analyzer.

        Args:
            telemetry_path: Path to telemetry JSON or SQLite DB
            hours: Lookback window in hours (default: 7 days)
        """
        self.telemetry_path = Path(telemetry_path)
        self.hours = hours
        self._data = None

    def load_telemetry(self) -> Dict[str, Any]:
        """Load telemetry data from JSON or SQLite."""
        if self._data is not None:
            return self._data

        if self.telemetry_path.suffix == ".json":
            self._data = self._load_from_json()
        elif self.telemetry_path.suffix == ".db":
            self._data = self._load_from_sqlite()
        else:
            logger.warning(f"Unknown telemetry format: {self.telemetry_path}")
            self._data = {"providers": {}}

        return self._data

    def _load_from_json(self) -> Dict[str, Any]:
        """Load telemetry from JSON file."""
        try:
            with open(self.telemetry_path) as f:
                data = json.load(f)
            logger.info(f"Loaded telemetry from JSON: {self.telemetry_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load JSON telemetry: {e}")
            return {"providers": {}}

    def _load_from_sqlite(self) -> Dict[str, Any]:
        """Load telemetry from SQLite database."""
        try:
            conn = sqlite3.connect(str(self.telemetry_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cutoff = datetime.now() - timedelta(hours=self.hours)

            cursor.execute(
                """
                SELECT 
                    provider,
                    model,
                    COUNT(*) as total_calls,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_calls,
                    AVG(response_time_ms) as avg_latency_ms,
                    AVG(tokens_per_second) as avg_tps
                FROM api_calls
                WHERE timestamp >= ?
                GROUP BY provider, model
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
                    "success_rate": round(success_rate, 3),
                    "avg_latency_ms": round(row["avg_latency_ms"] or 5000, 1),
                    "avg_tps": round(row["avg_tps"] or 0, 1),
                }

            conn.close()
            logger.info(f"Loaded telemetry from SQLite: {len(providers)} entries")
            return {"providers": providers, "generated_at": datetime.now().isoformat()}

        except Exception as e:
            logger.error(f"Failed to load SQLite telemetry: {e}")
            return {"providers": {}}

    def get_provider_scores(self) -> Dict[str, Dict[str, float]]:
        """
        Compute performance scores per provider/model.

        Returns:
            Dict mapping "provider/model" to score dict:
            {
                "success_rate": 0.95,
                "avg_latency_ms": 1200,
                "avg_tps": 85.5,
                "sample_count": 150
            }
        """
        data = self.load_telemetry()
        providers = data.get("providers", {})

        scores = {}
        for key, metrics in providers.items():
            if isinstance(metrics, dict):
                scores[key] = {
                    "success_rate": metrics.get("success_rate", 0.5),
                    "avg_latency_ms": metrics.get("avg_latency_ms", 5000),
                    "avg_tps": metrics.get("avg_tps", 0),
                    "sample_count": metrics.get("total_calls", 0),
                }

        return scores

    def get_top_performers(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get top-performing provider/model combinations.

        Ranked by: success_rate * 0.5 + (1 - latency/30000) * 0.3 + tps/1000 * 0.2
        """
        scores = self.get_provider_scores()

        ranked = []
        for key, metrics in scores.items():
            if metrics["sample_count"] < 5:
                continue

            success = metrics["success_rate"]
            latency = max(0, 1 - metrics["avg_latency_ms"] / 30000)
            tps = min(1, metrics["avg_tps"] / 1000)

            composite = success * 0.5 + latency * 0.3 + tps * 0.2

            ranked.append(
                {
                    "provider_model": key,
                    "composite_score": round(composite, 3),
                    **metrics,
                }
            )

        ranked.sort(key=lambda x: x["composite_score"], reverse=True)
        return ranked[:limit]

    def get_reliability_report(self) -> Dict[str, Any]:
        """Generate reliability report for all providers."""
        scores = self.get_provider_scores()

        by_provider = defaultdict(list)
        for key, metrics in scores.items():
            provider = key.split("/")[0] if "/" in key else "unknown"
            by_provider[provider].append(metrics)

        report = {}
        for provider, metrics_list in by_provider.items():
            total_calls = sum(m["sample_count"] for m in metrics_list)
            avg_success = (
                sum(m["success_rate"] * m["sample_count"] for m in metrics_list)
                / total_calls
                if total_calls > 0
                else 0
            )
            avg_latency = sum(m["avg_latency_ms"] for m in metrics_list) / len(
                metrics_list
            )

            report[provider] = {
                "total_calls": total_calls,
                "avg_success_rate": round(avg_success, 3),
                "avg_latency_ms": round(avg_latency, 1),
                "model_count": len(metrics_list),
                "status": "healthy"
                if avg_success > 0.9
                else "degraded"
                if avg_success > 0.7
                else "unhealthy",
            }

        return report


def analyze_telemetry_file(
    telemetry_path: str, output_path: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to analyze a telemetry file."""
    analyzer = TelemetryAnalyzer(Path(telemetry_path))

    report = {
        "generated_at": datetime.now().isoformat(),
        "lookback_hours": analyzer.hours,
        "provider_scores": analyzer.get_provider_scores(),
        "top_performers": analyzer.get_top_performers(),
        "reliability_report": analyzer.get_reliability_report(),
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved telemetry report to {output_path}")

    return report


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else None
        analyze_telemetry_file(path, output)
    else:
        print("Usage: python telemetry_analyzer.py <telemetry_path> [output_path]")
