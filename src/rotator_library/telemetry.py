"""
Telemetry and monitoring system for LLM API Proxy.
Tracks API call metrics, success/failure rates, latency, and provider performance.
"""

import time
import sqlite3
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TelemetryManager:
    """Manages telemetry collection and storage for API calls."""

    def __init__(self, db_path: str = "/tmp/llm_proxy_telemetry.db"):
        """Initialize telemetry manager with SQLite database."""
        self.db_path = db_path
        self._init_db()
        logger.info(f"TelemetryManager initialized: {db_path}")

    def _init_db(self):
        """Initialize database schema if not exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                error_reason TEXT,
                response_time_ms INTEGER,
                time_to_first_token_ms INTEGER,
                tokens_per_second REAL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost_estimate_usd REAL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON api_calls(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_provider_model ON api_calls(provider, model)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_success ON api_calls(success)
        """)

        # Rate limit tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rate_limits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                limit_type TEXT NOT NULL,
                current_count INTEGER DEFAULT 0,
                limit_limit INTEGER,
                reset_time TEXT,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rate_limit_provider_model ON rate_limits(provider, model)
        """)

        # Provider health tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS provider_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                model TEXT,
                is_healthy BOOLEAN DEFAULT 1,
                last_check_time TEXT,
                failure_rate REAL DEFAULT 0,
                consecutive_failures INTEGER DEFAULT 0,
                last_success_time TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_health_provider ON provider_health(provider, model)
        """)

        # TPS metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tps_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                tps REAL NOT NULL,
                window_minutes INTEGER NOT NULL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Search API credit tracking tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_api_credits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                api_key_hash TEXT NOT NULL,
                credits_remaining INTEGER DEFAULT 0,
                credits_used_total INTEGER DEFAULT 0,
                monthly_allowance INTEGER DEFAULT 1000,
                is_exhausted BOOLEAN DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reset_date TEXT,
                UNIQUE(provider, api_key_hash)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                provider TEXT NOT NULL,
                api_key_hash TEXT NOT NULL,
                search_type TEXT NOT NULL,
                credits_consumed INTEGER DEFAULT 1,
                query_hash TEXT,
                success BOOLEAN DEFAULT 1,
                error_message TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_search_credits_provider 
            ON search_api_credits(provider, is_exhausted)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_search_usage_time 
            ON search_api_usage(provider, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tps_provider_model ON tps_metrics(provider, model)
        """)

        conn.commit()
        conn.close()

    def record_call(
        self,
        provider: str,
        model: str,
        success: bool,
        response_time_ms: int,
        error_reason: Optional[str] = None,
        time_to_first_token_ms: Optional[int] = None,
        tokens_per_second: Optional[float] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cost_estimate_usd: Optional[float] = None,
    ):
        """Record a single API call with metrics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO api_calls (
                    provider, model, success, error_reason,
                    response_time_ms, time_to_first_token_ms, tokens_per_second,
                    input_tokens, output_tokens, cost_estimate_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    provider,
                    model,
                    success,
                    error_reason,
                    response_time_ms,
                    time_to_first_token_ms,
                    tokens_per_second,
                    input_tokens,
                    output_tokens,
                    cost_estimate_usd,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to record telemetry: {e}")

    def get_metrics_summary(
        self,
        hours: int = 24,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get aggregated metrics summary."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since = datetime.now() - timedelta(hours=hours)

        where_clause = "WHERE timestamp >= ?"
        params = [since]

        if provider:
            where_clause += " AND provider = ?"
            params.append(provider)

        if model:
            where_clause += " AND model = ?"
            params.append(model)

        # Total calls
        cursor.execute(f"SELECT COUNT(*) FROM api_calls {where_clause}", params)
        total_calls = cursor.fetchone()[0]

        # Success rate
        cursor.execute(
            f"""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful
            FROM api_calls {where_clause}
        """,
            params,
        )
        row = cursor.fetchone()
        success_rate = (row[1] / row[0] * 100) if row[0] > 0 else 0

        # Average response time
        cursor.execute(
            f"""
            SELECT AVG(response_time_ms)
            FROM api_calls {where_clause} AND response_time_ms IS NOT NULL
        """,
            params,
        )
        avg_response_time = cursor.fetchone()[0] or 0

        # Error distribution
        cursor.execute(
            f"""
            SELECT error_reason, COUNT(*) as count
            FROM api_calls {where_clause} AND success = 0 AND error_reason IS NOT NULL
            GROUP BY error_reason
            ORDER BY count DESC
        """,
            params,
        )
        error_distribution = dict(cursor.fetchall())

        # Provider breakdown
        cursor.execute(
            f"""
            SELECT provider, COUNT(*)
            FROM api_calls {where_clause}
            GROUP BY provider
            ORDER BY COUNT(*) DESC
        """,
            params,
        )
        provider_breakdown = dict(cursor.fetchall())

        conn.close()

        return {
            "period_hours": hours,
            "total_calls": total_calls,
            "success_rate_percent": round(success_rate, 2),
            "avg_response_time_ms": round(avg_response_time, 2),
            "error_distribution": error_distribution,
            "provider_breakdown": provider_breakdown,
        }

    def get_provider_stats(self, provider: str, hours: int = 24) -> Dict[str, Any]:
        """Get detailed stats for a specific provider."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since = datetime.now() - timedelta(hours=hours)

        cursor.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                AVG(response_time_ms) as avg_response_time
            FROM api_calls
            WHERE provider = ? AND timestamp >= ?
        """,
            (provider, since),
        )

        row = cursor.fetchone()

        conn.close()

        return {
            "provider": provider,
            "period_hours": hours,
            "total_calls": row[0] or 0,
            "successful_calls": row[1] or 0,
            "success_rate": round((row[1] / row[0] * 100) if row[0] > 0 else 0, 2),
            "avg_response_time_ms": round(row[2] or 0, 2),
        }

    def cleanup_old_data(self, days: int = 7):
        """Remove telemetry data older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM api_calls WHERE timestamp < ?", (cutoff,))

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        logger.info(f"Cleaned up {deleted} old telemetry records")

    def increment_rate_limit(
        self,
        provider: str,
        model: str,
        limit_type: str = "rpm",
        limit_count: Optional[int] = None,
    ):
        """Increment rate limit counter for provider/model."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO rate_limits
                (provider, model, limit_type, current_count, limit_limit, last_updated)
                VALUES (?, ?, ?, COALESCE((SELECT current_count FROM rate_limits
                    WHERE provider = ? AND model = ? AND limit_type = ?), 0) + 1, ?, CURRENT_TIMESTAMP)
            """,
                (provider, model, limit_type, provider, model, limit_type, limit_count),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to increment rate limit: {e}")

    def check_rate_limit(
        self,
        provider: str,
        model: str,
        limit_type: str = "rpm",
    ) -> tuple[bool, Optional[int]]:
        """Check if provider/model is rate limited. Returns (is_limited, retry_after_ms)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT current_count, limit_limit, reset_time
                FROM rate_limits
                WHERE provider = ? AND model = ? AND limit_type = ?
                ORDER BY last_updated DESC
                LIMIT 1
            """,
                (provider, model, limit_type),
            )

            row = cursor.fetchone()
            conn.close()

            if not row:
                return False, None

            current, limit_value, reset_time = row

            if limit_value and current >= limit_value:
                if reset_time:
                    try:
                        reset_dt = datetime.fromisoformat(reset_time)
                        retry_after = max(
                            0, int((reset_dt - datetime.now()).total_seconds() * 1000)
                        )
                        return True, retry_after
                    except:
                        pass
                return True, 60000

            return False, None

        except Exception as e:
            logger.error(f"Failed to check rate limit: {e}")
            return False, None

    def reset_rate_limits(self, provider: Optional[str] = None):
        """Reset rate limit counters for specific provider or all."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if provider:
                cursor.execute(
                    "UPDATE rate_limits SET current_count = 0 WHERE provider = ?",
                    (provider,),
                )
            else:
                cursor.execute("UPDATE rate_limits SET current_count = 0")

            deleted = cursor.rowcount
            conn.commit()
            conn.close()

            logger.info(f"Reset {deleted} rate limit counters")

        except Exception as e:
            logger.error(f"Failed to reset rate limits: {e}")

    def update_provider_health(
        self,
        provider: str,
        model: Optional[str],
        is_healthy: bool,
        failure_rate: float = 0,
    ):
        """Update provider health status."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO provider_health
                (provider, model, is_healthy, last_check_time, failure_rate,
                 consecutive_failures, last_success_time)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?,
                 CASE WHEN ? = 1 THEN 0 ELSE consecutive_failures + 1 end,
                 CASE WHEN ? = 1 THEN CURRENT_TIMESTAMP ELSE last_success_time end)
            """,
                (
                    provider,
                    model,
                    is_healthy,
                    failure_rate,
                    int(is_healthy),
                    int(is_healthy),
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to update provider health: {e}")

    def get_provider_health(
        self,
        provider: str,
        model: Optional[str] = None,
    ) -> dict:
        """Get provider health status."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if model:
                cursor.execute(
                    """
                    SELECT is_healthy, failure_rate, consecutive_failures, last_success_time
                    FROM provider_health
                    WHERE provider = ? AND model = ?
                    ORDER BY last_check_time DESC
                    LIMIT 1
                """,
                    (provider, model),
                )
            else:
                cursor.execute(
                    """
                    SELECT is_healthy, AVG(failure_rate), MAX(consecutive_failures), MAX(last_success_time)
                    FROM provider_health
                    WHERE provider = ?
                """,
                    (provider,),
                )

            row = cursor.fetchone()
            conn.close()

            if row:
                is_healthy, failure_rate, cons_failures, last_success = row
                return {
                    "provider": provider,
                    "model": model,
                    "is_healthy": bool(is_healthy),
                    "failure_rate": failure_rate or 0,
                    "consecutive_failures": cons_failures or 0,
                    "last_success_time": last_success,
                }

            return {
                "provider": provider,
                "model": model,
                "is_healthy": True,
                "failure_rate": 0,
                "consecutive_failures": 0,
                "last_success_time": None,
            }

        except Exception as e:
            logger.error(f"Failed to get provider health: {e}")
            return {"provider": provider, "is_healthy": True, "error": str(e)}

    def record_tps(
        self,
        provider: str,
        model: str,
        tps: float,
        window_minutes: int = 1,
    ):
        """Record TPS metric for provider/model."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO tps_metrics
                (provider, model, tps, window_minutes, timestamp)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (provider, model, tps, window_minutes),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to record TPS: {e}")

    def register_search_api_key(
        self, provider: str, api_key: str, monthly_allowance: int = 1000
    ):
        """Register a search API key for credit tracking."""
        import hashlib

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO search_api_credits
                (provider, api_key_hash, credits_remaining, monthly_allowance,
                 is_exhausted, last_updated)
                VALUES (?, ?, ?, ?, 0, CURRENT_TIMESTAMP)
            """,
                (provider, key_hash, monthly_allowance, monthly_allowance),
            )

            conn.commit()
            conn.close()
            logger.info(f"Registered search API key for {provider}")

        except Exception as e:
            logger.error(f"Failed to register search API key: {e}")

    def record_search_usage(
        self,
        provider: str,
        api_key: str,
        search_type: str,
        credits_consumed: int = 1,
        query: str = "",
        success: bool = True,
        error_message: str = None,
    ):
        """Record search API usage and update credit balance."""
        import hashlib

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16] if query else None

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO search_api_usage
                (provider, api_key_hash, search_type, credits_consumed, query_hash, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    provider,
                    key_hash,
                    search_type,
                    credits_consumed,
                    query_hash,
                    success,
                    error_message,
                ),
            )

            cursor.execute(
                """
                UPDATE search_api_credits
                SET credits_used_total = credits_used_total + ?,
                    credits_remaining = MAX(0, credits_remaining - ?),
                    is_exhausted = CASE WHEN credits_remaining <= ? THEN 1 ELSE 0 END,
                    last_updated = CURRENT_TIMESTAMP
                WHERE provider = ? AND api_key_hash = ?
            """,
                (
                    credits_consumed,
                    credits_consumed,
                    credits_consumed,
                    provider,
                    key_hash,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to record search usage: {e}")

    def get_search_credit_status(self, provider: str, api_key: str) -> dict:
        """Get current credit status for a search provider API key."""
        import hashlib

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT credits_remaining, credits_used_total, monthly_allowance,
                       is_exhausted, last_updated
                FROM search_api_credits
                WHERE provider = ? AND api_key_hash = ?
            """,
                (provider, key_hash),
            )

            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    "provider": provider,
                    "credits_remaining": row[0],
                    "credits_used_total": row[1],
                    "monthly_allowance": row[2],
                    "is_exhausted": bool(row[3]),
                    "last_updated": row[4],
                }
            return {"provider": provider, "credits_remaining": 0, "is_exhausted": True}

        except Exception as e:
            logger.error(f"Failed to get search credit status: {e}")
            return {"provider": provider, "error": str(e), "is_exhausted": True}

    def check_search_credits_available(
        self, provider: str, api_key: str, required_credits: int = 1
    ) -> bool:
        """Check if a search provider has enough credits available."""
        status = self.get_search_credit_status(provider, api_key)
        return (
            not status.get("is_exhausted", True)
            and status.get("credits_remaining", 0) >= required_credits
        )

    def mark_search_key_exhausted(self, provider: str, api_key: str):
        """Mark an API key as exhausted (out of credits)."""
        import hashlib

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE search_api_credits
                SET is_exhausted = 1, credits_remaining = 0, last_updated = CURRENT_TIMESTAMP
                WHERE provider = ? AND api_key_hash = ?
            """,
                (provider, key_hash),
            )

            conn.commit()
            conn.close()
            logger.warning(f"Marked {provider} API key as exhausted")

        except Exception as e:
            logger.error(f"Failed to mark search key exhausted: {e}")


# Global telemetry instance
_telemetry_manager: Optional[TelemetryManager] = None


def get_telemetry_manager() -> TelemetryManager:
    """Get or create the global telemetry manager instance."""
    global _telemetry_manager
    if _telemetry_manager is None:
        _telemetry_manager = TelemetryManager()
    return _telemetry_manager
