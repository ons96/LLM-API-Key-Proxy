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
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class TelemetryManager:
    """Manages telemetry collection and storage for API calls with connection pooling."""

    def __init__(self, db_path: str = "/tmp/llm_proxy_telemetry.db"):
        """Initialize telemetry manager with SQLite database."""
        self.db_path = db_path
        self._connection_pool = []
        self._max_connections = 5
        self._init_db()
        logger.info(f"TelemetryManager initialized: {db_path}")

    def _get_connection(self):
        """Get a connection from the pool or create a new one."""
        if self._connection_pool:
            return self._connection_pool.pop()
        return sqlite3.connect(self.db_path)

    def _return_connection(self, conn):
        """Return a connection to the pool."""
        if len(self._connection_pool) < self._max_connections:
            self._connection_pool.append(conn)
        else:
            conn.close()

    @contextmanager
    def _pooled_connection(self):
        """Context manager for pooled connections."""
        conn = self._get_connection()
        try:
            yield conn
        finally:
            self._return_connection(conn)

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

        # LLM provider credit tracking (one-time credit / daily renewable /
        # unlimited-rate-limited / freemium). Mirrors search_api_credits so
        # rotator can pick a gate-aware replacement on exhaustion.
        # See task-board #290 for free_type taxonomy.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_provider_credits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                api_key_hash TEXT NOT NULL,
                free_type TEXT NOT NULL DEFAULT 'unlimited_rate_limited',
                credits_remaining REAL DEFAULT -1,  /* -1 = unlimited */
                credits_used_total REAL DEFAULT 0,
                initial_allowance REAL DEFAULT -1,
                reset_period TEXT,                   /* daily|monthly|never */
                reset_date TEXT,                     /* ISO timestamp */
                is_exhausted BOOLEAN DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(provider, api_key_hash)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_provider_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                provider TEXT NOT NULL,
                api_key_hash TEXT NOT NULL,
                model TEXT,
                credits_consumed REAL DEFAULT 1.0,
                success BOOLEAN DEFAULT 1,
                error_message TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_llm_credits_provider
            ON llm_provider_credits(provider, is_exhausted)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_llm_usage_provider_time
            ON llm_provider_usage(provider, timestamp)
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
        """Record a single API call with metrics using pooled connection."""
        try:
            with self._pooled_connection() as conn:
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
        """Remove telemetry data older than specified days across all tables."""
        cutoff = datetime.now() - timedelta(days=days)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM api_calls WHERE timestamp < ?", (cutoff,))
        deleted = cursor.rowcount

        # rate_limits + provider_health + tps_metrics were added alongside
        # api_calls; purge them on the same rolling window so the DB stays
        # bounded on the 1GB VPS. rate_limits resets via new INSERTs rather
        # than UPDATE, so stale rows have no reset_time or expired ones.
        cursor.execute("DELETE FROM rate_limits WHERE last_updated < ?", (cutoff,))
        deleted += cursor.rowcount

        cursor.execute(
            "DELETE FROM provider_health WHERE last_check_time < ?", (cutoff,)
        )
        deleted += cursor.rowcount

        cursor.execute("DELETE FROM tps_metrics WHERE timestamp < ?", (cutoff,))
        deleted += cursor.rowcount

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

    def record_rate_limit_hit(
        self,
        provider: str,
        model: str,
        limit_type: str = "rpm",
        retry_after: float = 60.0,
        reset_time: Optional[str] = None,
    ):
        """Record an explicit rate-limit hit with a reset time.

        Complements ``increment_rate_limit`` (counter-only) by persisting
        ``reset_time`` so ``check_rate_limit`` can compute retry_after
        across process restarts. Called by RateLimitTracker when an
        upstream 429/usage-cap is observed.
        """
        try:
            if reset_time is None:
                reset_time = (
                    datetime.now() + timedelta(seconds=retry_after)
                ).isoformat()

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO rate_limits
                (provider, model, limit_type, current_count, reset_time,
                 last_updated)
                VALUES (?, ?, ?,
                        COALESCE(
                            (SELECT current_count FROM rate_limits
                             WHERE provider = ? AND model = ?
                               AND limit_type = ?),
                            0) + 1,
                        ?, CURRENT_TIMESTAMP)
            """,
                (
                    provider,
                    model,
                    limit_type,
                    provider,
                    model,
                    limit_type,
                    reset_time,
                ),
            )
            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to record rate limit hit: {e}")

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

            # If a future reset_time is recorded, treat as limited even when
            # limit_value is NULL (rate-limit hit persisted via
            # record_rate_limit_hit without a known numeric cap).
            if reset_time:
                try:
                    reset_dt = datetime.fromisoformat(reset_time)
                    # reset_time may be tz-aware (from rate_limiter's UTC
                    # timestamp) or naive (CURRENT_TIMESTAMP). Compare in a
                    # tz-consistent way to avoid aware-vs-naive TypeError.
                    if reset_dt.tzinfo is not None:
                        now_dt = datetime.now(reset_dt.tzinfo)
                    else:
                        now_dt = datetime.now()
                    retry_after = max(
                        0,
                        int((reset_dt - now_dt).total_seconds() * 1000),
                    )
                    if retry_after > 0:
                        return True, retry_after
                    # Reset time has passed -> not limited; fall through
                except (ValueError, TypeError):
                    pass

            if limit_value and current >= limit_value:
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
        """Update provider health status.

        provider_health has no UNIQUE constraint, so ``INSERT OR REPLACE``
        would always start consecutive_failures from NULL (-> 1). We
        read the latest existing consecutive_failures via a subquery so
        failures accumulate across checks.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO provider_health
                (provider, model, is_healthy, last_check_time, failure_rate,
                 consecutive_failures, last_success_time)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?,
                 CASE WHEN ? = 1
                      THEN 0
                      ELSE COALESCE(
                          (SELECT consecutive_failures
                             FROM provider_health
                            WHERE provider = ?
                              AND (model IS ? OR model = ?)
                            ORDER BY last_check_time DESC, rowid DESC
                            LIMIT 1),
                          0) + 1
                 END,
                 CASE WHEN ? = 1
                      THEN CURRENT_TIMESTAMP
                      ELSE NULL
                 END)
            """,
                (
                    provider,
                    model,
                    is_healthy,
                    failure_rate,
                    int(is_healthy),
                    provider,
                    model,
                    model,
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
                    ORDER BY last_check_time DESC, rowid DESC
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

    # ──────────────────────────────────────────────────────────────────
    # LLM provider credit tracking (task-board #290 free_type taxonomy)
    # ------------------------------------------------------------------
    # free_type semantics:
    #   one_time_credit       – fixed pool that depletes; never refills
    #   daily_renewable       – resets to initial_allowance every reset_date
    #   unlimited_rate_limited – credits_remaining = -1; only RPM/TPM gates
    #   freemium              – soft tier; rollover usage but provider decides

    def register_llm_provider_credits(
        self,
        provider: str,
        api_key: str,
        free_type: str = "unlimited_rate_limited",
        initial_allowance: float = -1.0,
        reset_period: str = "never",
        reset_date: Optional[str] = None,
    ):
        """Register an LLM provider API key for credit tracking. UNIQUE(provider, api_key_hash)."""
        import hashlib

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO llm_provider_credits
                (provider, api_key_hash, free_type, credits_remaining,
                 initial_allowance, reset_period, reset_date,
                 is_exhausted, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, CURRENT_TIMESTAMP)
                """,
                (
                    provider,
                    key_hash,
                    free_type,
                    initial_allowance,
                    initial_allowance,
                    reset_period,
                    reset_date,
                ),
            )
            conn.commit()
            conn.close()
            logger.info(
                f"Registered LLM provider {provider} ({free_type}, "
                f"allowance={initial_allowance})"
            )

        except Exception as e:
            logger.error(f"Failed to register LLM provider credits: {e}")

    def decrement_llm_credentials(
        self,
        provider: str,
        api_key: str,
        model: str = "",
        credits_consumed: float = 1.0,
        success: bool = True,
        error_message: Optional[str] = None,
    ):
        """Record an LLM call and decrement the credit balance.

        Skips decrement entirely for unlimited_rate_limited (credits_remaining
        will stay at -1) — caller should gate via check_available first.
        """
        import hashlib

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        try:
            with self._pooled_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO llm_provider_usage
                    (provider, api_key_hash, model, credits_consumed,
                     success, error_message)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        provider,
                        key_hash,
                        model,
                        credits_consumed,
                        success,
                        error_message,
                    ),
                )
                cursor.execute(
                    """
                    UPDATE llm_provider_credits
                    SET credits_used_total = credits_used_total + ?,
                        credits_remaining = CASE
                            WHEN credits_remaining < 0 THEN credits_remaining
                            ELSE MAX(0, credits_remaining - ?)
                        END,
                        is_exhausted = CASE
                            WHEN credits_remaining < 0 THEN 0
                            WHEN credits_remaining - ? <= 0 THEN 1
                            ELSE is_exhausted
                        END,
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

        except Exception as e:
            logger.error(f"Failed to decrement LLM credentials: {e}")

    def get_llm_provider_credit_status(self, provider: str, api_key: str) -> dict:
        """Get current credit status for an LLM provider API key."""
        import hashlib

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT free_type, credits_remaining, credits_used_total,
                       initial_allowance, reset_period, reset_date,
                       is_exhausted, last_updated
                FROM llm_provider_credits
                WHERE provider = ? AND api_key_hash = ?
                """,
                (provider, key_hash),
            )
            row = cursor.fetchone()
            conn.close()
            if row:
                return {
                    "provider": provider,
                    "free_type": row[0],
                    "credits_remaining": row[1],
                    "credits_used_total": row[2],
                    "initial_allowance": row[3],
                    "reset_period": row[4],
                    "reset_date": row[5],
                    "is_exhausted": bool(row[6]),
                    "last_updated": row[7],
                }
            # No row → assume unlimited (unregistered providers default to free)
            return {
                "provider": provider,
                "free_type": "unlimited_rate_limited",
                "credits_remaining": -1,
                "is_exhausted": False,
            }

        except Exception as e:
            logger.error(f"Failed to get LLM provider credit status: {e}")
            return {"provider": provider, "error": str(e), "is_exhausted": True}

    def check_llm_provider_available(
        self, provider: str, api_key: str, required_credits: float = 1.0
    ) -> bool:
        """Return True iff provider has enough credits to take another call.

        Unlimited-rate-limited (credits_remaining = -1) providers always pass
        the credit check; rate-limit cooldown is handled by the router.
        """
        status = self.get_llm_provider_credit_status(provider, api_key)
        if status.get("is_exhausted", True):
            return False
        remaining = status.get("credits_remaining", -1)
        if remaining < 0:
            return True  # unlimited
        return remaining >= required_credits

    def mark_llm_provider_exhausted(self, provider: str, api_key: str):
        """Permanently mark an API key as exhausted (e.g. one-time credit used)."""
        import hashlib

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE llm_provider_credits
                SET is_exhausted = 1, credits_remaining = 0,
                    last_updated = CURRENT_TIMESTAMP
                WHERE provider = ? AND api_key_hash = ?
                """,
                (provider, key_hash),
            )
            conn.commit()
            conn.close()
            logger.warning(
                f"Marked LLM provider {provider} as exhausted (one_time credit)"
            )

        except Exception as e:
            logger.error(f"Failed to mark LLM provider exhausted: {e}")

    def reset_daily_llm_provider_credits(
        self, provider: Optional[str] = None, api_key: Optional[str] = None
    ):
        """Reset daily_renewable providers (or a single key) when reset_date has passed.

        When called with no args, resets all due daily_renewable entries (suitable for cron).
        When called with (provider, api_key), resets just that one entry.
        """
        import hashlib

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            where_clauses = [
                "reset_period = 'daily'",
                "reset_date IS NOT NULL",
                "datetime(reset_date) <= datetime('now')",
                "initial_allowance >= 0",
            ]
            params: list = []
            if provider is not None:
                where_clauses.append("provider = ?")
                params.append(provider)
            if api_key is not None:
                key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
                where_clauses.append("api_key_hash = ?")
                params.append(key_hash)

            cursor.execute(
                f"""
                UPDATE llm_provider_credits
                SET credits_remaining = initial_allowance,
                    is_exhausted = 0,
                    credits_used_total = 0,
                    reset_date = datetime('now', '+1 day'),
                    last_updated = CURRENT_TIMESTAMP
                WHERE {" AND ".join(where_clauses)}
                """,
                tuple(params),
            )
            conn.commit()
            affected = cursor.rowcount
            conn.close()
            if affected:
                logger.info(f"Reset daily LLM provider credits for {affected} entries")
            return affected

        except Exception as e:
            logger.error(f"Failed to reset daily LLM provider credits: {e}")
            return 0

    def check_and_lazy_register(
        self,
        provider: str,
        api_key: str,
        free_type: str = "unlimited_rate_limited",
        initial_allowance: float = -1.0,
        reset_period: str = "never",
        reset_date: Optional[str] = None,
    ) -> bool:
        """Register provider credits if no row exists, then return availability.

        ponytail: collapses the lazy-init + availability-check pattern into one
        call so router/client sites don't repeat the register-then-check dance.
        Unlimited/freemium providers return True without consuming DB rows.
        """
        # ponytail: unlimited + freemium have no exhaustion semantics; skip DB entirely
        if free_type in ("unlimited_rate_limited", "freemium"):
            return True
        status = self.get_llm_provider_credit_status(provider, api_key)
        # A row exists iff free_type was persisted; default row reports unlimited.
        if status.get("free_type") == "unlimited_rate_limited" and free_type != "unlimited_rate_limited":
            # No prior row (default returned). Register now so exhaustion can track.
            self.register_llm_provider_credits(
                provider,
                api_key,
                free_type=free_type,
                initial_allowance=initial_allowance,
                reset_period=reset_period,
                reset_date=reset_date,
            )
        return self.check_llm_provider_available(provider, api_key)

    def record_llm_call_outcome(
        self,
        provider: str,
        api_key: str,
        success: bool,
        error_type: Optional[str] = None,
        model: str = "",
    ) -> None:
        """Glue method for router/client to record call outcome + exhaustion state.

        On success: decrement_llm_credentials (no-op for unlimited).
        On quota_exceeded: mark_llm_provider_exhausted. Permanence is derived
        from the persisted free_type: one_time_credit = permanent, daily_renewable
        = transient (daily cron resets), unlimited/freemium = no-op.

        ponytail: caller passes error_type only; permanence auto-derived here so
        call sites stay uniform. DB is source of truth for free_type.
        """
        try:
            if success:
                self.decrement_llm_credentials(
                    provider, api_key, model=model, success=True
                )
                return
            if error_type != "quota_exceeded":
                return  # ponytail: only quota_exceeded drives exhaustion; auth/rate-limit handled elsewhere
            status = self.get_llm_provider_credit_status(provider, api_key)
            free_type = status.get("free_type", "unlimited_rate_limited")
            if free_type in ("one_time_credit", "daily_renewable"):
                self.mark_llm_provider_exhausted(provider, api_key)
                # daily_renewable gets cleared by reset_daily_llm_provider_credits cron
        except Exception as e:
            logger.error(f"record_llm_call_outcome failed for {provider}: {e}")


# Global telemetry instance
_telemetry_manager: Optional[TelemetryManager] = None


def get_telemetry_manager() -> TelemetryManager:
    """Get or create the global telemetry manager instance."""
    global _telemetry_manager
    if _telemetry_manager is None:
        _telemetry_manager = TelemetryManager()
    return _telemetry_manager


if __name__ == "__main__":
    # ponytail: smallest self-test that fails if glue methods break.
    # Verifies: check_and_lazy_register no-op for unlimited, record_llm_call_outcome
    # marks one_time_credit exhausted on quota_exceeded, daily_renewable stays
    # available after success, unlimited unaffected by quota.
    import os
    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    tm = TelemetryManager(db_path=tmp.name)
    tm._init_db()

    # 1. unlimited via check_and_lazy_register -> True, no row created
    assert tm.check_and_lazy_register("groq", "k1", free_type="unlimited_rate_limited") is True
    # 2. one_time_credit: register -> available, quota_exceeded -> exhausted -> unavailable
    assert tm.check_and_lazy_register("together", "k2", free_type="one_time_credit", initial_allowance=5.0) is True
    tm.record_llm_call_outcome("together", "k2", success=False, error_type="quota_exceeded")
    assert tm.check_llm_provider_available("together", "k2") is False, "one_time_credit should be exhausted after quota"
    # 3. daily_renewable: success decrements, quota -> exhausted (until reset_date passes)
    assert tm.check_and_lazy_register("gemini", "k3", free_type="daily_renewable", initial_allowance=10.0, reset_period="daily", reset_date="2099-01-01") is True
    tm.record_llm_call_outcome("gemini", "k3", success=True, model="gemini-pro")
    assert tm.check_llm_provider_available("gemini", "k3") is True, "daily_renewable should have credits left"
    tm.record_llm_call_outcome("gemini", "k3", success=False, error_type="quota_exceeded")
    assert tm.check_llm_provider_available("gemini", "k3") is False, "daily_renewable should be exhausted after quota (reset_date future)"
    # 4. unlimited + quota -> no-op, still available (no row)
    tm.record_llm_call_outcome("groq", "k1", success=False, error_type="quota_exceeded")
    assert tm.check_llm_provider_available("groq", "k1") is True, "unlimited must remain available on quota"

    os.unlink(tmp.name)
    print("telemetry glue self-test: OK")
