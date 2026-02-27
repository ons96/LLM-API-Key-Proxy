#!/usr/bin/env python3
"""
Billing Tracker Module

Tracks API usage and costs for paid providers.
Calculates costs based on token usage and pricing configuration.
Persists records to SQLite for historical analysis and budget monitoring.
"""

import sqlite3
import logging
import yaml
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PRICING_CONFIG_PATH = Path(__file__).parent.parent / "config" / "pricing.yaml"
DEFAULT_DB_PATH = "billing.db"


@dataclass
class UsageRecord:
    """Represents a single usage record."""
    timestamp: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class BillingTracker:
    """Manages billing data and cost calculations."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        pricing_config_path: Path = DEFAULT_PRICING_CONFIG_PATH,
    ):
        """
        Initialize the billing tracker.

        Args:
            db_path: Path to the SQLite database file.
            pricing_config_path: Path to the YAML pricing configuration.
        """
        self.db_path = db_path
        self.pricing_config_path = pricing_config_path
        self.pricing_data: Dict[str, Any] = {}

        self._initialize_database()
        self._load_pricing_config()

        logger.info(f"BillingTracker initialized. DB: {self.db_path}")

    def _initialize_database(self):
        """Initialize SQLite database with required schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create usage records table
                cursor.execute(
                    """
                CREATE TABLE IF NOT EXISTS usage_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    estimated_cost_usd REAL NOT NULL
                )
                """
                )

                # Create indexes for common queries
                cursor.execute(
                    """
                CREATE INDEX IF NOT EXISTS idx_usage_timestamp 
                ON usage_records(timestamp)
                """
                )
                cursor.execute(
                    """
                CREATE INDEX IF NOT EXISTS idx_usage_provider 
                ON usage_records(provider)
                """
                )
                cursor.execute(
                    """
                CREATE INDEX IF NOT EXISTS idx_usage_model 
                ON usage_records(model)
                """
                )

                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize billing database: {e}")
            raise

    def _load_pricing_config(self):
        """Load pricing data from YAML configuration."""
        if not self.pricing_config_path.exists():
            logger.warning(
                f"Pricing config not found at {self.pricing_config_path}. "
                "Cost tracking will be disabled/zero."
            )
            self.pricing_data = {}
            return

        try:
            with open(self.pricing_config_path, "r") as f:
                self.pricing_data = yaml.safe_load(f) or {}
            logger.info(
                f"Loaded pricing configuration for {len(self.pricing_data)} providers."
            )
        except Exception as e:
            logger.error(f"Failed to load pricing config: {e}")
            self.pricing_data = {}

    def calculate_cost(
        self, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """
        Calculate estimated cost based on token usage.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic').
            model: Model name (e.g., 'gpt-4o', 'claude-3-opus-20240229').
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD. Returns 0.0 if pricing not found.
        """
        provider_key = provider.lower()
        model_key = model.lower()

        # Handle cases where model name might include provider prefix (e.g., "openai/gpt-4o")
        if "/" in model_key:
            parts = model_key.split("/", 1)
            if len(parts) == 2:
                potential_provider, potential_model = parts
                # If the passed provider is generic or matches the prefix, use the prefix
                if provider_key == "unknown" or provider_key == potential_provider:
                    provider_key = potential_provider
                    model_key = potential_model

        provider_pricing = self.pricing_data.get(provider_key)
        if not provider_pricing:
            logger.debug(f"No pricing data for provider: {provider_key}")
            return 0.0

        model_pricing = provider_pricing.get(model_key)
        if not model_pricing:
            # Try to find a partial match or default if necessary, 
            # but for now strict match is safer to avoid incorrect billing.
            logger.debug(f"No pricing data for model: {provider_key}/{model_key}")
            return 0.0

        input_cost_per_1m = model_pricing.get("input_cost_per_1m", 0.0)
        output_cost_per_1m = model_pricing.get("output_cost_per_1m", 0.0)

        # Calculate cost: (tokens / 1,000,000) * cost_per_1m
        input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * output_cost_per_1m

        total_cost = input_cost + output_cost
        return round(total_cost, 6)

    def log_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        timestamp: Optional[datetime] = None,
    ):
        """
        Log a usage record to the database.

        Args:
            provider: Provider name.
            model: Model name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            timestamp: Timestamp of the request (defaults to now).
        """
        if timestamp is None:
            timestamp = datetime.now()

        total_tokens = input_tokens + output_tokens
        cost = self.calculate_cost(provider, model, input_tokens, output_tokens)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO usage_records 
                    (timestamp, provider, model, input_tokens, output_tokens, total_tokens, estimated_cost_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp.isoformat(),
                        provider,
                        model,
                        input_tokens,
                        output_tokens,
                        total_tokens,
                        cost,
                    ),
                )
                conn.commit()
                logger.debug(
                    f"Logged usage: {provider}/{model} - {total_tokens} tokens, ${cost:.6f}"
                )
        except Exception as e:
            logger.error(f"Failed to log usage: {e}")

    def get_total_spend(
        self,
        provider: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> float:
        """
        Get total estimated spend for a given period and provider.

        Args:
            provider: Filter by provider (optional).
            start_date: Start of period (optional).
            end_date: End of period (optional).

        Returns:
            Total cost in USD.
        """
        query = "SELECT SUM(estimated_cost_usd) FROM usage_records WHERE 1=1"
        params = []

        if provider:
            query += " AND provider = ?"
            params.append(provider)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                result = cursor.fetchone()
                return result[0] if result[0] else 0.0
        except Exception as e:
            logger.error(f"Failed to query total spend: {e}")
            return 0.0

    def get_usage_stats(
        self, days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get usage statistics grouped by provider and model for the last N days.

        Args:
            days: Number of days to look back.

        Returns:
            List of dictionaries containing stats.
        """
        since_date = datetime.now() - timedelta(days=days)

        query = """
            SELECT 
                provider,
                model,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(estimated_cost_usd) as total_cost
            FROM usage_records
            WHERE timestamp >= ?
            GROUP BY provider, model
            ORDER BY total_cost DESC
        """

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, (since_date.isoformat(),))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get usage stats: {e}")
            return []
