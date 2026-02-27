#!/usr/bin/env python3
"""
Provider Status Tracker Module

Real-time monitoring of LLM API provider health, uptime, latency, rate limits, and response times.
Feeds provider health data to the proxy's routing logic for intelligent provider selection.
"""

import asyncio
import logging
import os
import sqlite3
import time
import aiohttp
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ProviderHealthStatus:
    """Current health status for a single provider."""

    status: str  # healthy/degraded/down
    response_time_ms: float = 0.0
    uptime_percent: float = 100.0
    rate_limit_percent: float = 0.0
    last_check_timestamp: str = ""
    error_message: str = ""
    consecutive_failures: int = 0


class ProviderStatusTracker:
    """Main class for tracking provider health and status."""

    def __init__(
        self,
        check_interval_minutes: int = 5,
        max_consecutive_failures: int = 3,
        degraded_latency_threshold_ms: int = 1000,
    ):
        """
        Initialize the provider status tracker.

        Args:
            check_interval_minutes: How often to run health checks (default: 5)
            max_consecutive_failures: Number of failures before marking provider as down (default: 3)
            degraded_latency_threshold_ms: Response time threshold for degraded status (default: 1000ms)
        """
        self.check_interval_minutes = check_interval_minutes
        self.max_consecutive_failures = max_consecutive_failures
        self.degraded_latency_threshold_ms = degraded_latency_threshold_ms

        # Database setup
        self.db_path = "provider_status.db"
        self._initialize_database()

        # Provider configuration
        self.providers_to_monitor = self._discover_providers()

        # Health check state
        self.running = False
        self.check_task = None

        logger.info(
            f"ProviderStatusTracker initialized with {len(self.providers_to_monitor)} providers"
        )
        logger.info(f"Providers to monitor: {', '.join(self.providers_to_monitor)}")

    def _discover_providers(self) -> List[str]:
        """Discover all configured providers from environment variables."""
        providers = []

        # List of known free/non-paid providers
        free_providers = [
            "groq",
            "openrouter",
            "together",
            "g4f",
            "gemini_cli",
            "nvidia",
            "mistral",
            "huggingface",
            "gemini",
            "google",
            "replicate",
        ]

        # Check which providers have API keys configured
        for provider in free_providers:
            # Check for API key environment variables
            if provider == "g4f":
                # G4F doesn't require API keys
                providers.append(provider)
            elif provider == "gemini_cli":
                # Check for Gemini CLI configuration
                if os.getenv("GEMINI_CLI_PROJECT_ID"):
                    providers.append(provider)
            else:
                # Check for standard API key patterns
                api_key_env = f"{provider.upper()}_API_KEY"
                if os.getenv(api_key_env):
                    providers.append(provider)

        # Add G4F endpoints as separate providers
        g4f_endpoints = ["g4f_main", "g4f_groq", "g4f_grok", "g4f_gemini", "g4f_nvidia"]
        for endpoint in g4f_endpoints:
            providers.append(endpoint)

        return list(set(providers))  # Remove duplicates

    def _initialize_database(self):
        """Initialize SQLite database with required schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create main health checks table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS provider_health_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_time_ms REAL,
                    uptime_percent REAL,
                    rate_limit_percent REAL,
                    last_check_timestamp DATETIME NOT NULL,
                    error_message TEXT,
                    consecutive_failures INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """)

                # Create index for faster queries
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_provider_timestamp 
                ON provider_health_checks(provider_name, last_check_timestamp)
                """)

                conn.commit()

            logger.info(f"Database initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def _get_db_connection(self):
        """Get database connection with proper error handling."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def _store_health_check_result(
        self,
        provider_name: str,
        status: str,
        response_time_ms: float = 0.0,
        uptime_percent: float = 100.0,
        rate_limit_percent: float = 0.0,
        error_message: str = "",
        consecutive_failures: int = 0,
    ):
        """Store health check result in database."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                # Insert new record
                cursor.execute(
                    """
                INSERT INTO provider_health_checks 
                (provider_name, status, response_time_ms, uptime_percent, 
                 rate_limit_percent, last_check_timestamp, error_message, consecutive_failures)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        provider_name,
                        status,
                        response_time_ms,
                        uptime_percent,
                        rate_limit_percent,
                        datetime.now().isoformat(),
                        error_message,
                        consecutive_failures,
                    ),
                )

                # Clean up old records (>7 days)
                seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
                cursor.execute(
                    """
                DELETE FROM provider_health_checks 
