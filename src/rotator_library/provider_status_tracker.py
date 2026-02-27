#!/usr/bin/env python3
"""
Provider Status Tracker Module

Real-time monitoring of LLM API provider health, uptime, latency, rate limits, and response times.
Feeds provider health data to the proxy's routing logic for intelligent provider selection.
Includes alerting capabilities for outage notifications.
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

from .alert_manager import AlertManager

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
        enable_alerts: bool = True,
    ):
        """
        Initialize the provider status tracker.

        Args:
            check_interval_minutes: How often to run health checks (default: 5)
            max_consecutive_failures: Number of failures before marking provider as down (default: 3)
            degraded_latency_threshold_ms: Response time threshold for degraded status (default: 1000ms)
            enable_alerts: Whether to enable alerting for outages (default: True)
        """
        self.check_interval_minutes = check_interval_minutes
        self.max_consecutive_failures = max_consecutive_failures
        self.degraded_latency_threshold_ms = degraded_latency_threshold_ms
        self.enable_alerts = enable_alerts

        # Database setup
        self.db_path = "provider_status.db"
        self._initialize_database()

        # Provider configuration
        self.providers_to_monitor = self._discover_providers()

        # Health check state
        self.running = False
        self.check_task = None
        
        # Alert manager
        self.alert_manager = AlertManager() if enable_alerts else None
        
        # Track previous statuses for change detection
        self.previous_statuses: Dict[str, str] = {}

        logger.info(
            f"ProviderStatusTracker initialized with {len(self.providers_to_monitor)} providers"
        )
        logger.info(f"Providers to monitor: {', '.join(self.providers_to_monitor)}")
        if enable_alerts:
            logger.info("Alerting enabled for provider outages")

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

    def _get_previous_status(self, provider_name: str) -> Optional[str]:
        """Get the most recent previous status for a provider."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT status FROM provider_health_checks 
                    WHERE provider_name = ? 
                    ORDER BY last_check_timestamp DESC 
                    LIMIT 1
                    """,
                    (provider_name,)
                )
                row = cursor.fetchone()
                return row["status"] if row else None
        except Exception as e:
            logger.error(f"Failed to get previous status for {provider_name}: {e}")
            return None

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
        """Store health check result in database and trigger alerts if needed."""
        try:
            # Get previous status before storing new one
            previous_status = self._get_previous_status(provider_name)
            
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
                WHERE last_check_timestamp < ?
                """,
                    (seven_days_ago,),
                )

                conn.commit()
                
            # Check for status changes and send alerts
            if self.enable_alerts and self.alert_manager:
                asyncio.create_task(
                    self._check_and_alert(
                        provider_name=provider_name,
                        current_status=status,
                        previous_status=previous_status,
                        response_time_ms=response_time_ms,
                        error_message=error_message,
                        consecutive_failures=consecutive_failures,
                        uptime_percent=uptime_percent
                    )
                )

        except Exception as e:
            logger.error(f"Failed to store health check result: {e}")
            
    async def _check_and_alert(
        self,
        provider_name: str,
        current_status: str,
        previous_status: Optional[str],
        response_time_ms: float,
        error_message: str,
        consecutive_failures: int,
        uptime_percent: float
    ):
        """Check if alert should be sent and send it."""
        if not previous_status:
            # First check, no previous state to compare
            return
            
        # Alert on state changes or critical conditions
        should_alert = False
        
        # Provider went down
        if current_status == "down" and previous_status != "down":
            should_alert = True
            logger.warning(f"Provider {provider_name} transitioned to DOWN state")
            
        # Provider recovered
        elif current_status == "healthy" and previous_status in ("down", "degraded"):
            should_alert = True
            logger.info(f"Provider {provider_name} recovered to HEALTHY state")
            
        # Provider became degraded from healthy
        elif current_status == "degraded" and previous_status == "healthy":
            should_alert = True
            logger.warning(f"Provider {provider_name} transitioned to DEGRADED state")
            
        if should_alert:
            await self.alert_manager.send_alert(
                provider=provider_name,
                status=current_status,
                details={
                    "previous_status": previous_status,
                    "response_time_ms": response_time_ms,
                    "error_message": error_message,
                    "consecutive_failures": consecutive_failures,
                    "uptime_percent": uptime_percent
                }
            )

    async def _check_provider_health(self, provider: str) -> ProviderHealthStatus:
        """
        Perform health check on a specific provider.
        This is a placeholder implementation - subclasses should override
        or the actual health check logic should be implemented here.
        """
        start_time = time.time()
        
        try:
            # Placeholder health check - in real implementation, this would
            # make actual API calls to check provider health
            # For now, simulate based on provider type
            
            if provider.startswith("g4f"):
                # Simulate G4F check
                await asyncio.sleep(0.1)
                response_time = (time.time() - start_time) * 1000
                
                return ProviderHealthStatus(
                    status="healthy",
                    response_time_ms=response_time,
                    last_check_timestamp=datetime.now().isoformat(),
                    consecutive_failures=0
                )
            else:
                # Generic provider check would go here
                await asyncio.sleep(0.05)
                response_time = (time.time() - start_time) * 1000
                
                return ProviderHealthStatus(
                    status="healthy",
                    response_time_ms=response_time,
                    last_check_timestamp=datetime.now().isoformat(),
                    consecutive_failures=0
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ProviderHealthStatus(
                status="down",
                response_time_ms=response_time,
                last_check_timestamp=datetime.now().isoformat(),
                error_message=str(e),
                consecutive_failures=1  # Would be incremented based on history
            )

    async def _run_health_checks(self):
        """Main loop for running health checks."""
        while self.running:
            try:
                logger.debug("Starting health check cycle")
                
                for provider in self.providers_to_monitor:
                    try:
                        health_status = await self._check_provider_health(provider)
                        
                        self._store_health_check_result(
                            provider_name=provider,
                            status=health_status.status,
                            response_time_ms=health_status.response_time_ms,
                            error_message=health_status.error_message,
                            consecutive_failures=health_status.consecutive_failures
                        )
                        
                    except Exception as e:
                        logger.error(f"Health check failed for {provider}: {e}")
                        # Store failure
                        self._store_health_check_result(
                            provider_name=provider,
                            status="down",
                            error_message=str(e),
                            consecutive_failures=1
                        )
                        
                # Wait for next check interval
                await asyncio.sleep(self.check_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    def start_monitoring(self):
        """Start the health check monitoring loop."""
        if not self.running:
            self.running = True
            self.check_task = asyncio.create_task(self._run_health_checks())
            logger.info("Provider health monitoring started")

    def stop_monitoring(self):
        """Stop the health check monitoring loop."""
        self.running = False
        if self.check_task:
            self.check_task.cancel()
            logger.info("Provider health monitoring stopped")

    def get_provider_status(self, provider_name: str) -> Optional[ProviderHealthStatus]:
        """Get current health status for a specific provider."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM provider_health_checks 
                    WHERE provider_name = ? 
                    ORDER BY last_check_timestamp DESC 
                    LIMIT 1
                    """,
                    (provider_name,)
                )
                row = cursor.fetchone()
                
                if row:
                    return ProviderHealthStatus(
                        status=row["status"],
                        response_time_ms=row["response_time_ms"],
                        uptime_percent=row["uptime_percent"],
                        rate_limit_percent=row["rate_limit_percent"],
                        last_check_timestamp=row["last_check_timestamp"],
                        error_message=row["error_message"] or "",
                        consecutive_failures=row["consecutive_failures"] or 0
                    )
                return None
                
        except Exception as e:
            logger.error(f"Failed to get status for {provider_name}: {e}")
            return None

    def get_all_provider_statuses(self) -> Dict[str, ProviderHealthStatus]:
        """Get current health status for all providers."""
        statuses = {}
        
        for provider in self.providers_to_monitor:
            status = self.get_provider_status(provider)
            if status:
                statuses[provider] = status
                
        return statuses

    def get_alert_config(self) -> Dict[str, Any]:
        """Get current alert configuration summary."""
        if self.alert_manager:
            return self.alert_manager.get_alert_summary()
        return {"enabled": False}
