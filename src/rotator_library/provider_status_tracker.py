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
                WHERE created_at < ?
                """,
                    (seven_days_ago,),
                )

                conn.commit()

        except Exception as e:
            logger.error(
                f"Failed to store health check result for {provider_name}: {e}"
            )

    def _calculate_uptime_percent(
        self, provider_name: str, window_hours: int = 24
    ) -> float:
        """Calculate uptime percentage for a provider over the given time window."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                # Get successful checks
                window_start = (
                    datetime.now() - timedelta(hours=window_hours)
                ).isoformat()
                cursor.execute(
                    """
                SELECT COUNT(*) as success_count 
                FROM provider_health_checks 
                WHERE provider_name = ? 
                AND status = 'healthy' 
                AND last_check_timestamp >= ?
                """,
                    (provider_name, window_start),
                )

                success_count = cursor.fetchone()[0] or 0

                # Get total checks
                cursor.execute(
                    """
                SELECT COUNT(*) as total_count 
                FROM provider_health_checks 
                WHERE provider_name = ? 
                AND last_check_timestamp >= ?
                """,
                    (provider_name, window_start),
                )

                total_count = cursor.fetchone()[0] or 1  # Avoid division by zero

                return (success_count / total_count) * 100.0

        except Exception as e:
            logger.error(f"Failed to calculate uptime for {provider_name}: {e}")
            return 100.0  # Assume healthy if we can't calculate

    def _get_latest_status(self, provider_name: str) -> Optional[ProviderHealthStatus]:
        """Get the latest health status for a provider."""
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
                    (provider_name,),
                )

                row = cursor.fetchone()
                if row:
                    return ProviderHealthStatus(
                        status=row["status"],
                        response_time_ms=row["response_time_ms"] or 0.0,
                        uptime_percent=row["uptime_percent"] or 100.0,
                        rate_limit_percent=row["rate_limit_percent"] or 0.0,
                        last_check_timestamp=row["last_check_timestamp"],
                        error_message=row["error_message"] or "",
                        consecutive_failures=row["consecutive_failures"] or 0,
                    )

            return None

        except Exception as e:
            logger.error(f"Failed to get latest status for {provider_name}: {e}")
            return None

    async def _health_check_provider(
        self, provider_name: str, session: aiohttp.ClientSession
    ) -> Tuple[str, float, str]:
        """Perform health check for a single provider."""
        start_time = time.time()
        status = "healthy"
        response_time_ms = 0.0
        error_message = ""

        try:
            if provider_name == "g4f" or provider_name.startswith("g4f_"):
                # Handle G4F (local library or specific endpoint)
                endpoint = (
                    provider_name.replace("g4f_", "")
                    if "_" in provider_name
                    else "main"
                )
                return await self._health_check_g4f_endpoint(endpoint, session)
            elif provider_name == "groq":
                return await self._health_check_groq(session)
            elif provider_name == "openrouter":
                return await self._health_check_openrouter(session)
            elif provider_name == "together":
                return await self._health_check_together(session)
            elif provider_name == "gemini_cli":
                return await self._health_check_gemini_cli(session)
            elif provider_name == "nvidia":
                return await self._health_check_nvidia(session)
            elif provider_name == "mistral":
                return await self._health_check_mistral(session)
            elif provider_name == "huggingface":
                return await self._health_check_huggingface(session)
            elif provider_name == "gemini":
                return await self._health_check_gemini(session)
            elif provider_name == "google":
                return await self._health_check_google(session)
            else:
                # Generic health check for unknown providers
                return await self._health_check_generic(provider_name, session)

        except asyncio.TimeoutError:
            status = "down"
            error_message = "Request timed out"
            logger.warning(f"Health check timeout for {provider_name}")
        except Exception as e:
            status = "down"
            error_message = str(e)
            logger.error(f"Health check failed for {provider_name}: {e}")

        response_time_ms = (time.time() - start_time) * 1000
        return status, response_time_ms, error_message

    async def _health_check_generic(
        self, provider_name: str, session: aiohttp.ClientSession
    ) -> Tuple[str, float, str]:
        """Generic health check using /models endpoint."""
        try:
            # Try to get API base URL from environment
            api_base_env = f"{provider_name.upper()}_API_BASE"
            api_base = os.getenv(api_base_env, f"https://api.{provider_name}.com/v1")

            # Try to get API key
            api_key_env = f"{provider_name.upper()}_API_KEY"
            api_key = os.getenv(api_key_env, "")

            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            start_time = time.time()
            async with session.get(
                f"{api_base}/models",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response_time_ms = (time.time() - start_time) * 1000

                if response.status == 200:
                    return "healthy", response_time_ms, ""
                else:
                    error_msg = f"HTTP {response.status}: {await response.text()}"
                    return "degraded", response_time_ms, error_msg

        except Exception as e:
            return "down", 0.0, str(e)

    async def _health_check_groq(
        self, session: aiohttp.ClientSession
    ) -> Tuple[str, float, str]:
        """Health check for Groq."""
        try:
            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                return "down", 0.0, "No API key configured"

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            start_time = time.time()
            async with session.get(
                "https://api.groq.com/openai/v1/models",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response_time_ms = (time.time() - start_time) * 1000

                if response.status == 200:
                    return "healthy", response_time_ms, ""
                else:
                    error_msg = f"HTTP {response.status}: {await response.text()}"
                    return "degraded", response_time_ms, error_msg

        except Exception as e:
            return "down", 0.0, str(e)

    async def _health_check_openrouter(
        self, session: aiohttp.ClientSession
    ) -> Tuple[str, float, str]:
        """Health check for OpenRouter."""
        try:
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            if not api_key:
                return "down", 0.0, "No API key configured"

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            start_time = time.time()
            async with session.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response_time_ms = (time.time() - start_time) * 1000

                if response.status == 200:
                    return "healthy", response_time_ms, ""
                else:
                    error_msg = f"HTTP {response.status}: {await response.text()}"
                    return "degraded", response_time_ms, error_msg

        except Exception as e:
            return "down", 0.0, str(e)

    async def _health_check_together(
        self, session: aiohttp.ClientSession
    ) -> Tuple[str, float, str]:
        """Health check for Together AI."""
        try:
            api_key = os.getenv("TOGETHER_API_KEY", "")
            if not api_key:
                return "down", 0.0, "No API key configured"

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            start_time = time.time()
            async with session.get(
                "https://api.together.xyz/v1/models",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response_time_ms = (time.time() - start_time) * 1000

                if response.status == 200:
                    return "healthy", response_time_ms, ""
                else:
                    error_msg = f"HTTP {response.status}: {await response.text()}"
                    return "degraded", response_time_ms, error_msg

        except Exception as e:
            return "down", 0.0, str(e)

    async def _health_check_g4f_endpoint(
        self, endpoint: str, session: aiohttp.ClientSession
    ) -> Tuple[str, float, str]:
        """Health check for G4F (using local library)."""
        try:
            import g4f

            start_time = time.time()

            # Try a few models in order of likelihood to work
            test_models = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
            response = None
            last_error = ""

            for model in test_models:
                try:
                    # Note: We run this in a thread executor because g4f might be sync or blocking
                    # Use typing.cast to avoid "AsyncResult" is not awaitable error
                    from typing import cast

                    coro = g4f.ChatCompletion.create_async(
                        model=model,
                        messages=[{"role": "user", "content": "ping"}],
                    )
                    response = await cast(Any, coro)
                    if response:
                        break
                except Exception as e:
                    last_error = str(e)
                    continue

            response_time_ms = (time.time() - start_time) * 1000

            if response:
                return "healthy", response_time_ms, ""
            else:
                return (
                    "degraded",
                    response_time_ms,
                    f"All models failed. Last error: {last_error}",
                )

        except Exception as e:
            # If g4f library fails, it's down
            return "down", 0.0, str(e)

    async def _health_check_gemini_cli(
        self, session: aiohttp.ClientSession
    ) -> Tuple[str, float, str]:
        """Health check for Gemini CLI."""
        try:
            project_id = os.getenv("GEMINI_CLI_PROJECT_ID", "")
            if not project_id:
                return "down", 0.0, "No project ID configured"

            # Test credential validity
            start_time = time.time()
            async with session.get(
                "https://generativelanguage.googleapis.com/v1/models",
                params={"key": project_id},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response_time_ms = (time.time() - start_time) * 1000

                if response.status == 200:
                    return "healthy", response_time_ms, ""
                else:
                    error_msg = f"HTTP {response.status}: {await response.text()}"
                    return "degraded", response_time_ms, error_msg

        except Exception as e:
            return "down", 0.0, str(e)

    async def _health_check_nvidia(
        self, session: aiohttp.ClientSession
    ) -> Tuple[str, float, str]:
        """Health check for Nvidia."""
        try:
            api_key = os.getenv("NVIDIA_API_KEY", "")
            if not api_key:
                return "down", 0.0, "No API key configured"

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            start_time = time.time()
            async with session.get(
                "https://integrate.api.nvidia.com/v1/models",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response_time_ms = (time.time() - start_time) * 1000

                if response.status == 200:
                    return "healthy", response_time_ms, ""
                else:
                    error_msg = f"HTTP {response.status}: {await response.text()}"
                    return "degraded", response_time_ms, error_msg

        except Exception as e:
            return "down", 0.0, str(e)

    async def _health_check_mistral(
        self, session: aiohttp.ClientSession
    ) -> Tuple[str, float, str]:
        """Health check for Mistral."""
        try:
            api_key = os.getenv("MISTRAL_API_KEY", "")
            if not api_key:
                return "down", 0.0, "No API key configured"

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            start_time = time.time()
            async with session.get(
                "https://api.mistral.ai/v1/models",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response_time_ms = (time.time() - start_time) * 1000

                if response.status == 200:
                    return "healthy", response_time_ms, ""
                else:
                    error_msg = f"HTTP {response.status}: {await response.text()}"
                    return "degraded", response_time_ms, error_msg

        except Exception as e:
            return "down", 0.0, str(e)

    async def _health_check_huggingface(
        self, session: aiohttp.ClientSession
    ) -> Tuple[str, float, str]:
        """Health check for HuggingFace."""
        try:
            api_key = os.getenv("HUGGINGFACE_API_KEY", "")
            if not api_key:
                return "down", 0.0, "No API key configured"

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            start_time = time.time()
            async with session.get(
                "https://api-inference.huggingface.co/models",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response_time_ms = (time.time() - start_time) * 1000

                if response.status == 200:
                    return "healthy", response_time_ms, ""
                else:
                    error_msg = f"HTTP {response.status}: {await response.text()}"
                    return "degraded", response_time_ms, error_msg

        except Exception as e:
            return "down", 0.0, str(e)

    async def _health_check_gemini(
        self, session: aiohttp.ClientSession
    ) -> Tuple[str, float, str]:
        """Health check for Gemini."""
        try:
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                return "down", 0.0, "No API key configured"

            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            }

            start_time = time.time()
            async with session.get(
                "https://generativelanguage.googleapis.com/v1/models",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response_time_ms = (time.time() - start_time) * 1000

                if response.status == 200:
                    return "healthy", response_time_ms, ""
                else:
                    error_msg = f"HTTP {response.status}: {await response.text()}"
                    return "degraded", response_time_ms, error_msg

        except Exception as e:
            return "down", 0.0, str(e)

    async def _health_check_google(
        self, session: aiohttp.ClientSession
    ) -> Tuple[str, float, str]:
        """Health check for Google."""
        try:
            api_key = os.getenv("GOOGLE_API_KEY", "")
            if not api_key:
                return "down", 0.0, "No API key configured"

            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            }

            start_time = time.time()
            async with session.get(
                "https://generativelanguage.googleapis.com/v1/models",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response_time_ms = (time.time() - start_time) * 1000

                if response.status == 200:
                    return "healthy", response_time_ms, ""
                else:
                    error_msg = f"HTTP {response.status}: {await response.text()}"
                    return "degraded", response_time_ms, error_msg

        except Exception as e:
            return "down", 0.0, str(e)

    async def _run_health_checks(self):
        """Run health checks for all providers."""
        if not self.providers_to_monitor:
            logger.warning("No providers configured for monitoring")
            return

        logger.info(
            f"Starting health checks for {len(self.providers_to_monitor)} providers"
        )

        # Create aiohttp session for concurrent requests
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Run health checks concurrently
            tasks = []
            for provider in self.providers_to_monitor:
                task = asyncio.create_task(
                    self._health_check_provider(provider, session)
                )
                tasks.append((provider, task))

            # Process results as they complete
            results = {}
            for provider, task in tasks:
                try:
                    status, response_time, error_msg = await task
                    results[provider] = {
                        "status": status,
                        "response_time_ms": response_time,
                        "error_message": error_msg,
                    }

                    # Log individual results
                    if status == "healthy":
                        logger.info(f"✓ {provider}: {response_time:.1f}ms")
                    else:
                        logger.warning(f"✗ {provider}: {status} - {error_msg}")

                except Exception as e:
                    logger.error(f"Health check failed for {provider}: {e}")
                    results[provider] = {
                        "status": "down",
                        "response_time_ms": 0.0,
                        "error_message": str(e),
                    }

            # Store results in database
            for provider, result in results.items():
                # Get previous status to calculate consecutive failures
                previous_status = self._get_latest_status(provider)
                consecutive_failures = 0

                if previous_status and previous_status.status != "healthy":
                    consecutive_failures = previous_status.consecutive_failures + 1
                elif result["status"] != "healthy":
                    consecutive_failures = 1

                # Calculate uptime percentage
                uptime_percent = self._calculate_uptime_percent(provider)

                # Determine final status based on response time and failures
                final_status = result["status"]

                # Apply degraded status if response time is too high
                if (
                    result["status"] == "healthy"
                    and result["response_time_ms"] > self.degraded_latency_threshold_ms
                ):
                    final_status = "degraded"

                # Apply down status if too many consecutive failures
                if consecutive_failures >= self.max_consecutive_failures:
                    final_status = "down"

                # Store the result
                self._store_health_check_result(
                    provider_name=provider,
                    status=final_status,
                    response_time_ms=result["response_time_ms"],
                    uptime_percent=uptime_percent,
                    rate_limit_percent=0.0,  # TODO: Implement rate limit tracking
                    error_message=result["error_message"],
                    consecutive_failures=consecutive_failures,
                )

            logger.info("Health checks completed")

    def get_current_status(self) -> Dict[str, Any]:
        """Get current status snapshot for all providers."""
        status_snapshot = {"timestamp": datetime.now().isoformat(), "providers": {}}

        for provider in self.providers_to_monitor:
            latest_status = self._get_latest_status(provider)
            if latest_status:
                status_snapshot["providers"][provider] = {
                    "status": latest_status.status,
                    "response_time_ms": latest_status.response_time_ms,
                    "uptime_percent": latest_status.uptime_percent,
                    "rate_limit_percent": latest_status.rate_limit_percent,
                    "last_check": latest_status.last_check_timestamp,
                    "error_message": latest_status.error_message,
                    "consecutive_failures": latest_status.consecutive_failures,
                }
            else:
                # No health data available
                status_snapshot["providers"][provider] = {
                    "status": "unknown",
                    "response_time_ms": 0.0,
                    "uptime_percent": 0.0,
                    "rate_limit_percent": 0.0,
                    "last_check": "never",
                    "error_message": "No health data available",
                    "consecutive_failures": 0,
                }

        return status_snapshot

    def get_best_provider(self) -> Dict[str, Any]:
        """Get the healthiest/fastest provider."""
        current_status = self.get_current_status()
        providers_data = current_status["providers"]

        # Filter healthy providers
        healthy_providers = []
        for provider, data in providers_data.items():
            if data["status"] == "healthy":
                healthy_providers.append((provider, data))

        if not healthy_providers:
            # No healthy providers, find the least bad option
            degraded_providers = []
            for provider, data in providers_data.items():
                if data["status"] == "degraded":
                    degraded_providers.append((provider, data))

            if degraded_providers:
                # Sort by response time
                degraded_providers.sort(key=lambda x: x[1]["response_time_ms"])
                best_provider = degraded_providers[0][0]
                return {
                    "best_provider": best_provider,
                    "reason": f"No healthy providers available, using least degraded: {best_provider}",
                    "alternatives": [],
                }
            else:
                return {
                    "best_provider": None,
                    "reason": "All providers are down",
                    "alternatives": [],
                }

        # Sort healthy providers by response time (lowest first)
        healthy_providers.sort(key=lambda x: x[1]["response_time_ms"])

        best_provider = healthy_providers[0][0]
        alternatives = [
            p[0] for p in healthy_providers[1 : min(3, len(healthy_providers))]
        ]

        return {
            "best_provider": best_provider,
            "reason": f"lowest latency ({healthy_providers[0][1]['response_time_ms']:.1f}ms) + healthy status",
            "alternatives": alternatives,
        }

    def get_provider_history(
        self, provider_name: str, hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get historical data for a specific provider."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                # Calculate time window
                window_start = (datetime.now() - timedelta(hours=hours)).isoformat()

                cursor.execute(
                    """
                SELECT provider_name, status, response_time_ms, last_check_timestamp 
                FROM provider_health_checks 
                WHERE provider_name = ? 
                AND last_check_timestamp >= ?
                ORDER BY last_check_timestamp ASC
                """,
                    (provider_name, window_start),
                )

                history = []
                for row in cursor.fetchall():
                    history.append(
                        {
                            "provider_name": row["provider_name"],
                            "status": row["status"],
                            "response_time_ms": row["response_time_ms"],
                            "timestamp": row["last_check_timestamp"],
                        }
                    )

                return history

        except Exception as e:
            logger.error(f"Failed to get history for {provider_name}: {e}")
            return []

    def get_all_history(self, hours: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        """Get historical data for all providers."""
        history = {}

        for provider in self.providers_to_monitor:
            provider_history = self.get_provider_history(provider, hours)
            if provider_history:
                history[provider] = provider_history

        return history

    def export_to_csv(self) -> str:
        """Export current status to CSV format."""
        current_status = self.get_current_status()

        # CSV header
        csv_lines = [
            "provider_name,status,response_time_ms,uptime_percent,rate_limit_percent,last_check_timestamp,consecutive_failures"
        ]

        # Add data for each provider
        for provider, data in current_status["providers"].items():
            csv_line = f"{provider},{data['status']},{data['response_time_ms']},{data['uptime_percent']},{data['rate_limit_percent']},{data['last_check']},{data['consecutive_failures']}"
            csv_lines.append(csv_line)

        return "\n".join(csv_lines)

    def start(self):
        """Start the background health check scheduler."""
        if self.running:
            logger.warning("Provider status tracker is already running")
            return

        self.running = True
        logger.info(
            f"Starting provider status tracker (interval: {self.check_interval_minutes} minutes)"
        )

        # Run initial check immediately
        asyncio.create_task(self._run_health_checks())

        # Schedule periodic checks
        self.check_task = asyncio.create_task(self._schedule_periodic_checks())

    async def _schedule_periodic_checks(self):
        """Schedule periodic health checks."""
        while self.running:
            try:
                # Wait for the interval
                await asyncio.sleep(self.check_interval_minutes * 60)

                # Run health checks
                await self._run_health_checks()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic health check scheduling: {e}")

    async def stop(self):
        """Stop the background health check scheduler."""
        if not self.running:
            return

        self.running = False

        if self.check_task:
            self.check_task.cancel()
            try:
                await self.check_task
            except asyncio.CancelledError:
                pass

        logger.info("Provider status tracker stopped")

    def run_health_checks(self):
        """Run health checks immediately (synchronous wrapper)."""
        return asyncio.run(self._run_health_checks())
