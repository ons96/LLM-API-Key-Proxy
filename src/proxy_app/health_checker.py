import asyncio
import logging
import time
from typing import Dict, Any, List
import litellm

logger = logging.getLogger(__name__)


class HealthChecker:
    """
    Background service to periodically check provider health.
    """

    def __init__(self, router_integration: Any, interval_seconds: int = 300) -> None:
        self.router_integration = router_integration
        self.interval_seconds = interval_seconds
        self.is_running = False
        self.provider_status: Dict[str, Dict[str, Any]] = {}

    async def start(self):
        """Start the background health check loop."""
        self.is_running = True
        asyncio.create_task(self._health_check_loop())
        logger.info(f"Health checker started (interval: {self.interval_seconds}s)")

    async def stop(self):
        """Stop the background loop."""
        self.is_running = False
        logger.info("Health checker stopped")

    async def _health_check_loop(self) -> None:
        """Main loop for health checks."""
        while self.is_running:
            try:
                await self._check_all_providers()
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

            await asyncio.sleep(self.interval_seconds)

    async def _check_all_providers(self) -> None:
        """Ping all configured providers in parallel."""
        # Get active adapters/providers from integration
        adapters = self.router_integration.adapters

        async def check_single_provider(
            provider_name: str, adapter
        ) -> tuple[str, Dict[str, Any]]:
            """Check a single provider and return status."""
            try:
                # Use a lightweight model for ping if possible
                models = await adapter.list_models()
                if not models:
                    return provider_name, {"status": "skipped", "reason": "no_models"}

                # Pick the first one
                test_model = models[0]

                start_time = time.time()
                success = await self._ping_provider(provider_name, test_model)
                latency = (time.time() - start_time) * 1000

                status = {
                    "status": "healthy" if success else "unhealthy",
                    "avg_latency_ms": latency if success else None,
                    "last_check": time.time(),
                    "model_used": test_model,
                }

                if not success:
                    logger.warning(f"Health check failed for {provider_name}")

                return provider_name, status

            except Exception as e:
                logger.warning(f"Health check error for {provider_name}: {e}")
                return provider_name, {
                    "status": "error",
                    "error": str(e),
                    "last_check": time.time(),
                }

        # Run all health checks in parallel
        tasks = [
            check_single_provider(name, adapter) for name, adapter in adapters.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Update status for all providers
        for result in results:
            if isinstance(result, BaseException):
                logger.error(f"Health check task failed: {result}")
                continue

            if not isinstance(result, tuple):
                logger.error(
                    f"Health check task returned unexpected type: {type(result)}"
                )
                continue

            provider_name, status = result
            self.provider_status[provider_name] = status

    async def _ping_provider(self, provider: str, model: str) -> bool:
        """Send a minimal request to the provider."""
        try:
            # We use litellm directly or via adapter?
            # Adapter has 'chat_completions' method.
            # But adapter requires full request dict.
            # Let's try to use the adapter interface if possible to reuse auth logic.

            request = {
                "model": f"{provider}/{model}",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            }

            # Use adapter
            adapter = self.router_integration.adapters.get(provider)
            if not adapter:
                return False

            # Execute
            await adapter.chat_completions(request)
            return True

        except Exception as e:
            logger.debug(f"Ping failed for {provider}/{model}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Return current health stats."""
        return self.provider_status
