"""
Request Deduplication Module

Prevents duplicate concurrent requests to the same provider/model
by caching in-flight requests and returning the same response to all clients.
"""

import asyncio
import hashlib
import json
from typing import Dict, Any, Optional, AsyncGenerator, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PendingRequest:
    """Tracks an in-flight request and its waiting clients."""

    request_hash: str
    response: Optional[Any] = None
    error: Optional[Exception] = None
    completed: bool = False
    waiters: list = None

    def __post_init__(self):
        if self.waiters is None:
            self.waiters = []


class RequestDeduplicator:
    """
    Deduplicates identical concurrent requests.

    When multiple clients request the same model with the same messages,
    only one request goes to the provider. All clients receive the same response.
    """

    def __init__(self):
        self._pending: Dict[str, PendingRequest] = {}
        self._lock = asyncio.Lock()

    def _hash_request(self, request: Dict[str, Any]) -> str:
        """
        Create a hash of the request for deduplication.

        Only considers model and messages (not temperature, max_tokens, etc.)
        to maximize deduplication opportunities.
        """
        # Extract key fields for deduplication
        dedup_key = {
            "model": request.get("model", ""),
            "messages": request.get("messages", []),
        }

        # Create deterministic JSON string
        key_str = json.dumps(dedup_key, sort_keys=True, separators=(",", ":"))
        return hashlib.md5(key_str.encode()).hexdigest()

    async def execute_or_wait(
        self, request: Dict[str, Any], handler: callable
    ) -> Union[Dict[str, Any], AsyncGenerator]:
        """
        Execute request or wait for existing identical request to complete.

        Args:
            request: The API request dict
            handler: Async function that actually makes the provider call

        Returns:
            Response from handler (either dict or async generator)
        """
        request_hash = self._hash_request(request)

        async with self._lock:
            # Check if identical request is already in-flight
            if request_hash in self._pending:
                pending = self._pending[request_hash]

                # Create an event to wait for completion
                completion_event = asyncio.Event()
                pending.waiters.append(completion_event)

                logger.info(
                    f"Request deduplication: Waiting for identical request {request_hash[:8]}..."
                )

                # Release lock while waiting
                async with self._lock:
                    pass  # Exit lock context

                # Wait for the original request to complete
                await completion_event.wait()

                # Return the cached result
                if pending.error:
                    raise pending.error
                return pending.response

        # No existing request - we're the primary
        pending = PendingRequest(request_hash=request_hash)

        async with self._lock:
            self._pending[request_hash] = pending

        try:
            logger.info(
                f"Request deduplication: Primary request {request_hash[:8]} executing..."
            )

            # Execute the actual request
            response = await handler()

            # Store result
            async with self._lock:
                pending.response = response
                pending.completed = True

                # Notify all waiters
                for waiter in pending.waiters:
                    waiter.set()

            return response

        except Exception as e:
            # Store error
            async with self._lock:
                pending.error = e
                pending.completed = True

                # Notify all waiters
                for waiter in pending.waiters:
                    waiter.set()

            raise

        finally:
            # Cleanup
            async with self._lock:
                if request_hash in self._pending:
                    del self._pending[request_hash]
