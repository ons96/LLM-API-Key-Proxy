import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CacheEntry:
    expires_at: float
    value: Dict[str, Any]


class ResponseCache:
    """Simple in-memory TTL cache for non-streaming OpenAI-compatible responses."""

    def __init__(self) -> None:
        self.enabled = os.getenv("RESPONSE_CACHE_ENABLED", "false").lower() == "true"
        self.ttl_seconds = int(os.getenv("RESPONSE_CACHE_TTL_SECONDS", "30"))
        self.max_items = int(os.getenv("RESPONSE_CACHE_MAX_ITEMS", "1000"))
        self._store: Dict[str, CacheEntry] = {}

    def make_key(self, request_data: Dict[str, Any]) -> str:
        relevant: Dict[str, Any] = {}
        for k in (
            "model",
            "messages",
            "temperature",
            "top_p",
            "max_tokens",
            "tools",
            "tool_choice",
            "presence_penalty",
            "frequency_penalty",
            "n",
            "stop",
            "seed",
            "response_format",
            "reasoning_effort",
        ):
            if k in request_data:
                relevant[k] = request_data[k]

        payload = json.dumps(relevant, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        entry = self._store.get(key)
        if not entry:
            return None

        if entry.expires_at < time.time():
            self._store.pop(key, None)
            return None

        return entry.value

    def set(self, key: str, value: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        if self.ttl_seconds <= 0:
            return

        if len(self._store) >= self.max_items:
            # naive eviction: drop an arbitrary element
            self._store.pop(next(iter(self._store)))

        self._store[key] = CacheEntry(expires_at=time.time() + self.ttl_seconds, value=value)

    def purge(self) -> None:
        now = time.time()
        expired = [k for k, v in self._store.items() if v.expires_at < now]
        for k in expired:
            self._store.pop(k, None)
