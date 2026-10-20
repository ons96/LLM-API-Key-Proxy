"""The routing decision record and its response header.

Every response carries ``X-Gateway-Decision``: the chosen alias, provider, model,
quota tier, attempt index, fallback count, wait time and a reason code. One header
makes every routing bug reproducible from a client log.

Format is a compact ``key=value`` list separated by ``;`` so it survives proxies
that rewrote headers (no spaces, ASCII only)::

    alias=auto;provider=groq;model=llama-3.3-70b;tier=0;attempt=0;fallbacks=1;wait_ms=0;reason=ok
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

HEADER_NAME = "X-Gateway-Decision"

REASON_CODES = {
    "ok",
    "fallback",
    "capability_mismatch",
    "quota_exhausted",
    "paid_model_rejected",
    "all_failed",
    "selected",
}


@dataclass
class Decision:
    alias: str = ""
    provider: str = ""
    model: str = ""
    tier: Optional[int] = None
    attempt: int = 0
    fallbacks: int = 0
    wait_ms: int = 0
    reason: str = "selected"

    def to_header(self) -> str:
        parts = [
            f"alias={_clean(self.alias)}",
            f"provider={_clean(self.provider)}",
            f"model={_clean(self.model)}",
            f"tier={self.tier if self.tier is not None else -1}",
            f"attempt={self.attempt}",
            f"fallbacks={self.fallbacks}",
            f"wait_ms={self.wait_ms}",
            f"reason={_clean(self.reason)}",
        ]
        return ";".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean(value: Any) -> str:
    """Header-safe: no separators, no control characters, ASCII only."""
    text = str(value or "")
    for bad in (";", ",", "\n", "\r", "\t", " "):
        text = text.replace(bad, "_")
    return text.encode("ascii", "replace").decode("ascii")[:120]


def headers_for(decision: Decision | None) -> dict[str, str]:
    """Response headers to merge into a JSON or streaming response."""
    if decision is None:
        return {}
    return {HEADER_NAME: decision.to_header()}
