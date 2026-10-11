"""Build router headers for an OpenCode plugin or client wrapper."""

from collections.abc import Mapping


def router_headers(
    session_id: str,
    phase: str = "",
    action: str = "",
    verification_failed: bool = False,
) -> Mapping[str, str]:
    """Return bounded, provider-neutral headers for a router request."""
    headers = {"X-Session-Id": session_id[:256]}
    if phase:
        headers["X-Router-Phase"] = phase[:32]
    if action:
        headers["X-Router-Action"] = action[:256]
    if verification_failed:
        headers["X-Router-Verification"] = "failed"
    return headers
