"""Build client-safe HTTP error details from exception metadata only.

Upstream provider exceptions (LiteLLM, provider SDKs, raw HTTP clients) embed
response bodies, request URLs and sometimes credential material in
``str(exc)``. Anything copied into an HTTP ``detail`` field or an SSE payload
is read by the caller, so those strings must never cross the boundary.

The full message is always logged server-side; the client gets a stable,
metadata-only detail (exception class + request id) it can correlate with the
logs.
"""

import re
from typing import Optional

__all__ = ["safe_error_detail"]

# Exception class names are attacker-influenced metadata: keep identifier
# characters only and cap the length so a hostile name cannot bloat the body.
_TYPE_NAME_RE = re.compile(r"[^0-9A-Za-z_]")
_MAX_TYPE_NAME = 64
_MAX_REQUEST_ID = 64


def safe_error_detail(
    exc: BaseException,
    *,
    request_id: Optional[str] = None,
    prefix: str = "Router error",
) -> str:
    """Return ``"<prefix> (<ExceptionClass>); request_id=<id>"``.

    Never includes ``str(exc)``. ``request_id`` lets the caller quote the id
    in a bug report so the server-side log line (which does carry the full
    message) can be found.
    """
    type_name = _TYPE_NAME_RE.sub("", type(exc).__name__)[:_MAX_TYPE_NAME]
    if not type_name:
        type_name = "Exception"

    detail = f"{prefix} ({type_name})"
    if request_id:
        safe_id = _TYPE_NAME_RE.sub("", str(request_id))[:_MAX_REQUEST_ID]
        if safe_id:
            detail = f"{detail}; request_id={safe_id}"
    return detail
