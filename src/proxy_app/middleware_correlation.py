"""Correlation ID middleware for request tracking across all services."""
import uuid
from typing import Optional
from contextvars import ContextVar
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import logging

logger = logging.getLogger(__name__)

# Context variable to store correlation ID throughout the request lifecycle
# This allows it to be accessed in any async context within the same request
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID from the context."""
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID in the context."""
    correlation_id_var.set(correlation_id)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that ensures every request has a correlation ID.
    
    It checks for an incoming X-Correlation-ID header, and if not present,
    generates a new UUID. The correlation ID is then available throughout
    the request lifecycle via the context variable.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Check for incoming correlation ID header (case-insensitive)
        correlation_id = request.headers.get("X-Correlation-ID")
        
        # Also check for common alternative header names
        if not correlation_id:
            correlation_id = request.headers.get("X-Request-ID")
        if not correlation_id:
            correlation_id = request.headers.get("X-Trace-ID")
        if not correlation_id:
            correlation_id = request.headers.get("Request-ID")
        
        # Generate new correlation ID if not provided
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        # Store in context variable for the duration of this request
        token = correlation_id_var.set(correlation_id)
        
        # Add correlation ID to request state for easy access
        request.state.correlation_id = correlation_id
        
        # Process the request
        try:
            response = await call_next(request)
        finally:
            # Always restore the previous context
            correlation_id_var.reset(token)
        
        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        
        return response


def generate_correlation_id() -> str:
    """Generate a new correlation ID (UUID v4)."""
    return str(uuid.uuid4())
