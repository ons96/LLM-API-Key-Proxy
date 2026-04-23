"""
Security middleware for FastAPI application.
Implements security headers, request validation, and basic protections.
"""

from fastapi import Request, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses.
    OWASP compliant security headers.
    """
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # XSS Protection
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Content Security Policy (restrictive for API)
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; frame-ancestors 'none'"
        )
        
        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions Policy (prevent access to sensitive APIs)
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
            "magnetometer=(), microphone=(), payment=(), usb=()"
        )
        
        # Strict Transport Security (HTTPS only)
        # Note: Only enable if you're sure HTTPS is always available
        # response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


class InputValidationMiddleware(BaseHTTPMiddleware):
    """
    Validate incoming requests for common attack patterns.
    Basic protection against injection attacks.
    """
    
    # Patterns that might indicate prompt injection or attacks
    SUSPICIOUS_PATTERNS = [
        r"ignore\s+(previous|above|all)\s+instructions",
        r"system\s+prompt\s*:",
        r"ignore\s+your\s+(programming|instructions)",
        r"DAN\s+(mode|prompt)",
        r"jailbreak",
        r"\"\"\"[\s\S]*?ignore",
    ]
    
    def __init__(self, app, max_body_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_body_size = max_body_size
    
    async def dispatch(self, request: Request, call_next):
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length > self.max_body_size:
                    logger.warning(f"Request too large: {length} bytes")
                    raise HTTPException(status_code=413, detail="Payload too large")
            except ValueError:
                pass
        
        # Check for suspicious patterns in query params
        for key, value in request.query_params.items():
            if self._contains_suspicious_content(value):
                logger.warning(f"Suspicious content in query param: {key}")
                raise HTTPException(status_code=400, detail="Invalid input detected")
        
        response = await call_next(request)
        return response
    
    def _contains_suspicious_content(self, text: str) -> bool:
        """Check if text contains suspicious patterns."""
        text_lower = text.lower()
        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False


class AuditLogMiddleware(BaseHTTPMiddleware):
    """
    Log security-relevant events for audit purposes.
    Does not log sensitive data like API keys or message content.
    """
    
    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self.audit_logger = logging.getLogger("audit")
    
    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)
        
        # Log authentication attempts (without credentials)
        auth_header = request.headers.get("authorization", "")
        has_auth = bool(auth_header)
        
        # Extract client info
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")[:100]  # Truncate
        
        # Log the request (sanitized)
        self.audit_logger.info(
            f"Request: method={request.method} "
            f"path={request.url.path} "
            f"client={client_ip} "
            f"authenticated={has_auth} "
            f"ua={user_agent}"
        )
        
        try:
            response = await call_next(request)
            
            # Log response status
            self.audit_logger.info(
                f"Response: path={request.url.path} "
                f"status={response.status_code} "
                f"client={client_ip}"
            )
            
            return response
            
        except HTTPException as exc:
            # Log authentication failures
            if exc.status_code in (401, 403):
                self.audit_logger.warning(
                    f"Auth failure: path={request.url.path} "
                    f"status={exc.status_code} "
                    f"client={client_ip}"
                )
            raise
        
        except Exception as exc:
            # Log server errors
            self.audit_logger.error(
                f"Server error: path={request.url.path} "
                f"error={type(exc).__name__} "
                f"client={client_ip}"
            )
            raise


def setup_security_middleware(app, allowed_hosts: Optional[list] = None):
    """
    Configure security middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        allowed_hosts: List of allowed hostnames (e.g., ["api.example.com"])
    """
    # Security headers for all responses
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Input validation
    app.add_middleware(InputValidationMiddleware, max_body_size=10 * 1024 * 1024)
    
    # Audit logging
    app.add_middleware(AuditLogMiddleware, enabled=True)
    
    # Trusted host validation (if hosts specified)
    if allowed_hosts:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=allowed_hosts
        )
    
    logger.info("Security middleware configured")
