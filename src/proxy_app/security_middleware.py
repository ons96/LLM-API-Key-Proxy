"""
FastAPI middleware for request/response sanitization and security headers.
"""

import logging
import time
from typing import Callable, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from proxy_app.input_sanitizer import (
    sanitize_headers,
    sanitize_json,
    mask_api_key,
    SanitizationError
)

logger = logging.getLogger(__name__)

# Security headers to add to all responses
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'none'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware to sanitize all incoming requests and outgoing responses.
    Adds security headers and validates request size.
    """
    
    def __init__(self, app, max_body_size: int = 10 * 1024 * 1024):
        super().__init__(app)
        self.max_body_size = max_body_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Store sanitized headers for logging
        request.state.sanitized_headers = sanitize_headers(dict(request.headers))
        
        # Check request body size if Content-Length provided
        content_length = request.headers.get('content-length')
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_body_size:
                    logger.warning(f"Request body too large: {size} bytes")
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "Request entity too large",
                            "max_size": self.max_body_size,
                            "received_size": size
                        }
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid Content-Length header"}
                )
        
        # Process request with timing
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Add security headers
            for header, value in SECURITY_HEADERS.items():
                response.headers[header] = value
            
            # Sanitize response headers to prevent injection
            self._sanitize_response_headers(response)
            
            # Add processing time header (optional, for debugging)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except SanitizationError as e:
            logger.error(f"Sanitization error: {e}")
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid input data", "details": str(e)}
            )
        except Exception as e:
            logger.error(f"Unhandled exception in middleware: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"}
            )
    
    def _sanitize_response_headers(self, response: Response) -> None:
        """Remove dangerous characters from response headers."""
        safe_headers = {}
        for key, value in response.headers.items():
            # Remove CRLF to prevent header injection
            safe_key = str(key).replace('\r', '').replace('\n', '')
            safe_value = str(value).replace('\r', '').replace('\n', '')[:2000]
            safe_headers[safe_key] = safe_value
        
        response.headers.mutable_headers.clear()
        for key, value in safe_headers.items():
            response.headers[key] = value


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Basic rate limiting middleware to prevent DoS.
    Note: For production, use a proper rate limiter like slowapi.
    """
    
    def __init__(self, app, requests_per_minute: int = 1000):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Simple IP-based rate limiting
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old entries
        self.requests = {
            ip: times for ip, times in self.requests.items()
            if current_time - times[-1] < 60
        }
        
        # Check rate
        if client_ip in self.requests:
            recent_requests = len([
                t for t in self.requests[client_ip]
                if current_time - t < 60
            ])
            if recent_requests > self.requests_per_minute:
                logger.warning(f"Rate limit exceeded for {client_ip}")
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded", "retry_after": 60}
                )
            self.requests[client_ip].append(current_time)
        else:
            self.requests[client_ip] = [current_time]
        
        return await call_next(request)


async def sanitize_request_body(request: Request) -> Optional[dict]:
    """
    Dependency to sanitize and validate JSON request body.
    
    Returns:
        Sanitized JSON body or None if empty
        
    Raises:
        SanitizationError: If body cannot be sanitized
    """
    if request.method not in ["POST", "PUT", "PATCH"]:
        return None
    
    try:
        body = await request.json()
        return sanitize_json(body)
    except SanitizationError:
        raise
    except Exception as e:
        logger.warning(f"Failed to parse request body: {e}")
        return None


def sanitize_streaming_response(content: str) -> str:
    """
    Sanitize content in streaming responses.
    
    Args:
        content: SSE or chunked content string
        
    Returns:
        Sanitized content
    """
    # Remove potential SSE injection
    if '\n\n' in content:
        lines = content.split('\n')
        safe_lines = []
        for line in lines:
            # Ensure lines start with expected prefixes
            if line and not any(line.startswith(p) for p in ['data:', 'event:', 'id:', 'retry:', ':']):
                line = 'data: ' + line
            safe_lines.append(line.replace('\r', ''))
        return '\n'.join(safe_lines)
    return content.replace('\r', '')
