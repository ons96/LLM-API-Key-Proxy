import json
import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional, Callable
from datetime import datetime

# Context variable to store request ID across async boundaries
request_id_context: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging with request context."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add request ID if available in context
        current_request_id = request_id_context.get()
        if current_request_id:
            log_obj["request_id"] = current_request_id
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        # Add any extra fields from record
        for attr in ["path", "method", "status_code", "duration_ms", "model", "provider", "client_ip"]:
            if hasattr(record, attr):
                if "extra" not in log_obj:
                    log_obj["extra"] = {}
                log_obj["extra"][attr] = getattr(record, attr)
        
        # Standard source fields
        log_obj["source"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }
        
        return json.dumps(log_obj, default=str)


def get_request_id() -> Optional[str]:
    """Get current request ID from context."""
    return request_id_context.get()


def set_request_id(request_id: Optional[str]) -> None:
    """Set request ID in context."""
    request_id_context.set(request_id)


def generate_request_id() -> str:
    """Generate a new request ID."""
    return str(uuid.uuid4())


def setup_json_logging(level: int = logging.INFO, enable_json: bool = True) -> None:
    """Setup JSON logging configuration."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add stdout handler
    handler = logging.StreamHandler(sys.stdout)
    
    if enable_json:
        formatter = JSONFormatter()
    else:
        # Fallback to standard format if JSON disabled
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


class RequestIDMiddleware:
    """ASGI Middleware to handle request IDs for structured logging."""
    
    def __init__(self, app: Callable, header_name: str = "X-Request-ID"):
        self.app = app
        self.header_name = header_name
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        # Extract or generate request ID
        headers = dict(scope.get("headers", []))
        request_id: Optional[str] = None
        
        # Look for existing request ID in headers (case-insensitive)
        header_name_bytes = self.header_name.lower().encode()
        for key, value in headers.items():
            if key.lower() == header_name_bytes:
                request_id = value.decode()
                break
        
        if not request_id:
            request_id = generate_request_id()
        
        # Set in context
        token = request_id_context.set(request_id)
        
        # Wrap send to include request ID in response headers
        async def send_with_request_id(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((self.header_name.encode(), request_id.encode()))
                message["headers"] = headers
            await send(message)
        
        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            request_id_context.reset(token)
