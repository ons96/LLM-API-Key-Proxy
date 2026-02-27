"""Request logging utilities with correlation ID support."""
import logging
import time
import json
from typing import Optional, Dict, Any, Union, List
from datetime import datetime
from pathlib import Path

from proxy_app.middleware_correlation import get_correlation_id


# Configure logger
logger = logging.getLogger("request_logger")
logger.setLevel(logging.INFO)

# File handler configuration
LOG_DIR = Path("logs/requests")
LOG_DIR.mkdir(parents=True, exist_ok=True)
REQUEST_LOG_FILE = LOG_DIR / f"requests_{datetime.now().strftime('%Y%m%d')}.jsonl"


def log_request_to_console(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    model: Optional[str] = None,
    error: Optional[str] = None
) -> None:
    """
    Log request details to console with correlation ID.
    """
    correlation_id = get_correlation_id() or "N/A"
    
    # Color codes for console output
    RESET = "\033[0m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    
    # Determine status color
    if status_code < 300:
        status_color = GREEN
    elif status_code < 400:
        status_color = YELLOW
    else:
        status_color = RED
    
    # Build log message
    parts = [
        f"{BLUE}[{datetime.now().strftime('%H:%M:%S')}]{RESET}",
        f"{BLUE}[{correlation_id[:8]}...]{RESET}" if len(correlation_id) > 8 else f"{BLUE}[{correlation_id}]{RESET}",
    ]
    
    if model:
        parts.append(f"{BLUE}model={model}{RESET}")
    
    parts.extend([
        f"{BLUE}{method}{RESET}",
        f"{BLUE}{path}{RESET}",
        f"{status_color}{status_code}{RESET}",
        f"{BLUE}{duration_ms:.2f}ms{RESET}"
    ])
    
    if error:
        parts.append(f"{RED}error={error}{RESET}")
    
    print(" ".join(parts))


def log_request_to_file(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    request_body: Optional[Any] = None,
    response_body: Optional[Any] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    error: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log detailed request information to JSONL file.
    """
    correlation_id = get_correlation_id() or "N/A"
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "correlation_id": correlation_id,
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2),
        "model": model,
        "provider": provider,
    }
    
    # Add request body (truncated for large content)
    if request_body is not None:
        try:
            body_str = json.dumps(request_body, default=str)
            if len(body_str) > 10000:
                body_str = body_str[:10000] + "... [truncated]"
            log_entry["request_body"] = body_str
        except Exception:
            log_entry["request_body"] = str(request_body)[:1000]
    
    # Add response body (truncated)
    if response_body is not None:
        try:
            resp_str = json.dumps(response_body, default=str)
            if len(resp_str) > 10000:
                resp_str = resp_str[:10000] + "... [truncated]"
            log_entry["response_body"] = resp_str
        except Exception:
            log_entry["response_body"] = str(response_body)[:1000]
    
    if error:
        log_entry["error"] = error
    
    if extra:
        log_entry["extra"] = extra
    
    try:
        with open(REQUEST_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write request log: {e}")


def log_embedding_request(
    model: str,
    input_count: int,
    provider: Optional[str] = None,
    success: bool = True,
    error: Optional[str] = None,
    duration_ms: Optional[float] = None
) -> None:
    """Log embedding request details."""
    correlation_id = get_correlation_id() or "N/A"
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "correlation_id": correlation_id,
        "event": "embedding_request",
        "model": model,
        "input_count": input_count,
        "provider": provider,
        "success": success,
        "duration_ms": round(duration_ms, 2) if duration_ms else None,
        "error": error
    }
    
    try:
        with open(REQUEST_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write embedding log: {e}")
