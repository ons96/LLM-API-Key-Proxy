"""Detailed logger with correlation ID support for request tracking."""
import logging
import json
import time
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from proxy_app.middleware_correlation import get_correlation_id, correlation_id_var


class DetailedLogger:
    """
    Enhanced logging utility that includes correlation IDs for request tracing.
    """

    def __init__(self, name: str = "detailed_logger", log_dir: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        self.log_dir = log_dir
        self._log_file = None
        
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = log_dir / f"detailed_{datetime.now().strftime('%Y%m%d')}.log"

    def _format_message(self, message: str, extra: Optional[Dict[str, Any]] = None) -> str:
        """Format log message with correlation ID and optional extra data."""
        correlation_id = get_correlation_id()
        
        parts = [
            f"[corr_id={correlation_id}]" if correlation_id else "[corr_id=N/A]",
            message
        ]
        
        if extra:
            parts.append(json.dumps(extra, default=str))
        
        return " ".join(parts)

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log debug level message."""
        formatted = self._format_message(message, extra)
        self.logger.debug(formatted)
        self._write_to_file("DEBUG", formatted)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log info level message."""
        formatted = self._format_message(message, extra)
        self.logger.info(formatted)
        self._write_to_file("INFO", formatted)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log warning level message."""
        formatted = self._format_message(message, extra)
        self.logger.warning(formatted)
        self._write_to_file("WARNING", formatted)

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False) -> None:
        """Log error level message."""
        formatted = self._format_message(message, extra)
        self.logger.error(formatted, exc_info=exc_info)
        self._write_to_file("ERROR", formatted)

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log critical level message."""
        formatted = self._format_message(message, extra)
        self.logger.critical(formatted)
        self._write_to_file("CRITICAL", formatted)

    def log_request_start(
        self,
        method: str,
        endpoint: str,
        model: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log the start of an API request."""
        data = {
            "event": "request_start",
            "method": method,
            "endpoint": endpoint,
            "model": model,
            "timestamp": datetime.now().isoformat()
        }
        if extra:
            data.update(extra)
        
        self.info(f"Request started: {method} {endpoint}", data)

    def log_request_end(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_ms: float,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log the end of an API request."""
        data = {
            "event": "request_end",
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "timestamp": datetime.now().isoformat()
        }
        if extra:
            data.update(extra)
        
        self.info(f"Request completed: {method} {endpoint} - {status_code} ({duration_ms:.2f}ms)", data)

    def log_provider_call(
        self,
        provider: str,
        model: str,
        success: bool,
        duration_ms: float,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a provider API call."""
        data = {
            "event": "provider_call",
            "provider": provider,
            "model": model,
            "success": success,
            "duration_ms": round(duration_ms, 2),
            "timestamp": datetime.now().isoformat()
        }
        if error:
            data["error"] = error
        if extra:
            data.update(extra)
        
        status = "SUCCESS" if success else "FAILED"
        self.info(f"Provider call {status}: {provider}/{model} ({duration_ms:.2f}ms)", data)

    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an error with full context."""
        data = {
            "event": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat()
        }
        if context:
            data["context"] = context
        
        self.error(f"Error: {type(error).__name__}: {str(error)}", data, exc_info=True)

    def _write_to_file(self, level: str, message: str) -> None:
        """Write log message to file if configured."""
        if self._log_file:
            try:
                with open(self._log_file, "a") as f:
                    f.write(f"{datetime.now().isoformat()} [{level}] {message}\n")
            except Exception:
                pass  # Don't let file logging failures break the app
