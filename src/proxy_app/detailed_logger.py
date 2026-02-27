"""
Detailed logging infrastructure for proxy requests and streaming metrics.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

from proxy_app.streaming_metrics import StreamingMetrics


class DetailedLogger:
    """Handles detailed logging of requests, responses, and streaming metrics."""
    
    def __init__(self, log_dir: Optional[Path] = None, enable_console: bool = True):
        self.logger = logging.getLogger("proxy_app.detailed_logger")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers to avoid duplicates on re-init
        self.logger.handlers = []
        
        if enable_console:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # File logging setup
        self.log_dir = log_dir or Path(os.getenv("PROXY_LOG_DIR", "logs"))
        self._file_handler = None
        
        if self.log_dir:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                # We'll use rotating file handlers per day
                self._setup_file_logging()
            except Exception as e:
                self.logger.warning(f"Could not setup file logging: {e}")
    
    def _setup_file_logging(self):
        """Setup file logging with daily rotation."""
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"proxy_{date_str}.jsonl"
        
        # Use a file handler for structured logs
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        # Simple formatter for file - we'll write JSON directly
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(file_handler)
        self._file_handler = file_handler
    
    def _write_structured_log(self, data: Dict[str, Any]):
        """Write a structured log entry."""
        log_entry = json.dumps(data, default=str)
        
        if self._file_handler:
            # Write to file directly as JSON
            with open(self._file_handler.baseFilename, "a") as f:
                f.write(log_entry + "\n")
        
        # Also log to console at debug level
        self.logger.debug(log_entry)
    
    def log_request_start(
        self, 
        request_id: str, 
        endpoint: str, 
        model: str,
        stream: bool = False,
        **kwargs
    ):
        """Log the start of a request."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "event_type": "request_start",
            "endpoint": endpoint,
            "model": model,
            "stream": stream,
            **kwargs
        }
        self._write_structured_log(log_data)
        
        if stream:
            self.logger.info(f"[{request_id[:8]}] Streaming request started for model: {model}")
    
    def log_request_end(
        self, 
        request_id: str, 
        model: str,
        status_code: int = 200,
        error: Optional[str] = None,
        **kwargs
    ):
        """Log the end of a request."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "event_type": "request_end",
            "model": model,
            "status_code": status_code,
            "error": error,
            **kwargs
        }
        self._write_structured_log(log_data)
    
    def log_streaming_metrics(self, metrics: StreamingMetrics):
        """
        Log metrics collected from a streaming response.
        This is the callback used by StreamingMetricsCollector.
        """
        metrics_dict = metrics.to_dict()
        
        # Structured log entry
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": metrics.request_id,
            "event_type": "streaming_metrics",
            "metrics": metrics_dict
        }
        self._write_structured_log(log_data)
        
        # Human-readable console summary
        ttft = metrics.time_to_first_token_ms
        tps = metrics.tokens_per_second
        duration = metrics.total_duration_ms
        
        summary_parts = [
            f"[{metrics.request_id[:8]}] Stream complete",
            f"Model: {metrics.model}",
            f"Chunks: {metrics.total_chunks}",
        ]
        
        if ttft is not None:
            summary_parts.append(f"TTFT: {ttft:.1f}ms")
        if tps is not None:
            summary_parts.append(f"TPS: {tps:.1f}")
        if duration is not None:
            summary_parts.append(f"Duration: {duration:.1f}ms")
        if metrics.output_tokens > 0:
            summary_parts.append(f"Tokens: {metrics.output_tokens}")
        if metrics.error_count > 0:
            summary_parts.append(f"Errors: {metrics.error_count}")
            
        self.logger.info(" | ".join(summary_parts))
    
    def log_provider_error(
        self, 
        request_id: str, 
        provider: str, 
        error: Exception,
        model: Optional[str] = None
    ):
        """Log provider-specific errors."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "event_type": "provider_error",
            "provider": provider,
            "model": model,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        self._write_structured_log(log_data)
        self.logger.error(
            f"[{request_id[:8]}] Provider error ({provider}): {type(error).__name__}: {str(error)}"
        )
    
    def log_router_decision(
        self,
        request_id: str,
        model: str,
        selected_provider: str,
        available_providers: int,
        routing_reason: Optional[str] = None
    ):
        """Log routing decisions."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "event_type": "router_decision",
            "model": model,
            "selected_provider": selected_provider,
            "available_providers": available_providers,
            "routing_reason": routing_reason,
        }
        self._write_structured_log(log_data)
