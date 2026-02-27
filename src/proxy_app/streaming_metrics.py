"""
Streaming metrics collection for Phase 4.2 improvements.
Tracks time-to-first-token, throughput, and chunk-level metrics.
"""

import time
import json
import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, AsyncGenerator, Callable, Union
from datetime import datetime


@dataclass
class StreamingMetrics:
    """Metrics collected during a streaming response."""
    request_id: str
    model: str
    start_time: float
    first_token_time: Optional[float] = None
    end_time: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_chunks: int = 0
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def time_to_first_token_ms(self) -> Optional[float]:
        """Time from request start to first token in milliseconds."""
        if self.first_token_time:
            return (self.first_token_time - self.start_time) * 1000
        return None
    
    @property
    def total_duration_ms(self) -> Optional[float]:
        """Total duration of the stream in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None
    
    @property
    def tokens_per_second(self) -> Optional[float]:
        """Calculate tokens per second throughput."""
        duration = self.total_duration_ms
        if duration and duration > 0:
            return (self.output_tokens / duration) * 1000
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            "request_id": self.request_id,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_chunks": self.total_chunks,
            "error_count": self.error_count,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "total_duration_ms": self.total_duration_ms,
            "tokens_per_second": self.tokens_per_second,
            "metadata": self.metadata,
            "timestamp": datetime.utcnow().isoformat(),
        }


class StreamingMetricsCollector:
    """Collects metrics for streaming responses."""
    
    def __init__(self, request_id: str, model: str, metadata: Optional[Dict] = None):
        self.metrics = StreamingMetrics(
            request_id=request_id,
            model=model,
            start_time=time.time(),
            metadata=metadata or {}
        )
        self._first_token = True
        self._active = True
    
    def record_first_token(self):
        """Record the time of the first token."""
        if self._first_token and self._active:
            self.metrics.first_token_time = time.time()
            self._first_token = False
    
    def record_chunk(self, chunk_data: Optional[Union[Dict, str, bytes]] = None):
        """Record a chunk being sent and extract token info if available."""
        if not self._active:
            return
            
        self.metrics.total_chunks += 1
        
        # Parse chunk if it's bytes or string
        parsed_data = None
        if isinstance(chunk_data, (bytes, str)):
            parsed_data = self._parse_chunk(chunk_data)
        elif isinstance(chunk_data, dict):
            parsed_data = chunk_data
            
        if parsed_data:
            self._extract_usage_info(parsed_data)
    
    def _parse_chunk(self, chunk: Union[bytes, str]) -> Optional[Dict]:
        """Parse SSE formatted chunk to extract JSON data."""
        try:
            if isinstance(chunk, bytes):
                text = chunk.decode('utf-8')
            else:
                text = chunk
                
            # Handle SSE format: data: {...}\n\n
            if text.startswith('data: '):
                text = text[6:]
                
            text = text.strip()
            if text == '[DONE]' or not text:
                return None
                
            return json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
    
    def _extract_usage_info(self, data: Dict):
        """Extract token usage from chunk data."""
        # Handle usage field in chunk (some providers send this in the last chunk)
        if "usage" in data and data["usage"]:
            usage = data["usage"]
            if isinstance(usage, dict):
                self.metrics.output_tokens = usage.get("completion_tokens", self.metrics.output_tokens)
                self.metrics.input_tokens = usage.get("prompt_tokens", self.metrics.input_tokens)
        
        # Count content in delta
        if "choices" in data and data["choices"]:
            for choice in data["choices"]:
                delta = choice.get("delta", {})
                if delta and "content" in delta and delta["content"]:
                    # Estimate tokens: ~4 chars per token for English text
                    content = delta["content"]
                    self.metrics.output_tokens += max(1, len(content) // 4)
    
    def record_error(self, error: Optional[Exception] = None):
        """Record an error during streaming."""
        if self._active:
            self.metrics.error_count += 1
            if error:
                self.metrics.metadata["last_error"] = f"{type(error).__name__}: {str(error)}"
    
    def finalize(self) -> StreamingMetrics:
        """Finalize metrics collection and return the metrics."""
        if self._active:
            self.metrics.end_time = time.time()
            self._active = False
        return self.metrics
    
    async def wrap_generator(
        self, 
        generator: AsyncGenerator[Union[str, bytes], None],
        log_callback: Optional[Callable[[StreamingMetrics], None]] = None
    ) -> AsyncGenerator[Union[str, bytes], None]:
        """
        Wrap an async generator to collect metrics.
        
        Args:
            generator: The async generator yielding stream chunks
            log_callback: Optional callback to call with final metrics when done
            
        Yields:
            Chunks from the original generator
        """
        try:
            async for chunk in generator:
                # Record first token timing on first chunk
                if self.metrics.total_chunks == 0:
                    self.record_first_token()
                
                self.record_chunk(chunk)
                yield chunk
                
        except Exception as e:
            self.record_error(e)
            raise
        finally:
            metrics = self.finalize()
            if log_callback:
                try:
                    log_callback(metrics)
                except Exception:
                    # Don't let logging errors break the response
                    pass


class StreamingMetricsMiddleware:
    """Middleware-style wrapper for streaming endpoints."""
    
    def __init__(self, detailed_logger=None):
        self.logger = detailed_logger
    
    async def wrap_streaming_response(
        self,
        request_id: str,
        model: str,
        generator: AsyncGenerator[Union[str, bytes], None],
        metadata: Optional[Dict] = None
    ) -> AsyncGenerator[Union[str, bytes], None]:
        """Wrap a streaming generator with metrics collection."""
        collector = StreamingMetricsCollector(
            request_id=request_id,
            model=model,
            metadata=metadata
        )
        
        callback = None
        if self.logger and hasattr(self.logger, 'log_streaming_metrics'):
            callback = self.logger.log_streaming_metrics
            
        async for chunk in collector.wrap_generator(generator, callback):
            yield chunk
