# src/proxy_app/router_core.py
"""
Core router logic with streaming-aware timeout handling.
"""

import json
import logging
from typing import Dict, Any, Optional, List
import httpx

from rotator_library.timeout_config import TimeoutConfig
from rotator_library.client import get_timeout_for_request, is_streaming_request

logger = logging.getLogger(__name__)


class RouterCore:
    """
    Core routing logic that applies appropriate timeouts based on request type.
    """
    
    def __init__(self):
        self.timeout_stats = {
            "streaming_timeouts": 0,
            "non_streaming_timeouts": 0,
            "total_streaming": 0,
            "total_non_streaming": 0
        }
    
    def classify_request(self, body: bytes) -> tuple[bool, httpx.Timeout]:
        """
        Classify request as streaming or non-streaming and return appropriate timeout.
        
        Returns:
            Tuple of (is_streaming, timeout_config)
        """
        is_streaming = is_streaming_request(body)
        timeout = TimeoutConfig.streaming() if is_streaming else TimeoutConfig.non_streaming()
        
        # Update stats
        if is_streaming:
            self.timeout_stats["total_streaming"] += 1
        else:
            self.timeout_stats["total_non_streaming"] += 1
            
        return is_streaming, timeout
    
    def get_provider_client(self, is_streaming: bool) -> httpx.AsyncClient:
        """
        Get HTTP client configured for the specific request type.
        
        Creates clients with distinct timeout configurations to ensure
        streaming requests don't use excessive timeouts and non-streaming
        requests have sufficient time for generation.
        """
        timeout = TimeoutConfig.streaming() if is_streaming else TimeoutConfig.non_streaming()
        
        return httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            http2=True
        )
    
    def handle_timeout_error(self, error: httpx.TimeoutException, is_streaming: bool, provider: str):
        """
        Handle timeout errors with appropriate logging and stats.
        
        Args:
            error: The timeout exception
            is_streaming: Whether this was a streaming request
            provider: Provider identifier
        """
        if is_streaming:
            self.timeout_stats["streaming_timeouts"] += 1
            logger.warning(
                f"Streaming timeout with provider {provider}: "
                f"No data received for {TimeoutConfig.read_streaming()}s. "
                f"Provider may be stalled or overloaded."
            )
        else:
            self.timeout_stats["non_streaming_timeouts"] += 1
            logger.warning(
                f"Non-streaming timeout with provider {provider}: "
                f"Request exceeded {TimeoutConfig.read_non_streaming()}s. "
                f"Consider increasing TIMEOUT_READ_NON_STREAMING for large generations."
            )
    
    def get_timeout_recommendations(self) -> List[str]:
        """
        Generate recommendations based on timeout statistics.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        streaming_rate = (
            self.timeout_stats["streaming_timeouts"] / max(self.timeout_stats["total_streaming"], 1)
        )
        non_streaming_rate = (
            self.timeout_stats["non_streaming_timeouts"] / max(self.timeout_stats["total_non_streaming"], 1)
        )
        
        if streaming_rate > 0.1:  # More than 10% streaming timeouts
            recommendations.append(
                f"High streaming timeout rate ({streaming_rate:.1%}). "
                f"Current chunk timeout: {TimeoutConfig.read_streaming()}s. "
                "Providers may be sending keepalives too slowly."
            )
        
        if non_streaming_rate > 0.05:  # More than 5% non-streaming timeouts
            recommendations.append(
                f"High completion timeout rate ({non_streaming_rate:.1%}). "
                f"Current timeout: {TimeoutConfig.read_non_streaming()}s. "
                "Consider increasing TIMEOUT_READ_NON_STREAMING for long outputs."
            )
        
        return recommendations
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current timeout statistics."""
        return {
            **self.timeout_stats,
            "streaming_timeout_rate": (
                self.timeout_stats["streaming_timeouts"] / 
                max(self.timeout_stats["total_streaming"], 1)
            ),
            "non_streaming_timeout_rate": (
                self.timeout_stats["non_streaming_timeouts"] / 
                max(self.timeout_stats["total_non_streaming"], 1)
            ),
            "current_timeouts": {
                "streaming_read": TimeoutConfig.read_streaming(),
                "non_streaming_read": TimeoutConfig.read_non_streaming(),
                "connect": TimeoutConfig.connect(),
                "write": TimeoutConfig.write()
            }
        }
