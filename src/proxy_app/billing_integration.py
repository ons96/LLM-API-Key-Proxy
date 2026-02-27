#!/usr/bin/env python3
"""
Billing Integration Module

Provides helper functions to integrate billing tracking into the proxy request flow.
This module acts as a bridge between the proxy logic and the BillingTracker.
"""

import logging
from typing import Optional, Dict, Any
from .billing_tracker import BillingTracker

logger = logging.getLogger(__name__)

# Singleton instance for the application
_tracker_instance: Optional[BillingTracker] = None


def get_billing_tracker() -> BillingTracker:
    """
    Get the singleton BillingTracker instance.
    Initializes it on first call.
    """
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = BillingTracker()
    return _tracker_instance


def track_request_cost(
    provider: str,
    model: str,
    usage_data: Dict[str, Any],
    tracker: Optional[BillingTracker] = None,
):
    """
    Track the cost of a completed request.

    Args:
        provider: The provider used (e.g., 'openai').
        model: The model used (e.g., 'gpt-4o').
        usage_data: Dictionary containing usage information.
                    Expected keys: 'prompt_tokens', 'completion_tokens'.
                    Some APIs might use 'input_tokens'/'output_tokens'.
        tracker: Optional BillingTracker instance. If None, uses singleton.
    """
    if tracker is None:
        tracker = get_billing_tracker()

    # Normalize token keys
    input_tokens = usage_data.get("prompt_tokens") or usage_data.get("input_tokens", 0)
    output_tokens = (
        usage_data.get("completion_tokens") or usage_data.get("output_tokens", 0)
    )

    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        logger.warning(
            f"Invalid token data types for {provider}/{model}: {usage_data}"
        )
        return

    if input_tokens == 0 and output_tokens == 0:
        # Streamed responses or errors might not report usage immediately
        logger.debug(f"No token usage reported for {provider}/{model}")
        return

    try:
        tracker.log_usage(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except Exception as e:
        # Don't let billing errors break the proxy flow
        logger.error(f"Failed to track billing for {provider}/{model}: {e}")
