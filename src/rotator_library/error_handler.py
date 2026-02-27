import re
import json
import os
import logging
from typing import Optional, Dict, Any, Union
import httpx

from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
    AuthenticationError,
    InvalidRequestError,
    BadRequestError,
    OpenAIError,
    InternalServerError,
    Timeout,
    ContextWindowExceededError,
)

lib_logger = logging.getLogger("rotator_library")


def _parse_duration_string(duration_str: str) -> Optional[int]:
    """
    Parse duration strings in various formats to total seconds.

    Handles:
    - Compound durations: '156h14m36.752463453s', '2h30m', '45m30s'
    - Simple durations: '562476.752463453s', '3600s', '60m', '2h'
    - Plain seconds (no unit): '562476'

    Args:
        duration_str: Duration string to parse

    Returns:
        Total seconds as integer, or None if parsing fails
    """
    if not duration_str:
        return None

    total_seconds = 0
    remaining = duration_str.strip().lower()

    # Try parsing as plain number first (no units)
    try:
        return int(float(remaining))
    except ValueError:
        pass

    # Parse hours component
    hour_match = re.match(r"(\d+)h", remaining)
    if hour_match:
        total_seconds += int(hour_match.group(1)) * 3600
        remaining = remaining[hour_match.end() :]

    # Parse minutes component
    min_match = re.match(r"(\d+)m", remaining)
    if min_match:
        total_seconds += int(min_match.group(1)) * 60
        remaining = remaining[min_match.end() :]

    # Parse seconds component (including decimals like 36.752463453s)
    sec_match = re.match(r"([\d.]+)s", remaining)
    if sec_match:
        total_seconds += int(float(sec_match.group(1)))

    return total_seconds if total_seconds > 0 else None


def extract_retry_after_from_body(error_body: Optional[str]) -> Optional[int]:
    """
    Extract the retry-after time from an API error response body.

    Handles various error formats including:
    - Gemini CLI: "Your quota will reset after 39s."
    - Antigravity: "quota will reset after 156h14m36s"
    - Generic: "quota will reset after 120s", "retry after 60s"

    Args:
        error_body: The raw error response body

    Returns:
        The retry time in seconds, or None if not found
    """
    if not error_body:
        return None

    # Pattern to match various "reset after" formats - capture the full duration string
    patterns = [
        r"quota will reset after\s*([\dhmso.]+)",  # Matches compound: 156h14m36s or 120s
        r"reset after\s*([\dhmso.]+)",
        r"retry after\s*([\dhmso.]+)",
        r"try again in\s*(\d+)\s*seconds?",
    ]

    for pattern in patterns:
        match = re.search(pattern, error_body, re.IGNORECASE)
        if match:
            duration_str = match.group(1)
            result = _parse_duration_string(duration_str)
            if result is not None:
                return result

    return None


class NoAvailableKeysError(Exception):
    """Raised when no API keys are available for a request after waiting."""

    pass


class PreRequestCallbackError(Exception):
    """Raised when a pre-request callback fails."""

    pass


class CredentialNeedsReauthError(Exception):
    """
    Raised when a credential's refresh token is invalid and re-authentication is required.

    This is a rotatable error - the request should try the next credential while
    the broken credential is queued for re-authentication in the background.

    Unlike generic HTTPStatusError, this exception signals:
    - The credential is temporarily unavailable (needs user action)
    - Re-auth has already been queued
    - The request should rotate to the next credential without logging scary tracebacks

    Attributes:
        credential_path: Path to the credential file that needs re-auth
        message: Human-readable message about the error
    """

    def __init__(self, credential_path: str, message: str = ""):
        self.credential_path = credential_path
        self.message = (
            message or f"Credential '{credential_path}' requires re-authentication"
        )
        super().__init__(self.message)


class EmptyResponseError(Exception):
    """
    Raised when a provider returns an empty response after multiple retry attempts.

    This is a rotatable error - the request should try the next credential.
    Treated as a transient server-side issue (503 equivalent).

    Attributes:
        provider: The provider name (e.g., "antigravity")
        model: The model that was requested
        message: Human-readable message about the error
    """

    def __init__(self, provider: str, model: str, message: str = ""):
        self.provider = provider
        self.model = model
        self.message = (
            message
            or f"Empty response from {provider}/{model} after multiple retry attempts"
        )
        super().__init__(self.message)


# =============================================================================
# STREAMING ERROR RECOVERY (Phase 4.2)
# =============================================================================

class StreamingError(Exception):
    """
    Raised when an error occurs during streaming response generation.
    
    This captures the context of a mid-stream failure to enable recovery logic.
    
    Attributes:
        original_error: The underlying exception that caused the stream to fail
        partial_content: Content received before the error occurred
        credential_id: Identifier for the credential that was being used
        is_recoverable: Whether this error type supports stream recovery
    """
    
    def __init__(
        self, 
        original_error: Exception, 
        partial_content: str = "",
        credential_id: Optional[str] = None,
        is_recoverable: bool = False
    ):
        self.original_error = original_error
        self.partial_content = partial_content
        self.credential_id = credential_id
        self.is_recoverable = is_recoverable
        message = f"Streaming error with credential {credential_id}: {str(original_error)}"
        super().__init__(message)


class StreamingRecoveryError(Exception):
    """
    Raised when stream recovery fails after exhausting all retries.
    
    This is the final error returned to the client when recovery attempts fail.
    """
    
    def __init__(self, attempts: int, last_error: Optional[Exception] = None):
        self.attempts = attempts
        self.last_error = last_error
        message = f"Failed to recover stream after {attempts} attempt(s)"
        if last_error:
            message += f": {str(last_error)}"
        super().__init__(message)


# Abnormal errors that require attention and should always be reported to client
ABNORMAL_ERROR_TYPES = frozenset(
    {
        "forbidden",  # 403 - credential access issue
        "authentication",  # 401 - credential invalid/revoked
        "pre_request_callback_error",  # Internal proxy error
        "unknown",  # Unexpected error - surface per-credential details for debugging
    }
)

# Normal/expected errors during operation - only report if ALL credentials fail
NORMAL_ERROR_TYPES = frozenset(
    {
        "rate_limit",  # 429 - expected during high load
        "quota_exceeded",  # Expected when quota runs out
        "server_error",  # 5xx - transient provider issues
        "api_connection",  # Network issues - transient
    }
)

# Errors that can potentially be recovered from during streaming
RECOVERABLE_STREAMING_ERRORS = frozenset(
    {
        "rate_limit",
        "quota_exceeded", 
        "server_error",
        "api_connection",
        "timeout",
        "empty_response",
    }
)


def is_abnormal_error(error_type: str) -> bool:
    """
    Check if an error is abnormal and should be reported to the client.

    Abnormal errors indicate credential issues that need attention:
    - 403 Forbidden: Credential doesn't have access
    - 401 Unauthorized: Credential is invalid/revoked

    Args:
        error_type: The classified error type string

    Returns:
        True if the error is abnormal, False otherwise
    """
    return error_type in ABNORMAL_ERROR_TYPES


def classify_error(error: Exception) -> str:
    """
    Classify an exception into a standardized error type.
    
    Args:
        error: The exception to classify
        
    Returns:
        String classification of the error
    """
    if isinstance(error, (AuthenticationError,)):
        return "authentication"
    elif isinstance(error, (RateLimitError,)):
        return "rate_limit"
    elif isinstance(error, (ServiceUnavailableError, InternalServerError)):
        return "server_error"
    elif isinstance(error, (APIConnectionError,)):
        return "api_connection"
    elif isinstance(error, (Timeout,)):
        return "timeout"
    elif isinstance(error, (EmptyResponseError,)):
        return "empty_response"
    elif isinstance(error, (CredentialNeedsReauthError,)):
        return "authentication"
    elif isinstance(error, (BadRequestError, InvalidRequestError)):
        return "bad_request"
    elif isinstance(error, (StreamingError,)):
        return classify_error(error.original_error)
    else:
        return "unknown"


def is_streaming_recoverable_error(error: Exception) -> bool:
    """
    Determine if a streaming error can be recovered from via retry.
    
    Args:
        error: The exception that occurred during streaming
        
    Returns:
        True if the error type supports recovery/retry logic
    """
    error_type = classify_error(error)
    return error_type in RECOVERABLE_STREAMING_ERRORS


def create_sse_error_event(
    error: Exception, 
    error_code: Optional[str] = None,
    include_partial: bool = False,
    partial_content: str = ""
) -> str:
    """
    Format an error as a Server-Sent Events (SSE) data chunk.
    
    Args:
        error: The exception to format
        error_code: Optional error code to include
        include_partial: Whether to include partial content received before error
        partial_content: The partial content string if include_partial is True
        
    Returns:
        Formatted SSE string
    """
    error_data = {
        "error": {
            "message": str(error),
            "type": "stream_error",
            "code": error_code or classify_error(error),
        }
    }
    
    if include_partial and partial_content:
        error_data["error"]["partial_content"] = partial_content
        
    return f"data: {json.dumps(error_data)}\n\n"


def create_sse_retry_event(
    retry_count: int, 
    max_retries: int,
    message: str = ""
) -> str:
    """
    Create an SSE event indicating a retry attempt.
    
    Args:
        retry_count: Current retry attempt number
        max_retries: Maximum number of retries allowed
        message: Optional status message
        
    Returns:
        Formatted SSE string
    """
    data = {
        "retry_attempt": retry_count,
        "max_retries": max_retries,
        "message": message or f"Retrying stream (attempt {retry_count}/{max_retries})...",
        "type": "stream_recovery"
    }
    return f"data: {json.dumps(data)}\n\n"


def should_attempt_stream_recovery(
    error: Exception,
    attempt_count: int,
    max_attempts: int = 3
) -> bool:
    """
    Determine whether to attempt stream recovery based on error and attempt count.
    
    Args:
        error: The exception that occurred
        attempt_count: Number of recovery attempts already made
        max_attempts: Maximum allowed recovery attempts
        
    Returns:
        True if recovery should be attempted
    """
    if attempt_count >= max_attempts:
        return False
        
    if isinstance(error, StreamingError):
        return error.is_recoverable and is_streaming_recoverable_error(error.original_error)
    
    return is_streaming_recoverable_error(error)
