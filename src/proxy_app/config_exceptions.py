"""
Configuration and security exceptions for the proxy application.
"""

class ConfigurationError(Exception):
    """Raised when there's an error in configuration loading or validation."""
    pass


class ConfigValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
    def __init__(self, message: str, field: str = None):
        self.field = field
        super().__init__(message)


class PayloadTooLargeError(Exception):
    """
    Raised when request payload exceeds configured size limits.
    Maps to HTTP 413 Payload Too Large.
    """
    def __init__(self, content_length: int, max_allowed: int, content_type: str = None):
        self.content_length = content_length
        self.max_allowed = max_allowed
        self.content_type = content_type
        message = (
            f"Request body size ({content_length} bytes) exceeds maximum allowed "
            f"size ({max_allowed} bytes)"
        )
        if content_type:
            message += f" for content type '{content_type}'"
        super().__init__(message)
