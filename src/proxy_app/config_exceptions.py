"""Configuration exceptions."""

class ConfigLoadError(Exception):
    """Raised when configuration fails to load."""
    pass

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass
