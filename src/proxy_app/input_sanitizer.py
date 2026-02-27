"""
Input sanitization utilities for security hardening.
Prevents injection attacks, XSS, and handles malicious input patterns.
"""

import html
import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Security constants
MAX_INPUT_LENGTH = 10 * 1024 * 1024  # 10MB max total input
MAX_STRING_LENGTH = 1_000_000  # 1MB per individual string
MAX_JSON_DEPTH = 100
MAX_ARRAY_LENGTH = 10_000
ALLOWED_CONTROL_CHARS = frozenset('\t\n\r')

# Patterns for detecting injection attempts
SQLI_PATTERNS = [
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|EXEC|EXECUTE)\b)",
    r"(--|#|\/\*)",
    r"(\bOR\b\s+\b1\b\s*=\s*\b1\b)",
    r"(\bAND\b\s+\b1\b\s*=\s*\b1\b)",
]

PATH_TRAVERSAL_PATTERN = re.compile(r"(\.\.[/\\]|[/\\]\.\.)")
NULL_BYTE_PATTERN = re.compile(r"\x00")
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


class SanitizationError(ValueError):
    """Raised when input cannot be safely sanitized."""
    pass


def sanitize_string(
    value: str, 
    max_length: int = MAX_STRING_LENGTH,
    allow_markup: bool = False
) -> str:
    """
    Sanitize a string input by removing dangerous characters and limiting length.
    
    Args:
        value: Input string to sanitize
        max_length: Maximum allowed length
        allow_markup: If False, HTML/XML tags are escaped
        
    Returns:
        Sanitized string
        
    Raises:
        SanitizationError: If input contains critical security threats
    """
    if not isinstance(value, str):
        return value
    
    # Check for null bytes (indicates potential binary injection or C-style attacks)
    if NULL_BYTE_PATTERN.search(value):
        logger.warning("Null byte detected in input - potential injection attempt")
        value = value.replace('\x00', '')
    
    # Limit length before processing
    if len(value) > max_length:
        logger.warning(f"Input truncated from {len(value)} to {max_length} characters")
        value = value[:max_length]
    
    # Remove dangerous control characters while preserving whitespace
    sanitized = CONTROL_CHAR_PATTERN.sub('', value)
    
    # HTML escape to prevent XSS in rendered responses (unless explicitly allowed)
    if not allow_markup:
        sanitized = html.escape(sanitized)
    
    return sanitized


def sanitize_json(
    data: Any, 
    depth: int = 0, 
    max_depth: int = MAX_JSON_DEPTH,
    seen_objects: Optional[set] = None
) -> Any:
    """
    Recursively sanitize JSON data structures.
    
    Args:
        data: JSON data to sanitize
        depth: Current recursion depth
        max_depth: Maximum allowed recursion depth
        seen_objects: Set of object ids to detect circular references
        
    Returns:
        Sanitized data structure
        
    Raises:
        SanitizationError: If depth exceeded or circular reference detected
    """
    if seen_objects is None:
        seen_objects = set()
    
    # Check recursion depth
    if depth > max_depth:
        raise SanitizationError(f"JSON structure exceeds maximum depth of {max_depth}")
    
    # Handle circular references
    obj_id = id(data)
    if obj_id in seen_objects:
        raise SanitizationError("Circular reference detected in input")
    
    if isinstance(data, str):
        return sanitize_string(data)
    elif isinstance(data, dict):
        seen_objects.add(obj_id)
        try:
            result = {}
            for k, v in data.items():
                # Sanitize keys (prevent header injection via object keys)
                safe_key = sanitize_string(str(k), max_length=1000)
                result[safe_key] = sanitize_json(v, depth + 1, max_depth, seen_objects)
            return result
        finally:
            seen_objects.discard(obj_id)
    elif isinstance(data, list):
        seen_objects.add(obj_id)
        try:
            if len(data) > MAX_ARRAY_LENGTH:
                logger.warning(f"Array truncated from {len(data)} to {MAX_ARRAY_LENGTH}")
                data = data[:MAX_ARRAY_LENGTH]
            return [sanitize_json(item, depth + 1, max_depth, seen_objects) for item in data]
        finally:
            seen_objects.discard(obj_id)
    elif isinstance(data, (int, float, bool)) or data is None:
        return data
    else:
        # Convert unknown types to string and sanitize
        return sanitize_string(str(data))


def mask_api_key(key: Optional[str]) -> str:
    """
    Mask API key for safe logging.
    
    Args:
        key: API key to mask
        
    Returns:
        Masked key string
    """
    if not key:
        return "None"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


def sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """
    Sanitize HTTP headers to prevent header injection and log sensitive data safely.
    
    Args:
        headers: Dictionary of HTTP headers
        
    Returns:
        Sanitized headers dictionary
    """
    sensitive_headers = {
        'authorization', 'x-api-key', 'api-key', 'x-auth-token',
        'x-proxy-api-key', 'cookie', 'set-cookie', 'x-forwarded-api-key'
    }
    
    sanitized = {}
    for key, value in headers.items():
        key_lower = key.lower()
        
        # Mask sensitive values
        if key_lower in sensitive_headers:
            sanitized[key] = mask_api_key(str(value))
        else:
            # Prevent header injection by removing CRLF
            safe_value = str(value).replace('\r', '').replace('\n', '')[:1000]
            sanitized[key] = safe_value
    
    return sanitized


def validate_model_name(model: str) -> str:
    """
    Validate and sanitize model name to prevent path traversal and injection.
    
    Args:
        model: Model identifier string
        
    Returns:
        Sanitized model name
        
    Raises:
        SanitizationError: If model name contains forbidden patterns
    """
    if not model or not isinstance(model, str):
        raise SanitizationError("Model name must be a non-empty string")
    
    # Prevent path traversal
    if PATH_TRAVERSAL_PATTERN.search(model):
        raise SanitizationError("Model name contains path traversal characters")
    
    # Prevent command injection via backticks, semicolons, pipes
    dangerous_chars = set('`;|$&<>')
    if any(c in model for c in dangerous_chars):
        raise SanitizationError("Model name contains forbidden characters")
    
    # Limit length and sanitize
    return sanitize_string(model.strip(), max_length=1000)


def sanitize_chat_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize a chat message structure.
    
    Args:
        message: Message dictionary with role and content
        
    Returns:
        Sanitized message dictionary
    """
    if not isinstance(message, dict):
        raise SanitizationError("Message must be a dictionary")
    
    sanitized = {}
    
    # Sanitize role
    role = str(message.get('role', 'user'))
    if role not in {'system', 'user', 'assistant', 'tool', 'function'}:
        role = 'user'  # Default to user if invalid
    sanitized['role'] = role
    
    # Sanitize content
    content = message.get('content')
    if isinstance(content, str):
        sanitized['content'] = sanitize_string(content)
    elif isinstance(content, list):
        # Handle multimodal content
        sanitized['content'] = [
            sanitize_json(item) if isinstance(item, dict) else sanitize_string(str(item))
            for item in content[:100]  # Limit number of content parts
        ]
    elif content is None:
        sanitized['content'] = None
    else:
        sanitized['content'] = sanitize_string(str(content))
    
    # Sanitize name if present
    if 'name' in message:
        sanitized['name'] = sanitize_string(str(message['name']), max_length=256)
    
    return sanitized


def sanitize_embedding_input(input_data: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Sanitize embedding input data.
    
    Args:
        input_data: String or list of strings
        
    Returns:
        Sanitized input
    """
    if isinstance(input_data, str):
        return sanitize_string(input_data)
    elif isinstance(input_data, list):
        return [sanitize_string(s) for s in input_data[:MAX_ARRAY_LENGTH]]
    else:
        raise SanitizationError("Embedding input must be string or list of strings")
