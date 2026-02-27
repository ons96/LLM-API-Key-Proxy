#!/usr/bin/env python3
"""
Security Utilities Module

Provides secure hashing and verification for API keys using bcrypt.
Part of Phase 1.3 Security Hardening.
"""

import bcrypt
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def hash_api_key(api_key: str, rounds: int = 12) -> str:
    """
    Hash an API key using bcrypt.
    
    Args:
        api_key: The plaintext API key to hash
        rounds: Cost factor for bcrypt (default: 12)
        
    Returns:
        str: The bcrypt hash string
    """
    try:
        # Generate salt with specified rounds
        salt = bcrypt.gensalt(rounds=rounds)
        # Hash the API key
        hashed = bcrypt.hashpw(api_key.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to hash API key: {e}")
        raise


def verify_api_key(api_key: str, hashed_key: str) -> bool:
    """
    Verify an API key against a bcrypt hash.
    
    Args:
        api_key: The plaintext API key to verify
        hashed_key: The stored bcrypt hash
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        return bcrypt.checkpw(api_key.encode('utf-8'), hashed_key.encode('utf-8'))
    except Exception as e:
        logger.error(f"Error verifying API key: {e}")
        return False


def is_hashed_value(value: str) -> bool:
    """
    Check if a string appears to be a bcrypt hash.
    
    Args:
        value: The string to check
        
    Returns:
        bool: True if it looks like a bcrypt hash
    """
    if not value or not isinstance(value, str):
        return False
    return value.startswith('$2b$') or value.startswith('$2a$') or value.startswith('$2y$')


def mask_api_key(api_key: str, visible_chars: int = 4) -> str:
    """
    Mask an API key for logging/display purposes.
    
    Args:
        api_key: The API key to mask
        visible_chars: Number of characters to show at the end
        
    Returns:
        str: Masked key (e.g., '****...last4')
    """
    if not api_key or len(api_key) <= visible_chars:
        return '*' * len(api_key) if api_key else ''
    return '*' * (len(api_key) - visible_chars) + api_key[-visible_chars:]
