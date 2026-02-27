import os
from pathlib import Path
from typing import Optional

def get_logs_dir() -> Path:
    """
    Get the logs directory path, creating it if it doesn't exist.

    The logs directory is located at:
    - If ROTATOR_LOGS_DIR environment variable is set, use that
    - Otherwise, use a 'logs' directory in the current working directory

    Returns:
        Path to the logs directory
    """
    logs_dir = os.getenv("ROTATOR_LOGS_DIR", "logs")
    path = Path(logs_dir).resolve()

    # Create the directory if it doesn't exist
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Cannot create logs directory at {path}: {e}")

    return path


def get_cache_dir() -> Path:
    """
    Get the cache directory path, creating it if it doesn't exist.

    The cache directory is located at:
    - If ROTATOR_CACHE_DIR environment variable is set, use that
    - Otherwise, use a 'cache' directory in the current working directory

    Returns:
        Path to the cache directory
    """
    cache_dir = os.getenv("ROTATOR_CACHE_DIR", "cache")
    path = Path(cache_dir).resolve()

    # Create the directory if it doesn't exist
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Cannot create cache directory at {path}: {e}")

    return path


def get_temp_dir() -> Path:
    """
    Get the temp directory path, creating it if it doesn't exist.

    The temp directory is located at:
    - If ROTATOR_TEMP_DIR environment variable is set, use that
    - Otherwise, use a 'temp' directory in the current working directory

    Returns:
        Path to the temp directory
    """
    temp_dir = os.getenv("ROTATOR_TEMP_DIR", "temp")
    path = Path(temp_dir).resolve()

    # Create the directory if it doesn't exist
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Cannot create temp directory at {path}: {e}")

    return path


def get_config_dir() -> Path:
    """
    Get the config directory path, creating it if it doesn't exist.

    The config directory is located at:
    - If ROTATOR_CONFIG_DIR environment variable is set, use that
    - Otherwise, use a 'config' directory in the current working directory

    Returns:
        Path to the config directory
    """
    config_dir = os.getenv("ROTATOR_CONFIG_DIR", "config")
    path = Path(config_dir).resolve()

    # Create the directory if it doesn't exist
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Cannot create config directory at {path}: {e}")

    return path
