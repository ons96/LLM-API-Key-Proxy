import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml


def load_logging_config(config_path: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    """
    Load logging configuration from YAML file.
    
    Args:
        config_path: Path to logging config YAML. If None, uses default location
                    at config/logging_config.yaml relative to project root.
        
    Returns:
        Dictionary containing logging configuration with 'log_levels' key.
    """
    if config_path is None:
        # Try to find config relative to this file (src/proxy_app/logging_config.py)
        # Goes up to project root, then into config/
        current_file = Path(__file__).resolve()
        config_path = current_file.parent.parent.parent / "config" / "logging_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        logging.warning(f"Logging config not found at {config_path}, using defaults")
        return _get_default_config()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config if config else _get_default_config()
    except Exception as e:
        logging.warning(f"Failed to load logging config: {e}, using defaults")
        return _get_default_config()


def _get_default_config() -> Dict[str, Any]:
    """Return default logging configuration."""
    return {
        "log_levels": {
            "root": "INFO",
            "loggers": {}
        }
    }


def setup_logging(
    config_path: Optional[Union[Path, str]] = None,
    default_level: Optional[str] = None,
    env_prefix: str = "LOG_LEVEL_",
    enable_console: bool = True
) -> None:
    """
    Setup logging configuration from YAML file and environment variables.
    
    This should be called once at application startup (e.g., in main.py)
    before any loggers are instantiated.
    
    Args:
        config_path: Path to logging config YAML. If None, uses default location.
        default_level: Override for root log level (e.g., 'DEBUG', 'INFO').
        env_prefix: Prefix for environment variable overrides (e.g., LOG_LEVEL_PROXY_APP).
                   Environment variables should be uppercase with double underscore 
                   for hierarchy: LOG_LEVEL_PROXY_APP__DETAILED_LOGGER=DEBUG
        enable_console: If True, ensures a basic console handler exists for root logger.
    """
    config = load_logging_config(config_path)
    log_levels = config.get("log_levels", {})
    
    # Setup basic console handler if none exists and requested
    if enable_console and not logging.getLogger().handlers:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(console_handler)
    
    # Set root logger level
    root_level = default_level or log_levels.get("root", "INFO")
    logging.getLogger().setLevel(getattr(logging, root_level.upper(), logging.INFO))
    
    # Configure specific loggers from YAML
    loggers_config = log_levels.get("loggers", {})
    for logger_name, level in loggers_config.items():
        _set_logger_level(logger_name, level)
    
    # Apply environment variable overrides
    # Format: LOG_LEVEL_<logger_path>=LEVEL
    # Example: LOG_LEVEL_ROTATOR_LIBRARY__FAILURE_LOGGER=DEBUG
    for env_key, env_value in os.environ.items():
        if env_key.startswith(env_prefix):
            # Convert LOG_LEVEL_PROXY_APP__DETAILED_LOGGER to proxy_app.detailed_logger
            logger_suffix = env_key[len(env_prefix):]
            logger_name = logger_suffix.lower().replace("__", ".")
            _set_logger_level(logger_name, env_value, source="environment")


def _set_logger_level(logger_name: str, level: str, source: str = "config") -> None:
    """
    Set the log level for a specific logger.
    
    Args:
        logger_name: Name of the logger (e.g., 'proxy_app.detailed_logger')
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        source: Source of the configuration for logging purposes
    """
    try:
        level_upper = level.upper()
        log_level = getattr(logging, level_upper)
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        # Only log the configuration change if we're configuring explicitly
        if source == "environment":
            logging.info(f"Log level for '{logger_name}' set to {level_upper} via {source}")
    except AttributeError:
        logging.warning(f"Invalid log level '{level}' for logger '{logger_name}'")
    except Exception as e:
        logging.warning(f"Failed to set log level for '{logger_name}': {e}")


def get_effective_level(logger_name: str) -> int:
    """
    Get the effective log level for a logger (inherited if not set explicitly).
    
    Args:
        logger_name: Name of the logger
        
    Returns:
        The effective logging level constant
    """
    return logging.getLogger(logger_name).getEffectiveLevel()


def is_debug_enabled(logger_name: str) -> bool:
    """
    Check if DEBUG level is enabled for a logger.
    
    Args:
        logger_name: Name of the logger
        
    Returns:
        True if DEBUG is enabled, False otherwise
    """
    return logging.getLogger(logger_name).isEnabledFor(logging.DEBUG)
