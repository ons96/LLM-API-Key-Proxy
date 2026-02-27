"""Settings and configuration management with rate limiting support."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitSettings:
    """Rate limiting configuration."""
    enabled: bool = True
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 60
    cleanup_interval: int = 300
    per_key_limits: Dict[str, Dict[str, int]] = field(default_factory=dict)
    anonymous_limits: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "rpm": 10,
        "rph": 100,
        "rpd": 1000
    })


@dataclass
class Settings:
    """Application settings."""
    rate_limiting: RateLimitSettings = field(default_factory=RateLimitSettings)
    # Add other settings fields as needed
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        """Create settings from dictionary."""
        rate_limit_data = data.get('rate_limiting', {})
        
        return cls(
            rate_limiting=RateLimitSettings(
                enabled=rate_limit_data.get('enabled', True),
                requests_per_minute=rate_limit_data.get('requests_per_minute', 60),
                requests_per_hour=rate_limit_data.get('requests_per_hour', 1000),
                requests_per_day=rate_limit_data.get('requests_per_day', 10000),
                burst_size=rate_limit_data.get('burst_size', 60),
                cleanup_interval=rate_limit_data.get('cleanup_interval', 300),
                per_key_limits=rate_limit_data.get('per_key_limits', {}),
                anonymous_limits=rate_limit_data.get('anonymous_limits', {
                    "enabled": True,
                    "rpm": 10,
                    "rph": 100,
                    "rpd": 1000
                })
            )
        )


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not path.exists():
        logger.warning(f"Config file not found: {path}")
        return {}
    
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load config {path}: {e}")
        return {}


def get_settings() -> Settings:
    """Load settings from configuration files."""
    # Try to load from router_config.yaml
    config_paths = [
        Path("config/router_config.yaml"),
        Path("/app/config/router_config.yaml"),
        Path(os.environ.get("ROUTER_CONFIG_PATH", "config/router_config.yaml"))
    ]
    
    config_data = {}
    for path in config_paths:
        if path.exists():
            config_data = load_yaml_config(path)
            logger.debug(f"Loaded config from {path}")
            break
    
    return Settings.from_dict(config_data)


# Global settings cache
_settings_cache: Optional[Settings] = None


def get_cached_settings() -> Settings:
    """Get cached settings or load new."""
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = get_settings()
    return _settings_cache


def reload_settings() -> Settings:
    """Force reload settings from disk."""
    global _settings_cache
    _settings_cache = get_settings()
    logger.info("Settings reloaded")
    return _settings_cache
