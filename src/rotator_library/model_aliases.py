"""
Model Aliasing Support

Provides resolution of friendly alias names to canonical model identifiers.
Supports both global aliases (provider-agnostic) and provider-specific aliases.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml

logger = logging.getLogger(__name__)

CONFIG_ROOT = Path(__file__).resolve().parent.parent.parent / "config"
DEFAULT_ALIASES_PATH = CONFIG_ROOT / "aliases.yaml"


class ModelAliasManager:
    """
    Singleton manager for model aliases.
    
    Loads aliases from config/aliases.yaml and provides resolution
    of alias names to canonical model identifiers.
    Supports chained aliases (alias -> alias -> model) with cycle detection.
    """
    
    _instance: Optional["ModelAliasManager"] = None
    _initialized: bool = False
    
    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize alias manager (only runs once due to singleton)."""
        if ModelAliasManager._initialized:
            return
        ModelAliasManager._initialized = True
        
        self.config_path = config_path or str(DEFAULT_ALIASES_PATH)
        self.global_aliases: Dict[str, str] = {}
        self.provider_aliases: Dict[str, Dict[str, str]] = {}
        self.settings = {
            "max_recursion_depth": 5,
            "allow_chaining": True,
            "case_sensitive": False,
            "log_resolutions": True,
        }
        self._load_aliases()
    
    def _load_aliases(self):
        """Load aliases from YAML configuration."""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"Aliases config not found at {self.config_path}")
                return
            
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            self.global_aliases = config.get('global', {})
            self.provider_aliases = config.get('providers', {})
            self.settings.update(config.get('settings', {}))
            
            total_aliases = len(self.global_aliases) + sum(
                len(aliases) for aliases in self.provider_aliases.values()
            )
            
            if total_aliases > 0:
                logger.info(f"Loaded {total_aliases} model aliases from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load aliases from {self.config_path}: {e}")
            self.global_aliases = {}
            self.provider_aliases = {}
    
    def reload(self):
        """Reload aliases from disk."""
        self.global_aliases.clear()
        self.provider_aliases.clear()
        self._load_aliases()
    
    def _normalize_key(self, key: str) -> str:
        """Normalize alias key based on case sensitivity setting."""
        if not self.settings.get('case_sensitive', False):
            return key.lower()
        return key
    
    def resolve(self, model_id: str, provider: Optional[str] = None, depth: int = 0) -> str:
        """
        Resolve a model ID or alias to its canonical form.
        
        Args:
            model_id: The model ID or alias to resolve
            provider: Optional provider context for provider-specific aliases
            depth: Current recursion depth (for internal use)
            
        Returns:
            The canonical model ID, or the original if not found
        """
        if not model_id:
            return model_id
            
        max_depth = self.settings.get('max_recursion_depth', 5)
        if depth > max_depth:
            logger.warning(f"Max alias recursion depth ({max_depth}) reached for '{model_id}'")
            return model_id
        
        normalized_id = self._normalize_key(model_id)
        
        # Check provider-specific aliases first (higher priority than global)
        if provider:
            provider_normalized = self._normalize_key(provider)
            if provider_normalized in self.provider_aliases:
                provider_aliases = self.provider_aliases[provider_normalized]
                if normalized_id in provider_aliases:
                    resolved = provider_aliases[normalized_id]
                    if self.settings.get('log_resolutions', True):
                        logger.debug(f"Resolved alias '{model_id}' -> '{resolved}' (provider: {provider})")
                    
                    # Recursively resolve if chaining is enabled
                    if self.settings.get('allow_chaining', True):
                        return self.resolve(resolved, provider, depth + 1)
                    return resolved
        
        # Check global aliases
        if normalized_id in self.global_aliases:
            resolved = self.global_aliases[normalized_id]
            if self.settings.get('log_resolutions', True):
                logger.debug(f"Resolved alias '{model_id}' -> '{resolved}' (global)")
            
            if self.settings.get('allow_chaining', True):
                # Pass provider from resolved ID if it contains a provider prefix
                if '/' in resolved:
                    new_provider, new_model = resolved.split('/', 1)
                    return self.resolve(new_model, new_provider, depth + 1)
                return self.resolve(resolved, provider, depth + 1)
            return resolved
        
        # No alias found, return original
        return model_id
    
    def resolve_with_provider(self, model_id: str, default_provider: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """
        Resolve alias and extract provider/model components.
        
        Args:
            model_id: The model ID or alias to resolve
            default_provider: Fallback provider if resolution doesn't yield one
            
        Returns:
            Tuple of (resolved_model_id, provider)
        """
        resolved = self.resolve(model_id, default_provider)
        
        if '/' in resolved:
            parts = resolved.split('/', 1)
            return parts[1], parts[0]
        
        return resolved, default_provider
    
    def get_all_aliases(self) -> Dict[str, str]:
        """Get all global aliases."""
        return self.global_aliases.copy()
    
    def get_provider_aliases(self, provider: str) -> Dict[str, str]:
        """Get aliases for a specific provider."""
        return self.provider_aliases.get(self._normalize_key(provider), {}).copy()
    
    def is_alias(self, model_id: str, provider: Optional[str] = None) -> bool:
        """Check if a given model ID is an alias."""
        normalized = self._normalize_key(model_id)
        
        if provider:
            provider_normalized = self._normalize_key(provider)
            if provider_normalized in self.provider_aliases:
                if normalized in self.provider_aliases[provider_normalized]:
                    return True
        
        return normalized in self.global_aliases
    
    def get_canonical_form(self, model_id: str, provider: Optional[str] = None) -> Optional[str]:
        """
        Get the canonical form of an alias without recursive resolution.
        Returns None if the model_id is not an alias.
        """
        normalized = self._normalize_key(model_id)
        
        if provider:
            provider_normalized = self._normalize_key(provider)
            if provider_normalized in self.provider_aliases:
                if normalized in self.provider_aliases[provider_normalized]:
                    return self.provider_aliases[provider_normalized][normalized]
        
        return self.global_aliases.get(normalized)


# Module-level convenience functions
_alias_manager: Optional[ModelAliasManager] = None

def get_alias_manager() -> ModelAliasManager:
    """Get or create the singleton alias manager."""
    global _alias_manager
    if _alias_manager is None:
        _alias_manager = ModelAliasManager()
    return _alias_manager

def resolve_alias(model_id: str, provider: Optional[str] = None) -> str:
    """Resolve a model alias to its canonical form."""
    return get_alias_manager().resolve(model_id, provider)

def resolve_with_provider(model_id: str, default_provider: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Resolve alias and extract provider/model components."""
    return get_alias_manager().resolve_with_provider(model_id, default_provider)
