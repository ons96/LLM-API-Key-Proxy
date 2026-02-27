```python
"""
Router Wrapper - Manages routing logic for virtual and real models
"""
import os
import yaml
import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from ..rotator_library import RotatingClient
from ..rotator_library.credential_manager import CredentialManager

logger = logging.getLogger(__name__)

# Global router instance
_router_instance = None
_virtual_models_cache = None


def initialize_router():
    """Initialize the router with configuration."""
    global _router_instance, _virtual_models_cache
    
    # Load virtual models config
    config_path = Path(__file__).parent.parent.parent / "config" / "virtual_models.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            _virtual_models_cache = config.get('virtual_models', {})
    
    # Initialize rotating client
    credential_manager = CredentialManager()
    _router_instance = RotatingClient(credential_manager=credential_manager)
    
    logger.info(f"Router initialized with {len(_virtual_models_cache)} virtual models")


def get_router() -> 'RotatingClient':
    """Get the router instance."""
    if _router_instance is None:
        initialize_router()
    return _router_instance


def get_virtual_models() -> Dict[str, Any]:
    """Get all virtual models configuration."""
    global _virtual_models_cache
    if _virtual_models_cache is None:
        initialize_router()
    return _virtual_models_cache


def get_virtual_model_config(model_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a specific virtual model."""
    virtual_models = get_virtual_models()
    return virtual_models.get(model_name)


def resolve_virtual_model(virtual_model_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve a virtual model to an actual provider model.
    
    Returns:
        Tuple of (actual_model_name, provider_name) or (None, None) if unavailable
    """
    config = get_virtual_model_config(virtual_model_name)
    if not config:
        return None, None
    
    # Get preferred provider
    preferred_provider = config.get('preferred_provider')
    model_mapping = config.get('model_mapping', {})
    
    if preferred_provider and preferred_provider in model_mapping:
        actual_model = model_mapping[preferred_provider]
        return actual_model, preferred_provider
    
    # Fallback to first available
    if model_mapping:
        first_provider = list(model_mapping.keys())[0]
        actual_model = model_mapping[first_provider]
        return actual_model, first_provider
    
    return None, None
```
