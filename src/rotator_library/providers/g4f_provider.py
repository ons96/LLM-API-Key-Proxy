import os
import httpx
import logging
from typing import List, Dict, Any, Optional
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class G4FProvider(ProviderInterface):
    """
    Provider implementation for G4F (GPT4Free).
    Supports multiple endpoints and dynamic model discovery.
    """
    
    skip_cost_calculation: bool = True
    
    def __init__(self):
        self.provider_name = "g4f"
        lib_logger.info("Initializing G4F provider")
        
        # G4F can use various API bases or default to local/known endpoints
        self.api_base = os.getenv("G4F_API_BASE", "http://localhost:8080")
        self.default_tier_priority = 5  # G4F is fallback tier
        
        lib_logger.debug(f"G4F configured with API base: {self.api_base}, tier priority: {self.default_tier_priority}")
    
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches available models from G4F endpoint.
        """
        lib_logger.debug(f"Fetching G4F models from {self.api_base}")
        
        try:
            models_url = f"{self.api_base.rstrip('/')}/models"
            response = await client.get(models_url, timeout=10.0)
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            for model in data.get("data", []):
                model_id = model.get("id")
                if model_id:
                    models.append(f"g4f/{model_id}")
            
            lib_logger.info(f"Retrieved {len(models)} models from G4F")
            return models
            
        except Exception as e:
            lib_logger.error(f"Failed to fetch G4F models: {e}")
            # Return default fallback models if fetch fails
            default_models = [
                "g4f/gpt-3.5-turbo",
                "g4f/gpt-4",
                "g4f/claude-2"
            ]
            lib_logger.warning(f"Returning {len(default_models)} default G4F models due to fetch failure")
            return default_models
    
    def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """
        G4F typically doesn't require authentication, but support it if provided.
        """
        lib_logger.debug("Generating auth header for G4F (optional)")
        if credential_identifier:
            return {"Authorization": f"Bearer {credential_identifier}"}
        return {}
    
    def has_custom_logic(self) -> bool:
        return False
    
    def get_model_options(self, model_name: str) -> Dict[str, Any]:
        """
        G4F models typically don't have special options.
        """
        lib_logger.debug(f"Getting options for G4F model: {model_name}")
        return {
            "tier_priority": self.default_tier_priority,
            "requires_auth": False
        }
