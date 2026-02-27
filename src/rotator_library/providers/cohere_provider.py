```python
import os
import httpx
import logging
from typing import List, Dict, Any, Optional
from .provider_interface import ProviderInterface
from ..model_definitions import ModelDefinitions

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class CohereProvider(ProviderInterface):
    """
    Cohere API provider implementation.
    Supports Cohere's command models for free tier access.
    """
    
    skip_cost_calculation: bool = True  # Skip cost calculation for free tier
    
    # Cohere API base URL
    API_BASE = "https://api.cohere.com"
    
    # Default models available on free tier
    DEFAULT_MODELS = [
        "cohere/command",
        "cohere/command-light",
        "cohere/command-r",
        "cohere/command-r-plus",
    ]

    def __init__(self, provider_name: str = "cohere"):
        self.provider_name = provider_name
        self.api_base = os.getenv("COHERE_API_BASE") or self.API_BASE
        
        # Initialize model definitions loader
        self.model_definitions = ModelDefinitions()

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from Cohere API.
        Combines dynamic discovery with static model definitions.
        """
        models = []

        # First, try to get static model definitions
        static_models = self.model_definitions.get_all_provider_models(
            self.provider_name
        )
        if static_models:
            models.extend(static_models)
            lib_logger.info(
                f"Loaded {len(static_models)} static models for {self.provider_name}"
            )

        # Add default models if no static models found
        if not models:
            models.extend(self.DEFAULT_MODELS)
            lib_logger.info(
                f"Using {len(self.DEFAULT_MODELS)} default models for {self.provider_name}"
            )

        # Try dynamic discovery to get additional models
        try:
            models_url = f"{self.api_base.rstrip('/')}/v1/models"
            response = await client.get(
                models_url, 
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                dynamic_models = [
                    f"cohere/{model['id']}"
                    for model in data.get("models", [])
                    if model["id"] not in [m.split("/")[-1] for m in models]
                ]
                
                if dynamic_models:
                    models.extend(dynamic_models)
                    lib_logger.debug(
                        f"Discovered {len(dynamic_models)} additional models for {self.provider_name}"
                    )
            else:
                lib_logger.debug(
                    f"Dynamic model discovery returned status {response.status_code}"
                )

        except httpx.RequestError as e:
            lib_logger.debug(f"Dynamic model discovery failed: {e}")
        except Exception as e:
            lib_logger.debug(f"Dynamic model discovery error: {e}")

        return list(set(models))  # Remove duplicates

    def get_model_options(self, model_name: str) -> Dict[str, Any]:
        """
        Get options for a specific model from static definitions or environment variables.

        Args:
            model_name: Model name (without provider prefix)

        Returns:
            Dictionary of model options
        """
        # Extract model name without provider prefix if present
        if "/" in model_name:
            model_name = model_name.split("/")[-1]

        return self.model_definitions.get_model_options(self.provider_name, model_name)

    def has_custom_logic(self) -> bool:
        """
        Returns False since we want to use the standard litellm flow
        with just custom API base configuration.
        """
        return False

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """
        Returns the standard Bearer token header for API key authentication.
        """
        return {"Authorization": f"Bearer {credential_identifier}"}
    
    def get_api_base(self) -> str:
        """
        Returns the API base URL for Cohere.
        """
        return self.api_base
```
