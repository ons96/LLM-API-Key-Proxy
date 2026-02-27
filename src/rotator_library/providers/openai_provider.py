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


class OpenAIProvider(ProviderInterface):
    """
    Provider implementation for the OpenAI API.
    Supports direct integration with OpenAI's paid API including
    cost calculation, organization management, and proper error handling.
    """
    
    # Enable cost calculation for OpenAI (paid provider with known pricing)
    skip_cost_calculation: bool = False
    
    def __init__(self):
        self.provider_name = "openai"
        self.api_base = "https://api.openai.com/v1"
        
        # Load organization and project IDs if available for enterprise features
        self.organization = os.getenv("OPENAI_ORGANIZATION")
        self.project = os.getenv("OPENAI_PROJECT")
        
        # Initialize model definitions loader for static config
        self.model_definitions = ModelDefinitions()

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the OpenAI API.
        Combines dynamic discovery with static model definitions for known pricing.
        """
        models = []
        
        # First, get static model definitions (includes pricing info)
        static_models = self.model_definitions.get_all_provider_models(
            self.provider_name
        )
        if static_models:
            models.extend(static_models)
            lib_logger.info(f"Loaded {len(static_models)} static OpenAI models")

        # Dynamic discovery for additional/newer models not in static config
        try:
            response = await client.get(
                f"{self.api_base}/models",
                headers=self._get_headers(api_key),
                timeout=30.0
            )
            response.raise_for_status()
            
            # Get existing model IDs for deduplication
            existing_ids = {m.split("/")[-1] for m in static_models}
            
            dynamic_models = []
            for model in response.json().get("data", []):
                model_id = model["id"]
                # Skip fine-tuned models (user-specific) and already known models
                if model_id not in existing_ids and not model_id.startswith("ft:"):
                    dynamic_models.append(f"{self.provider_name}/{model_id}")
            
            if dynamic_models:
                models.extend(dynamic_models)
                lib_logger.info(f"Discovered {len(dynamic_models)} additional OpenAI models dynamically")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                lib_logger.error("OpenAI API key invalid or expired during model discovery")
            elif e.response.status_code == 429:
                lib_logger.warning("OpenAI rate limit exceeded during model discovery")
            else:
                lib_logger.error(f"OpenAI API error {e.response.status_code} during model discovery: {e}")
        except httpx.RequestError as e:
            lib_logger.error(f"Network error connecting to OpenAI API: {e}")
        except Exception as e:
            lib_logger.error(f"Unexpected error fetching OpenAI models: {e}")

        return models

    def get_model_options(self, model_name: str) -> Dict[str, Any]:
        """
        Get options for a specific OpenAI model including pricing and capabilities.
        
        Args:
            model_name: Model name (with or without provider prefix)
            
        Returns:
            Dictionary of model options including cost per token if available
        """
        # Extract model name without provider prefix if present
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
            
        # Get from model definitions (includes pricing for OpenAI models)
        options = self.model_definitions.get_model_options(self.provider_name, model_name)
        
        # Ensure we return a dict
        if not options:
            options = {}
            
        # Infer context window from model name if not specified
        if "context_window" not in options:
            if "gpt-4-turbo" in model_name or "gpt-4-0125" in model_name or "gpt-4-1106" in model_name:
                options["context_window"] = 128000
            elif "gpt-4o-mini" in model_name:
                options["context_window"] = 128000
            elif "gpt-4o" in model_name:
                options["context_window"] = 128000
            elif "gpt-4" in model_name:
                if "32k" in model_name:
                    options["context_window"] = 32768
                else:
                    options["context_window"] = 8192
            elif "gpt-3.5-turbo" in model_name:
                if "0125" in model_name or "1106" in model_name:
                    options["context_window"] = 16385
                elif "16k" in model_name:
                    options["context_window"] = 16384
                else:
                    options["context_window"] = 4096
                    
        return options

    def has_custom_logic(self) -> bool:
        """
        Returns False to use standard litellm flow with OpenAI-specific configuration.
        Custom logic is handled via model definitions and options.
        """
        return False

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """
        Returns authentication headers for OpenAI API including organization
        and project headers if configured via environment variables.
        """
        return self._get_headers(credential_identifier)
    
    def _get_headers(self, api_key: str) -> Dict[str, str]:
        """
        Build headers for OpenAI API requests.
        Includes Organization and Project headers for enterprise features.
        """
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Add organization header if available (for enterprise/teams)
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
            
        # Add project header if available (newer OpenAI API feature)
        if self.project:
            headers["OpenAI-Project"] = self.project
            
        return headers

    def is_available(self, api_key: Optional[str] = None) -> bool:
        """
        Check if OpenAI provider is properly configured.
        Requires OPENAI_API_KEY environment variable or passed api_key.
        """
        if api_key:
            return True
        return bool(os.getenv("OPENAI_API_KEY"))
