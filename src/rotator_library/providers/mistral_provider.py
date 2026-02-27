# src/rotator_library/providers/mistral_provider.py

import os
import logging
from typing import List, Dict, Any
import httpx
from .provider_interface import ProviderInterface
from ..model_definitions import ModelDefinitions

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class MistralProvider(ProviderInterface):
    """
    Mistral AI provider adapter.
    Implements OpenAI-compatible API for Mistral's LLM services.
    """

    skip_cost_calculation: bool = False

    def __init__(self):
        self.provider_name = "mistral"
        self.api_base = os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1")
        self.model_definitions = ModelDefinitions()

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetch available models from Mistral API.
        Combines static definitions with dynamic discovery.
        """
        models = []

        # Load static model definitions
        static_models = self.model_definitions.get_all_provider_models(
            self.provider_name
        )
        if static_models:
            models.extend(static_models)
            lib_logger.info(
                f"Loaded {len(static_models)} static models for {self.provider_name}"
            )

        # Dynamic discovery
        try:
            models_url = f"{self.api_base.rstrip('/')}/models"
            response = await client.get(
                models_url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0
            )
            response.raise_for_status()

            data = response.json()
            dynamic_models = [
                f"{self.provider_name}/{model['id']}"
                for model in data.get("data", [])
                if model["id"] not in [m.split("/")[-1] for m in static_models]
            ]

            if dynamic_models:
                models.extend(dynamic_models)
                lib_logger.debug(
                    f"Discovered {len(dynamic_models)} additional models for {self.provider_name}"
                )

        except Exception as e:
            lib_logger.debug(f"Dynamic model discovery failed for Mistral: {e}")

        return models

    def get_model_options(self, model_name: str) -> Dict[str, Any]:
        """
        Get model-specific options.
        """
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        return self.model_definitions.get_model_options(self.provider_name, model_name)

    def has_custom_logic(self) -> bool:
        """
        Uses standard OpenAI-compatible flow via litellm.
        """
        return False

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """
        Standard Bearer token authentication.
        """
        return {"Authorization": f"Bearer {credential_identifier}"}
