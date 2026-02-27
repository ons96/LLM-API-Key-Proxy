#!/usr/bin/env python3
"""
Replicate Provider Module

Provider implementation for Replicate (replicate.com).
Supports running open-source models via Replicate's cloud API.
Uses OpenAI-compatible chat completions endpoint where available.
"""

import os
import httpx
import logging
from typing import List, Dict, Any
from .provider_interface import ProviderInterface
from ..model_definitions import ModelDefinitions

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class ReplicateProvider(ProviderInterface):
    """
    Provider implementation for Replicate.
    Uses Replicate's OpenAI-compatible API endpoint.
    """
    
    skip_cost_calculation: bool = True  # Skip cost calculation for free tier usage

    def __init__(self, provider_name: str = "replicate"):
        self.provider_name = provider_name
        # Get API base URL from environment or use default
        self.api_base = os.getenv(
            f"{provider_name.upper()}_API_BASE", 
            "https://api.replicate.com/v1"
        )
        
        # Initialize model definitions loader
        self.model_definitions = ModelDefinitions()

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches available models from Replicate API.
        Combines static definitions with dynamic discovery.
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

        # Try dynamic discovery from Replicate API
        try:
            models_url = f"{self.api_base.rstrip('/')}/models"
            response = await client.get(
                models_url, 
                headers={"Authorization": f"Bearer {api_key}"},
                params={"limit": "100"}  # Replicate supports pagination
            )
            response.raise_for_status()

            data = response.json()
            replicate_models = data.get("results", [])
            
            dynamic_models = [
                f"{self.provider_name}/{model['owner']}/{model['name']}"
                for model in replicate_models
                if f"{model['owner']}/{model['name']}" not in [
                    m.split("/", 2)[-1] if len(m.split("/")) > 2 else m.split("/")[-1] 
                    for m in static_models
                ]
            ]

            if dynamic_models:
                models.extend(dynamic_models)
                lib_logger.debug(
                    f"Discovered {len(dynamic_models)} additional models for {self.provider_name}"
                )

        except Exception as e:
            lib_logger.debug(f"Dynamic model discovery failed for {self.provider_name}: {e}")

        return models

    def get_model_options(self, model_name: str) -> Dict[str, Any]:
        """
        Get options for a specific model from static definitions.
        
        Args:
            model_name: Model name (with or without provider prefix)
            
        Returns:
            Dictionary of model options
        """
        # Extract model name without provider prefix if present
        if "/" in model_name:
            parts = model_name.split("/")
            if len(parts) > 2:
                # Format: replicate/owner/model
                model_name = f"{parts[-2]}/{parts[-1]}"
            else:
                model_name = parts[-1]

        return self.model_definitions.get_model_options(self.provider_name, model_name)

    def has_custom_logic(self) -> bool:
        """
        Returns False to use standard litellm flow with custom API base.
        """
        return False

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """
        Returns the Authorization header for Replicate.
        Uses Bearer token authentication for OpenAI-compatible API.
        
        Args:
            credential_identifier: The API key/token
            
        Returns:
            Dictionary containing Authorization header
        """
        return {"Authorization": f"Bearer {credential_identifier}"}
