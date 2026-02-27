#!/usr/bin/env python3
"""
Anthropic Provider Module

Implementation of the ProviderInterface for Anthropic's Claude API.
Handles authentication, model discovery, and configuration for Anthropic's paid API.
"""

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


class AnthropicProvider(ProviderInterface):
    """
    Provider implementation for Anthropic API.
    Supports Claude models and Anthropic-specific authentication (x-api-key).
    """
    
    skip_cost_calculation: bool = False  # Enable cost tracking for paid provider

    def __init__(self, provider_name: str = "anthropic"):
        self.provider_name = provider_name
        # Get API base URL from environment, default to Anthropic's official API
        self.api_base = os.getenv(
            f"{provider_name.upper()}_API_BASE", 
            "https://api.anthropic.com/v1"
        )
        
        if not self.api_base:
            raise ValueError(
                f"Environment variable {provider_name.upper()}_API_BASE is required for Anthropic provider"
            )

        # Initialize model definitions loader
        self.model_definitions = ModelDefinitions()

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the Anthropic API.
        Combines static model definitions with dynamic discovery.
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

        # Then, try dynamic discovery to get additional models
        try:
            models_url = f"{self.api_base.rstrip('/')}/models"
            response = await client.get(
                models_url, 
                headers=await self.get_auth_header(api_key),
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

        except httpx.RequestError as e:
            lib_logger.warning(f"Dynamic model discovery failed for {self.provider_name}: {e}")
        except Exception as e:
            lib_logger.warning(f"Unexpected error during model discovery for {self.provider_name}: {e}")

        return models

    def get_model_options(self, model_name: str) -> Dict[str, Any]:
        """
        Get options for a specific model from static definitions or defaults.

        Args:
            model_name: Model name (with or without provider prefix)

        Returns:
            Dictionary of model options
        """
        # Extract model name without provider prefix if present
        if "/" in model_name:
            model_name = model_name.split("/")[-1]

        return self.model_definitions.get_model_options(self.provider_name, model_name)

    def has_custom_logic(self) -> bool:
        """
        Returns False to use the standard litellm routing flow.
        Anthropic is handled via standard OpenAI-compatible adapter in litellm.
        """
        return False

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """
        Returns Anthropic-specific authentication headers.
        
        Anthropic uses x-api-key header instead of standard Bearer tokens,
        and requires anthropic-version header.
        """
        return {
            "x-api-key": credential_identifier,
            "anthropic-version": "2023-06-01"
        }
