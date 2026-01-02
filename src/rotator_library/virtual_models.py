"""
Virtual model selection system for LLM-API-Key-Proxy.

This module provides intelligent model routing based on task types.
Users can request virtual models like "best/coding-fast" which automatically
routes to the best available provider for that task type.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)


@dataclass
class VirtualModelConfig:
    """Configuration for a virtual model type."""
    name: str
    description: str
    default_temperature: float
    default_max_tokens: int
    provider_priority: List[str]
    model_mappings: Dict[str, List[str]]  # provider -> list of model names


class VirtualModelRouter:
    """Routes virtual model requests to actual providers."""
    
    def __init__(self):
        self.virtual_models = self._load_virtual_models()
        self.model_temperatures = self._load_model_temperatures()
    
    def _load_virtual_models(self) -> Dict[str, VirtualModelConfig]:
        """Load virtual model configurations from environment."""
        
        # Default configurations - can be overridden by env vars
        default_configs = {
            "best/coding-fast": VirtualModelConfig(
                name="best/coding-fast",
                description="Fast coding assistant (Groq, Grok, G4F)",
                default_temperature=0.3,
                default_max_tokens=4096,
                provider_priority=["groq", "grok", "g4f"],
                model_mappings={
                    "groq": ["llama-3.3-70b", "mixtral-8x7b", "llama-3.1-8b"],
                    "grok": ["grok-beta", "grok-1"],
                    "g4f": ["gpt-4", "gpt-3.5-turbo"]
                }
            ),
            "best/coding-smart": VirtualModelConfig(
                name="best/coding-smart",
                description="Best coding reasoning (Claude, Gemini, Qwen)",
                default_temperature=0.5,
                default_max_tokens=8192,
                provider_priority=["qwen", "gemini", "openrouter"],
                model_mappings={
                    "qwen": ["qwen-32b", "qwen-72b", "qwen-max"],
                    "gemini": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
                    "openrouter": ["anthropic/claude-3.5-sonnet", "openai/gpt-4"]
                }
            ),
            "best/chat-fast": VirtualModelConfig(
                name="best/chat-fast",
                description="Fast chat responses (Groq, Grok, G4F)",
                default_temperature=0.7,
                default_max_tokens=2048,
                provider_priority=["groq", "grok", "g4f"],
                model_mappings={
                    "groq": ["llama-3.3-70b", "mixtral-8x7b", "llama-3.1-8b"],
                    "grok": ["grok-beta", "grok-1"],
                    "g4f": ["gpt-4", "gpt-3.5-turbo"]
                }
            ),
            "best/chat-smart": VirtualModelConfig(
                name="best/chat-smart",
                description="Smartest chat model (Claude, Gemini, OpenRouter)",
                default_temperature=0.8,
                default_max_tokens=4096,
                provider_priority=["claude", "gemini", "openrouter"],
                model_mappings={
                    "claude": ["claude-3.5-sonnet", "claude-3-opus", "claude-3-sonnet"],
                    "gemini": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
                    "openrouter": ["anthropic/claude-3.5-sonnet", "openai/gpt-4-turbo"]
                }
            ),
            "best/reasoning": VirtualModelConfig(
                name="best/reasoning",
                description="Deep reasoning/analysis (Claude, Qwen, OpenRouter)",
                default_temperature=0.4,
                default_max_tokens=8192,
                provider_priority=["qwen", "claude", "openrouter"],
                model_mappings={
                    "qwen": ["qwen-32b", "qwen-72b", "qwen-max"],
                    "claude": ["claude-3.5-sonnet", "claude-3-opus"],
                    "openrouter": ["anthropic/claude-3.5-sonnet", "openai/gpt-4"]
                }
            ),
            "best/generic": VirtualModelConfig(
                name="best/generic",
                description="General purpose (any available free model)",
                default_temperature=0.7,
                default_max_tokens=2048,
                provider_priority=["groq", "gemini", "grok", "g4f"],
                model_mappings={
                    "groq": ["llama-3.3-70b", "mixtral-8x7b"],
                    "gemini": ["gemini-2.5-flash", "gemini-2.0-flash"],
                    "grok": ["grok-beta"],
                    "g4f": ["gpt-3.5-turbo"]
                }
            )
        }
        
        # Override with environment variables if provided
        configs = {}
        for model_name, default_config in default_configs.items():
            env_key = f"{model_name.replace('/', '_').upper()}_PROVIDERS"
            env_value = os.getenv(env_key)
            
            if env_value:
                # Parse comma-separated provider list
                providers = [p.strip() for p in env_value.split(",") if p.strip()]
                
                # Create new config with custom provider priority
                config = VirtualModelConfig(
                    name=model_name,
                    description=default_config.description,
                    default_temperature=default_config.default_temperature,
                    default_max_tokens=default_config.default_max_tokens,
                    provider_priority=providers,
                    model_mappings=default_config.model_mappings
                )
                configs[model_name] = config
                logger.info(f"Custom provider priority loaded for {model_name}: {providers}")
            else:
                configs[model_name] = default_config
        
        return configs
    
    def _load_model_temperatures(self) -> Dict[str, float]:
        """Load temperature settings from environment."""
        
        return {
            "best/coding-fast": float(os.getenv("CODING_FAST_TEMPERATURE", "0.3")),
            "best/coding-smart": float(os.getenv("CODING_SMART_TEMPERATURE", "0.5")),
            "best/chat-fast": float(os.getenv("CHAT_FAST_TEMPERATURE", "0.7")),
            "best/chat-smart": float(os.getenv("CHAT_SMART_TEMPERATURE", "0.8")),
            "best/reasoning": float(os.getenv("REASONING_TEMPERATURE", "0.4")),
            "best/generic": float(os.getenv("GENERIC_TEMPERATURE", "0.7"))
        }
    
    def is_virtual_model(self, model: str) -> bool:
        """Check if a model identifier is a virtual model."""
        return model.startswith("best/")
    
    def get_virtual_model_config(self, model: str) -> Optional[VirtualModelConfig]:
        """Get configuration for a virtual model."""
        return self.virtual_models.get(model)
    
    def get_all_virtual_models(self) -> List[Dict[str, Any]]:
        """Get list of all available virtual models."""
        models = []
        
        for config in self.virtual_models.values():
            models.append({
                "id": config.name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "Mirro-Proxy-Virtual",
                "description": config.description,
                "mode": "chat",
                "supported_modalities": ["text"],
                "supported_output_modalities": ["text"],
                "virtual": True,
                "provider_priority": config.provider_priority,
                "default_temperature": config.default_temperature
            })
        
        return models
    
    def resolve_virtual_model(
        self, 
        model: str, 
        available_providers: List[str]
    ) -> Tuple[str, float, int]:
        """
        Resolve a virtual model to an actual provider/model.
        
        Args:
            model: Virtual model name (e.g., "best/coding-fast")
            available_providers: List of currently available providers
            
        Returns:
            Tuple of (actual_model_name, temperature, max_tokens)
            
        Raises:
            ValueError: If virtual model is not found or no providers available
        """
        
        if not self.is_virtual_model(model):
            raise ValueError(f"Not a virtual model: {model}")
        
        config = self.get_virtual_model_config(model)
        if not config:
            raise ValueError(f"Unknown virtual model: {model}")
        
        # Find the first available provider in priority order
        selected_provider = None
        selected_model = None
        
        for provider in config.provider_priority:
            if provider in available_providers:
                # Get available models for this provider
                if provider in config.model_mappings:
                    # Pick a random model from this provider's list
                    available_models = config.model_mappings[provider]
                    if available_models:
                        selected_provider = provider
                        selected_model = random.choice(available_models)
                        break
        
        if not selected_provider or not selected_model:
            raise ValueError(f"No available providers for virtual model: {model}")
        
        # Build actual model identifier
        actual_model = f"{selected_provider}/{selected_model}"
        
        # Get temperature (check for model-specific override first)
        temperature = self.model_temperatures.get(
            model, 
            config.default_temperature
        )
        
        max_tokens = config.default_max_tokens
        
        logger.info(
            f"Virtual model '{model}' resolved to '{actual_model}' "
            f"(temp={temperature}, max_tokens={max_tokens})"
        )
        
        return actual_model, temperature, max_tokens


# Global singleton instance
_virtual_model_router: Optional[VirtualModelRouter] = None


def get_virtual_model_router() -> VirtualModelRouter:
    """Get the global virtual model router instance."""
    global _virtual_model_router
    if _virtual_model_router is None:
        _virtual_model_router = VirtualModelRouter()
    return _virtual_model_router
