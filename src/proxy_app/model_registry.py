import yaml
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, config_path: str = "config/router_config.yaml"):
        self.config_path = config_path
        self.model_to_providers: Dict[str, List[str]] = {}
        self._build_registry()

    def _build_registry(self):
        """Builds a reverse index mapping model names to a list of providers."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            
            providers_config = config.get("providers", {})
            if isinstance(providers_config, dict):
                for provider, details in providers_config.items():
                    if not isinstance(details, dict):
                        continue
                    
                    free_tier_models = details.get("free_tier_models", [])
                    for model in free_tier_models:
                        # Map exact name
                        if model not in self.model_to_providers:
                            self.model_to_providers[model] = []
                        if provider not in self.model_to_providers[model]:
                            self.model_to_providers[model].append(provider)

                        # Map unprefixed name (e.g. meta-llama/llama-3.1-8b -> llama-3.1-8b)
                        if "/" in model:
                            _, unprefixed = model.split("/", 1)
                            if unprefixed not in self.model_to_providers:
                                self.model_to_providers[unprefixed] = []
                            if provider not in self.model_to_providers[unprefixed]:
                                self.model_to_providers[unprefixed].append(provider)

            logger.info(f"ModelRegistry initialized. Mapped {len(self.model_to_providers)} models.")
        except Exception as e:
            logger.error(f"Failed to build model registry: {e}")

    def get_providers(self, model_name: str) -> List[str]:
        """Get list of providers supporting the given model."""
        return self.model_to_providers.get(model_name, [])

