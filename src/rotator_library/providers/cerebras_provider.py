import httpx
import logging
from typing import List
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class CerebrasProvider(ProviderInterface):
    """
    Provider implementation for Cerebras Inference API.
    
    Cerebras offers extremely fast inference with a generous free tier.
    API Base: https://api.cerebras.ai/v1
    
    Free tier limits:
    - 1 million tokens per day
    - 64K context window
    - Rate limited
    
    Available models:
    - llama-3.1-8b (Llama 3.1 8B)
    - llama-3.3-70b (Llama 3.3 70B)
    - qwen-3-32b (Qwen 3 32B)
    - qwen-3-235b (Qwen 3 235B Instruct - preview)
    - gpt-oss-120b (GPT OSS 120B)
    - zai-glm-4.6 (ZAI GLM 4.6 - preview)
    """
    
    provider_name = "cerebras"
    provider_env_name = "cerebras"
    
    # High priority - very fast inference with good free tier
    default_tier_priority = 2
    
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from Cerebras API.
        """
        try:
            response = await client.get(
                "https://api.cerebras.ai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            
            if "data" in data:
                models = [f"cerebras/{model['id']}" for model in data.get("data", [])]
                if models:
                    lib_logger.info(f"Discovered {len(models)} models from Cerebras API")
                    return models
                    
        except httpx.RequestError as e:
            lib_logger.debug(f"Failed to fetch Cerebras models: {e}")
        except Exception as e:
            lib_logger.debug(f"Error parsing Cerebras models: {e}")
        
        # Fallback to known models
        static_models = [
            "cerebras/llama-3.1-8b",
            "cerebras/llama-3.3-70b",
            "cerebras/qwen-3-32b",
            "cerebras/qwen-3-235b",
            "cerebras/gpt-oss-120b",
            "cerebras/zai-glm-4.6",
        ]
        
        lib_logger.info(f"Using fallback Cerebras model list: {len(static_models)} models")
        return static_models
