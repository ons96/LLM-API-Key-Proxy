import httpx
import logging
from typing import List
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class AgentRouterProvider(ProviderInterface):
    """
    Provider implementation for AgentRouter.org API.
    
    AgentRouter provides OpenAI-compatible API access to various models.
    API Base: https://agentrouter.org/v1
    
    Available models (as of user's account):
    - deepseek-v3.2, deepseek-v3, deepseek-r1
    - glm-4.6, glm-4.5
    
    Note: Claude models may not be available outside of Claude Code.
    """
    
    provider_name = "agentrouter"
    provider_env_name = "agentrouter"
    
    # Medium priority - has credits but is janky third-party
    default_tier_priority = 3
    
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from AgentRouter API.
        Falls back to known models if API doesn't support model listing.
        """
        try:
            response = await client.get(
                "https://agentrouter.org/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            
            if "data" in data:
                models = [f"agentrouter/{model['id']}" for model in data.get("data", [])]
                if models:
                    lib_logger.info(f"Discovered {len(models)} models from AgentRouter API")
                    return models
                    
        except httpx.RequestError as e:
            lib_logger.debug(f"Failed to fetch AgentRouter models: {e}")
        except Exception as e:
            lib_logger.debug(f"Error parsing AgentRouter models: {e}")
        
        # Fallback to known models based on user's account
        static_models = [
            "agentrouter/deepseek-v3.2",
            "agentrouter/deepseek-v3",
            "agentrouter/deepseek-r1",
            "agentrouter/glm-4.6",
            "agentrouter/glm-4.5",
            # Claude models may not be available outside Claude Code
            "agentrouter/claude-3.5-sonnet",
            "agentrouter/claude-3-opus",
            "agentrouter/gpt-5",
            "agentrouter/gpt-4o",
        ]
        
        lib_logger.info(f"Using fallback AgentRouter model list: {len(static_models)} models")
        return static_models
