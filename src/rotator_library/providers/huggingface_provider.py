import httpx
import logging
from typing import List
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class HuggingFaceProvider(ProviderInterface):
    """
    Provider implementation for HuggingFace Inference API.
    
    HuggingFace offers free inference for many open-source models.
    API Base: https://api-inference.huggingface.co/models
    
    Free tier limits:
    - Queue-based inference (may wait if busy)
    - Rate limited per model
    - Some models require Pro subscription
    
    Popular free models:
    - Qwen/Qwen2.5-72B-Instruct
    - meta-llama/Llama-3.3-70B-Instruct
    - mistralai/Mistral-7B-Instruct-v0.3
    - HuggingFaceH4/zephyr-7b-beta
    """
    
    provider_name = "huggingface"
    provider_env_name = "huggingface"
    
    # Medium priority - free but queue-based
    default_tier_priority = 3
    
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Returns a static list of recommended HuggingFace models.
        
        HuggingFace doesn't have a simple "list all models" endpoint for inference,
        so we return a curated list of high-quality, free models.
        """
        # HuggingFace doesn't have a unified /v1/models endpoint for inference
        # We provide a curated list of recommended free models
        static_models = [
            # Large models (70B+)
            "huggingface/Qwen/Qwen2.5-72B-Instruct",
            "huggingface/meta-llama/Llama-3.3-70B-Instruct",
            # Medium models (7B-32B)
            "huggingface/mistralai/Mistral-7B-Instruct-v0.3",
            "huggingface/HuggingFaceH4/zephyr-7b-beta",
            "huggingface/microsoft/Phi-3-mini-4k-instruct",
            # Coding models
            "huggingface/Qwen/Qwen2.5-Coder-32B-Instruct",
            "huggingface/codellama/CodeLlama-34b-Instruct-hf",
        ]
        
        lib_logger.info(f"Returning {len(static_models)} curated HuggingFace models")
        return static_models
