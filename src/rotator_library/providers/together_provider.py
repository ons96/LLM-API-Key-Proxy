import os
import logging
from .openai_compatible_provider import OpenAICompatibleProvider

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class TogetherProvider(OpenAICompatibleProvider):
    """
    Together AI provider implementation.
    
    Together AI provides OpenAI-compatible API access to various open-source models
    including Llama, Mixtral, and others. Supports free tier access with rate limits.
    
    Environment Variables:
        TOGETHER_API_KEY: Required API key for authentication
        TOGETHER_API_BASE: Optional API base URL (defaults to https://api.together.xyz/v1)
    """
    
    def __init__(self):
        # Set default API base if not provided
        if not os.getenv("TOGETHER_API_BASE"):
            os.environ["TOGETHER_API_BASE"] = "https://api.together.xyz/v1"
        
        super().__init__("together")
        lib_logger.debug(f"TogetherProvider initialized with API base: {self.api_base}")
