import os
import httpx
import logging
import json
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from .provider_interface import ProviderInterface
import litellm

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class G4FProvider(ProviderInterface):
    """
    G4F (g4f) fallback provider implementation.
    
    G4F provides unified access to multiple free LLM providers as fallbacks
    when primary API keys are exhausted or rate-limited.
    """
    
    provider_name = "g4f"
    provider_env_name = "g4f"
    
    # G4F providers are typically free/fallback tier (lowest priority)
    default_tier_priority = 5
    
    def __init__(self):
        """
        Initialize G4F provider with configuration from environment variables.
        """
        # API key (optional for some G4F endpoints)
        self.api_key = os.getenv("G4F_API_KEY")
        
        # Base URLs for different G4F-compatible endpoints
        self.main_api_base = os.getenv("G4F_MAIN_API_BASE")
        self.groq_api_base = os.getenv("G4F_GROQ_API_BASE")
        self.grok_api_base = os.getenv("G4F_GROK_API_BASE")
        self.gemini_api_base = os.getenv("G4F_GEMINI_API_BASE")
        self.nvidia_api_base = os.getenv("G4F_NVIDIA_API_BASE")
        
        # Select appropriate base URL based on model type
        self.base_url = self._select_base_url()
        
        lib_logger.info(f"Initialized G4F provider with base URL: {self.base_url}")
    
    def _select_base_url(self) -> str:
        """
        Select the appropriate base URL for G4F API calls.
        Returns the main API base as default fallback.
        """
        return self.main_api_base or "https://g4f-api.example.com"
    
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetch available models from G4F.
        
        Priority:
        1. Try to get models from the G4F API endpoint (if configured)
        2. Fall back to dynamically discovering models from the g4f library itself
        """
        # First, try to get models from the API endpoint
        try:
            if self.base_url and "example.com" not in self.base_url:
                models_url = f"{self.base_url.rstrip('/')}/models"
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                response = await client.get(models_url, headers=headers, timeout=10.0)
                response.raise_for_status()
                
                models_data = response.json()
                models = []
                
                # Handle different response formats
                if "data" in models_data:
                    # OpenAI-compatible format
                    for model in models_data["data"]:
                        if isinstance(model, dict) and "id" in model:
                            models.append(f"g4f/{model['id']}")
                elif "models" in models_data:
                    # G4F-specific format
                    for model in models_data["models"]:
                        if isinstance(model, str):
                            models.append(f"g4f/{model}")
                        elif isinstance(model, dict) and "name" in model:
                            models.append(f"g4f/{model['name']}")
                
                if models:
                    lib_logger.info(f"Discovered {len(models)} models from G4F API")
                    return models
                    
        except httpx.RequestError as e:
            lib_logger.debug(f"Failed to fetch G4F models from API: {e}")
        except Exception as e:
            lib_logger.debug(f"Error parsing G4F models response: {e}")
        
        # Fallback: Dynamically discover all models from the g4f library
        try:
            from g4f.models import ModelUtils
            
            # Get all model names from g4f's model registry
            all_model_names = list(ModelUtils.convert.keys())
            
            # Filter to only include text-based chat models (exclude image/audio/video models)
            # by checking if they contain common non-chat keywords
            excluded_keywords = ['flux', 'dall-e', 'midjourney', 'stable-diffusion', 
                                 'sdxl', 'playground', 'imagen', 'whisper', 'tts',
                                 'suno', 'audio', 'music', 'video', 'image']
            
            chat_models = []
            for model_name in all_model_names:
                model_lower = model_name.lower()
                # Skip if it contains excluded keywords (image/audio generation models)
                if not any(kw in model_lower for kw in excluded_keywords):
                    chat_models.append(f"g4f/{model_name}")
            
            lib_logger.info(f"Discovered {len(chat_models)} chat models from g4f library (filtered from {len(all_model_names)} total)")
            return chat_models
            
        except ImportError as e:
            lib_logger.warning(f"g4f library not available for dynamic model discovery: {e}")
        except Exception as e:
            lib_logger.warning(f"Error discovering g4f models: {e}")
        
        # Final fallback: static list of common models
        static_models = [
            "g4f/gpt-3.5-turbo",
            "g4f/gpt-4",
            "g4f/gpt-4o",
            "g4f/gpt-4o-mini",
            "g4f/claude-3-sonnet",
            "g4f/claude-3-haiku",
            "g4f/claude-3.5-sonnet",
            "g4f/gemini-pro",
            "g4f/gemini-1.5-pro",
            "g4f/gemini-1.5-flash",
            "g4f/llama-3-8b",
            "g4f/llama-3-70b",
            "g4f/llama-3.1-8b",
            "g4f/llama-3.1-70b",
            "g4f/llama-3.1-405b",
            "g4f/llama-3.2-11b",
            "g4f/llama-3.2-90b",
            "g4f/mixtral-8x7b",
            "g4f/mistral-7b",
            "g4f/qwen-2-72b",
            "g4f/command-r-plus",
            "g4f/deepseek-coder",
            "g4f/phi-3-mini",
        ]
        
        lib_logger.info(f"Using fallback static model list: {len(static_models)} models")
        return static_models

    
    def has_custom_logic(self) -> bool:
        """
        G4F uses custom logic to handle multiple provider endpoints
        and G4F-specific error handling.
        """
        return True
    
    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handle chat completion requests via G4F.
        
        This method supports both streaming and non-streaming responses,
        with proper error handling for G4F-specific error codes.
        """
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)
        
        # Prepare request data
        request_data = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        
        # Add optional parameters
        if "temperature" in kwargs:
            request_data["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            request_data["max_tokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            request_data["top_p"] = kwargs["top_p"]
        if "frequency_penalty" in kwargs:
            request_data["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            request_data["presence_penalty"] = kwargs["presence_penalty"]
        if "stop" in kwargs:
            request_data["stop"] = kwargs["stop"]
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "G4F-Provider/1.0",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            # Make request to G4F API
            api_url = f"{self.base_url.rstrip('/')}/chat/completions"
            
            if stream:
                return self._handle_streaming_completion(client, api_url, headers, request_data)
            else:
                return await self._handle_non_streaming_completion(client, api_url, headers, request_data)
                
        except httpx.TimeoutException as e:
            lib_logger.error(f"G4F request timeout: {e}")
            raise Exception(f"G4F request timeout: {str(e)}")
        except httpx.HTTPStatusError as e:
            lib_logger.error(f"G4F HTTP error {e.response.status_code}: {e.response.text}")
            await self._handle_g4f_error(e)
        except httpx.RequestError as e:
            lib_logger.error(f"G4F request error: {e}")
            raise Exception(f"G4F request failed: {str(e)}")
        except Exception as e:
            lib_logger.error(f"G4F unexpected error: {e}")
            raise Exception(f"G4F completion failed: {str(e)}")
    
    async def _handle_non_streaming_completion(
        self,
        client: httpx.AsyncClient,
        api_url: str,
        headers: Dict[str, str],
        request_data: Dict[str, Any]
    ) -> litellm.ModelResponse:
        """
        Handle non-streaming chat completion.
        """
        response = await client.post(api_url, headers=headers, json=request_data)
        response.raise_for_status()
        
        response_data = response.json()
        
        # Convert G4F response to litellm format
        return self._convert_g4f_response(response_data)
    
    async def _handle_streaming_completion(
        self,
        client: httpx.AsyncClient,
        api_url: str,
        headers: Dict[str, str],
        request_data: Dict[str, Any]
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """
        Handle streaming chat completion.
        """
        async with client.stream(
            "POST",
            api_url,
            headers=headers,
            json=request_data
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if not line:
                    continue
                
                if line.startswith("data: "):
                    line_data = line[6:]  # Remove "data: " prefix
                    
                    if line_data.strip() == "[DONE]":
                        break
                    
                    try:
                        chunk_data = json.loads(line_data)
                        yield self._convert_g4f_chunk(chunk_data)
                    except json.JSONDecodeError:
                        lib_logger.warning(f"Failed to parse streaming chunk: {line_data}")
                        continue
    
    def _convert_g4f_response(self, response_data: Dict[str, Any]) -> litellm.ModelResponse:
        """
        Convert G4F API response to litellm ModelResponse format.
        """
        try:
            # Extract basic response fields
            model = response_data.get("model", "g4f/unknown")
            created = response_data.get("created", 0)
            
            # Convert choices
            choices = []
            for choice in response_data.get("choices", []):
                message = choice.get("message", {})
                choice_obj = litellm.Choices(
                    index=choice.get("index", 0),
                    message=litellm.Message(
                        role=message.get("role", "assistant"),
                        content=message.get("content", ""),
                    ),
                    finish_reason=choice.get("finish_reason"),
                )
                choices.append(choice_obj)
            
            # Convert usage if available
            usage_data = response_data.get("usage", {})
            usage = None
            if usage_data:
                usage = litellm.Usage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                )
            
            return litellm.ModelResponse(
                id=response_data.get("id", "g4f-completion"),
                object="chat.completion",
                created=created,
                model=model,
                choices=choices,
                usage=usage,
            )
            
        except Exception as e:
            lib_logger.error(f"Error converting G4F response: {e}")
            # Return a minimal valid response to prevent crashes
            return litellm.ModelResponse(
                id="g4f-error",
                object="chat.completion",
                created=0,
                model="g4f/error",
                choices=[litellm.Choices(
                    index=0,
                    message=litellm.Message(
                        role="assistant",
                        content=f"Error processing response: {str(e)}",
                    ),
                    finish_reason="error",
                )],
            )
    
    def _convert_g4f_chunk(self, chunk_data: Dict[str, Any]) -> litellm.ModelResponse:
        """
        Convert G4F streaming chunk to litellm ModelResponse format.
        """
        try:
            # Handle streaming chunk format
            delta = chunk_data.get("delta", {})
            choices = [litellm.Choices(
                index=chunk_data.get("index", 0),
                message=litellm.Message(
                    role=delta.get("role", "assistant"),
                    content=delta.get("content", ""),
                ),
                finish_reason=chunk_data.get("finish_reason"),
            )]
            
            return litellm.ModelResponse(
                id=chunk_data.get("id", "g4f-chunk"),
                object="chat.completion.chunk",
                created=chunk_data.get("created", 0),
                model=chunk_data.get("model", "g4f/streaming"),
                choices=choices,
            )
            
        except Exception as e:
            lib_logger.error(f"Error converting G4F chunk: {e}")
            # Return minimal chunk to prevent stream interruption
            return litellm.ModelResponse(
                id="g4f-chunk-error",
                object="chat.completion.chunk",
                created=0,
                model="g4f/error",
                choices=[litellm.Choices(
                    index=0,
                    message=litellm.Message(
                        role="assistant",
                        content="",
                    ),
                    finish_reason="error",
                )],
            )
    
    async def _handle_g4f_error(self, error: httpx.HTTPStatusError):
        """
        Handle G4F-specific error responses.
        """
        try:
            error_data = error.response.json()
            error_message = error_data.get("error", {}).get("message", "Unknown G4F error")
            error_code = error_data.get("error", {}).get("code", "unknown_error")
            
            lib_logger.error(f"G4F error {error_code}: {error_message}")
            
            # Handle specific G4F error codes
            if error.response.status_code == 429:
                raise Exception(f"G4F rate limit exceeded: {error_message}")
            elif error.response.status_code == 401:
                raise Exception(f"G4F authentication failed: {error_message}")
            elif error.response.status_code == 503:
                raise Exception(f"G4F service unavailable: {error_message}")
            else:
                raise Exception(f"G4F API error ({error.response.status_code}): {error_message}")
                
        except (json.JSONDecodeError, KeyError):
            # Fallback error handling if JSON parsing fails
            raise Exception(f"G4F API error ({error.response.status_code}): {error.response.text}")
    
    async def aembedding(
        self, client: httpx.AsyncClient, **kwargs
    ) -> litellm.EmbeddingResponse:
        """
        G4F providers typically don't support embeddings.
        Return a response indicating embeddings are not supported.
        """
        lib_logger.info("G4F embeddings not supported - returning empty response")
        
        # Return a minimal embedding response to prevent crashes
        # In practice, the caller should handle this case appropriately
        return litellm.EmbeddingResponse(
            id="g4f-embeddings-unsupported",
            object="list",
            data=[],  # Empty data array
            model="g4f/embeddings-unsupported",
        )
    
    def parse_quota_error(
        self, error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse G4F-specific quota/rate-limit errors.
        """
        error_message = str(error).lower()
        
        if "rate limit" in error_message or "429" in error_message:
            return {
                "retry_after": 60,  # Default 60 seconds for G4F
                "reason": "RATE_LIMITED",
                "reset_timestamp": None,
                "quota_reset_timestamp": None,
            }
        elif "quota" in error_message or "exhausted" in error_message:
            return {
                "retry_after": 300,  # Default 5 minutes for quota exhaustion
                "reason": "QUOTA_EXHAUSTED", 
                "reset_timestamp": None,
                "quota_reset_timestamp": None,
            }
        
        return None
    
    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """
        Returns the Authorization header for G4F API key authentication.
        """
        return {"Authorization": f"Bearer {credential_identifier}"}
    
    def get_credential_tier_name(self, credential: str) -> Optional[str]:
        """
        G4F credentials are typically free-tier.
        """
        return "free-tier"