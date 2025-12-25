"""
G4F (g4f) Fallback Provider Implementation

This provider handles routing to multiple G4F-compatible endpoints:
- Main G4F API
- Groq-compatible endpoint
- Grok-compatible endpoint
- Gemini-compatible endpoint
- NVIDIA-compatible endpoint

G4F providers are used as fallback (Tier 3+) when primary providers fail.
"""
import os
import json
import time
import httpx
import litellm
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger('rotator_library')
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


# G4F endpoint configuration
G4F_ENDPOINTS = {
    "main": {
        "env_var": "G4F_MAIN_API_BASE",
        "default": None,
        "description": "Main G4F-compatible API endpoint"
    },
    "groq": {
        "env_var": "G4F_GROQ_API_BASE", 
        "default": None,
        "description": "Groq-compatible endpoint"
    },
    "grok": {
        "env_var": "G4F_GROK_API_BASE",
        "default": None,
        "description": "Grok-compatible endpoint"
    },
    "gemini": {
        "env_var": "G4F_GEMINI_API_BASE",
        "default": None,
        "description": "Gemini-compatible endpoint"
    },
    "nvidia": {
        "env_var": "G4F_NVIDIA_API_BASE",
        "default": None,
        "description": "NVIDIA-compatible endpoint"
    },
}


class G4FProvider(ProviderInterface):
    """
    G4F fallback provider implementation.
    
    Supports multiple G4F-compatible endpoints with automatic routing based on
    the model identifier. G4F providers are designed to be used as fallback
    when primary API providers are exhausted or rate-limited.
    
    Priority Tier: Default is Tier 5 (lowest priority, used as last resort)
    
    Attributes:
        provider_name: Always "g4f" for this provider
        tier_priorities: Maps tier names to priority levels (lower = higher priority)
        skip_cost_calculation: G4F responses typically don't include token usage
    """
    
    provider_name = "g4f"
    
    # G4F is a fallback provider - lowest priority by default
    tier_priorities: Dict[str, int] = {
        "standard": 5,  # Default tier for G4F
    }
    
    default_tier_priority: int = 5
    
    skip_cost_calculation: bool = True  # G4F typically doesn't provide token counts
    
    # G4F doesn't support embeddings
    def __init__(self):
        """Initialize the G4F provider with configured endpoints."""
        self.provider_name = "g4f"
        self._endpoints: Dict[str, str] = {}
        self._api_key: Optional[str] = None
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load endpoint configurations from environment variables."""
        self._api_key = os.getenv("G4F_API_KEY")
        
        for endpoint_name, config in G4F_ENDPOINTS.items():
            env_value: Optional[str] = os.getenv(config["env_var"])
            if env_value:
                self._endpoints[endpoint_name] = env_value.rstrip("/")
                lib_logger.debug(f"G4F endpoint '{endpoint_name}': {self._endpoints[endpoint_name]}")
        
        if not self._endpoints:
            lib_logger.warning("No G4F endpoints configured. Provider will not be usable.")
    
    def _get_endpoint_for_model(self, model: str) -> Optional[str]:
        """
        Determine which G4F endpoint to use based on the model name.
        
        Routing logic:
        - If model contains "groq", route to Groq-compatible endpoint
        - If model contains "grok", route to Grok-compatible endpoint
        - If model contains "gemini", route to Gemini-compatible endpoint
        - If model contains "nvidia" or "nemotron", route to NVIDIA-compatible endpoint
        - Otherwise, use the main G4F endpoint
        
        Args:
            model: The model name (with or without provider prefix)
            
        Returns:
            Endpoint URL or None if no suitable endpoint is configured
        """
        model_lower = model.lower()
        
        # Check for specific provider hints in the model name
        if "groq" in model_lower and "groq" in self._endpoints:
            return self._endpoints["groq"]
        elif "grok" in model_lower and "grok" in self._endpoints:
            return self._endpoints["grok"]
        elif "gemini" in model_lower and "gemini" in self._endpoints:
            return self._endpoints["gemini"]
        elif any(x in model_lower for x in ["nvidia", "nemotron"]) and "nvidia" in self._endpoints:
            return self._endpoints["nvidia"]
        elif "main" in self._endpoints:
            return self._endpoints["main"]
        elif self._endpoints:
            # Fall back to the first available endpoint
            return next(iter(self._endpoints.values()))
        
        return None
    
    def _get_auth_header(self) -> Dict[str, str]:
        """Get the authorization header for G4F requests."""
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers
    
    def _convert_to_g4f_format(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert OpenAI-style messages to G4F-compatible format.
        
        G4F APIs typically expect messages in a standard format similar to OpenAI.
        This method ensures compatibility by preserving the message structure.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Returns:
            Dict suitable for G4F API requests
        """
        return {
            "messages": messages,
        }
    
    def _parse_g4f_response(self, response: Dict[str, Any], model: str) -> litellm.ModelResponse:
        """
        Parse G4F API response and convert to OpenAI-compatible format.
        
        Args:
            response: Raw response from G4F API
            model: The model name used for the request
            
        Returns:
            litellm.ModelResponse in OpenAI-compatible format
        """
        try:
            # Extract content from G4F response
            # G4F responses typically follow OpenAI format
            choices = response.get("choices", [])
            
            if not choices:
                # Try alternative response formats
                if "error" in response:
                    error_msg = response["error"].get("message", "Unknown G4F error")
                    error_code = response.get("error", {}).get("code", 500)
                    raise litellm.APIError(
                        status_code=error_code,
                        message=error_msg,
                        llm_provider="g4f",
                        model=model
                    )
                raise litellm.APIError(
                    status_code=500,
                    message="Empty response from G4F",
                    llm_provider="g4f",
                    model=model
                )
            
            # Convert G4F response to LiteLLM format
            choice = choices[0]
            message = choice.get("message", {})
            
            # Build usage info if available
            usage = response.get("usage", {})
            if usage:
                # Ensure required usage fields
                usage_obj = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
            else:
                # G4F doesn't always provide usage - create empty usage
                usage_obj = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            
            # Build the OpenAI-compatible response
            response_obj = {
                "id": response.get("id", f"g4f-{int(time.time())}"),
                "object": "chat.completion",
                "created": response.get("created", int(time.time())),
                "model": model,
                "choices": [{
                    "index": choice.get("index", 0),
                    "message": {
                        "role": message.get("role", "assistant"),
                        "content": message.get("content", ""),
                    },
                    "finish_reason": choice.get("finish_reason", "stop"),
                }],
                "usage": usage_obj,
            }
            
            return litellm.ModelResponse(**response_obj)
            
        except litellm.APIError:
            raise
        except Exception as e:
            lib_logger.error(f"Error parsing G4F response: {e}")
            raise litellm.APIError(
                status_code=500,
                message=f"Failed to parse G4F response: {str(e)}",
                llm_provider="g4f",
                model=model
            )
    
    def _parse_chunk(self, chunk: Dict[str, Any], model: str) -> Dict[str, Any]:
        """
        Parse a streaming chunk from G4F and convert to OpenAI format.
        
        Args:
            chunk: Raw chunk from G4F streaming response
            model: The model name used for the request
            
        Returns:
            OpenAI-compatible chunk format
        """
        # Ensure chunk has required fields
        if not chunk.get("choices"):
            return {
                "id": f"g4f-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [],
            }
        
        choice = chunk["choices"][0]
        delta = choice.get("delta", {})
        
        return {
            "id": chunk.get("id", f"g4f-{int(time.time())}"),
            "object": "chat.completion.chunk",
            "created": chunk.get("created", int(time.time())),
            "model": model,
            "choices": [{
                "index": choice.get("index", 0),
                "delta": delta,
                "finish_reason": choice.get("finish_reason"),
            }],
        }
    
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetch available models from G4F endpoints.
        
        Note: G4F endpoints may not all support model listing.
        This returns a basic set of common models for each endpoint type.
        
        Args:
            api_key: Not used for G4F (uses G4F_API_KEY from env)
            client: httpx AsyncClient for making requests
            
        Returns:
            List of model name strings with g4f/ prefix
        """
        models = []
        
        # Add common models for each endpoint type
        endpoint_models = {
            "main": [
                "g4f/gpt-4",
                "g4f/gpt-4o",
                "g4f/claude-3-opus",
                "g4f/claude-3-sonnet",
            ],
            "groq": [
                "g4f/llama-3.1-70b",
                "g4f/llama-3.1-8b",
                "g4f/mixtral-8x7b",
            ],
            "grok": [
                "g4f/grok-2",
                "g4f/grok-2-latest",
            ],
            "gemini": [
                "g4f/gemini-pro",
                "g4f/gemini-1.5-pro",
                "g4f/gemini-1.5-flash",
            ],
            "nvidia": [
                "g4f/nemotron-70b",
                "g4f/llama-3.1-70b-instruct",
            ],
        }
        
        for endpoint_name, endpoint_models_list in endpoint_models.items():
            if endpoint_name in self._endpoints:
                models.extend(endpoint_models_list)
        
        if models:
            lib_logger.info(f"Loaded {len(models)} G4F models from configured endpoints")
        else:
            lib_logger.warning("No G4F endpoints configured, no models available")
        
        return models
    
    def has_custom_logic(self) -> bool:
        """
        G4F provider uses custom logic for requests.
        
        Returns:
            True, since we handle the request ourselves
        """
        return True
    
    async def acompletion(
        self,
        client: httpx.AsyncClient,
        **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handle chat completion requests via G4F.
        
        Args:
            client: httpx AsyncClient for making requests
            **kwargs: Standard LiteLLM completion kwargs
            
        Returns:
            ModelResponse or streaming generator
        """
        model = kwargs.get("model", "g4f/unknown")
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)
        
        # Extract model name without provider prefix for endpoint routing
        model_name = model.split("/")[-1] if "/" in model else model
        
        # Get the appropriate endpoint
        endpoint = self._get_endpoint_for_model(model)
        if not endpoint:
            raise litellm.APIError(
                status_code=503,
                message="No G4F endpoint configured. Set G4F_MAIN_API_BASE or other G4F_*_API_BASE variables.",
                llm_provider="g4f",
                model=model
            )
        
        lib_logger.info(f"G4F request: model={model}, endpoint={endpoint}, stream={stream}")
        
        # Build request payload
        payload = self._convert_to_g4f_format(messages)
        payload["model"] = model_name
        
        # Add optional parameters
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        
        headers = self._get_auth_header()
        headers["Content-Type"] = "application/json"
        
        try:
            if stream:
                return self._stream_completion(endpoint, payload, headers, model, client)
            else:
                return await self._non_stream_completion(endpoint, payload, headers, model, client)
        except httpx.HTTPStatusError as e:
            error_body = e.response.content.decode(errors="ignore")
            lib_logger.error(f"G4F HTTP error: {e.response.status_code} - {error_body[:200]}")
            
            raise litellm.APIError(
                status_code=e.response.status_code,
                message=f"G4F API error: {e.response.status_code}",
                llm_provider="g4f",
                model=model
            )
        except Exception as e:
            lib_logger.error(f"G4F request error: {e}")
            raise litellm.APIError(
                status_code=500,
                message=f"G4F request failed: {str(e)}",
                llm_provider="g4f",
                model=model
            )
    
    async def _non_stream_completion(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        model: str,
        client: httpx.AsyncClient
    ) -> litellm.ModelResponse:
        """
        Handle non-streaming completion requests.
        
        Args:
            endpoint: G4F API endpoint URL
            payload: Request payload
            headers: HTTP headers
            model: Model name for response
            client: httpx AsyncClient
            
        Returns:
            Parsed ModelResponse
        """
        api_endpoint = f"{endpoint}/chat/completions"
        
        response = await client.post(
            api_endpoint,
            json=payload,
            headers=headers,
            timeout=60.0
        )
        response.raise_for_status()
        
        response_data = response.json()
        return self._parse_g4f_response(response_data, model)
    
    async def _stream_completion(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        model: str,
        client: httpx.AsyncClient
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """
        Handle streaming completion requests.
        
        Args:
            endpoint: G4F API endpoint URL
            payload: Request payload
            headers: HTTP headers
            model: Model name for response
            client: httpx AsyncClient
            
        Yields:
            Parsed ModelResponse chunks
        """
        api_endpoint = f"{endpoint}/chat/completions"
        
        async with client.stream(
            "POST",
            api_endpoint,
            json=payload,
            headers=headers,
            timeout=120.0
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if not line:
                    continue
                
                # Handle SSE format (data: {...})
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        # Signal end of stream
                        yield litellm.ModelResponse(
                            id=f"g4f-{int(time.time())}",
                            object="chat.completion.chunk",
                            created=int(time.time()),
                            model=model,
                            choices=[]
                        )
                        break
                    
                    try:
                        chunk_data = json.loads(data)
                        parsed_chunk = self._parse_chunk(chunk_data, model)
                        yield litellm.ModelResponse(**parsed_chunk)
                    except json.JSONDecodeError:
                        lib_logger.warning(f"Failed to parse G4F stream chunk: {data[:100]}")
                else:
                    # Try to parse as raw JSON
                    try:
                        chunk_data = json.loads(line)
                        parsed_chunk = self._parse_chunk(chunk_data, model)
                        yield litellm.ModelResponse(**parsed_chunk)
                    except json.JSONDecodeError:
                        pass
    
    async def aembedding(self, client: httpx.AsyncClient, **kwargs) -> litellm.EmbeddingResponse:
        """
        G4F does not support embeddings.
        
        Raises:
            NotImplementedError: G4F providers don't support embeddings
        """
        raise NotImplementedError(
            "G4F providers do not support embeddings. "
            "Use a different provider for embedding requests."
        )
    
    def get_credential_tier_name(self, credential: str) -> Optional[str]:
        """
        G4F credentials are all in the standard tier.
        
        Args:
            credential: Not used (G4F doesn't have tiered credentials)
            
        Returns:
            "standard" for all G4F credentials
        """
        return "standard"
    
    def get_credential_priority(self, credential: str) -> Optional[int]:
        """
        G4F credentials always have priority 5 (lowest).
        
        Args:
            credential: Not used
            
        Returns:
            5 (G4F is always a fallback provider)
        """
        return 5
    
    def get_model_tier_requirement(self, model: str) -> Optional[int]:
        """
        G4F models don't have tier requirements.
        
        Args:
            model: Not used
            
        Returns:
            None (no tier restriction)
        """
        return None
    
    def get_rotation_mode(self, provider_name: str) -> str:
        """
        Get rotation mode for G4F provider.
        
        G4F uses balanced mode by default to distribute load across endpoints.
        
        Args:
            provider_name: The provider name
            
        Returns:
            "balanced" rotation mode
        """
        env_key = f"ROTATION_MODE_{provider_name.upper()}"
        return os.getenv(env_key, "balanced")
    
    @staticmethod
    def get_provider_priority(provider: str) -> int:
        """
        Get priority tier for a G4F provider.
        
        Args:
            provider: The provider name (e.g., "g4f", "g4f_groq")
            
        Returns:
            Priority tier (5 for G4F, configurable via PROVIDER_PRIORITY_* env vars)
        """
        env_key = f"PROVIDER_PRIORITY_{provider.upper()}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            try:
                return int(env_value)
            except ValueError:
                pass
        
        # Default priorities based on provider type
        g4f_variants = ["g4f", "g4f_main", "g4f_groq", "g4f_grok", "g4f_gemini", "g4f_nvidia"]
        if provider.lower() in g4f_variants:
            return 5  # G4F variants are lowest priority (fallback)
        
        return 10  # Unknown providers get lowest priority
    
    def parse_quota_error(
        self,
        error: Exception,
        error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse G4F quota/rate-limit errors.
        
        Args:
            error: The caught exception
            error_body: Optional raw response body string
            
        Returns:
            None (G4F doesn't provide structured quota info)
        """
        # G4F doesn't have a standardized error format
        # Return None to use generic error handling
        return None
