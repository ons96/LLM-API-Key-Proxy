"""
Replicate Provider Implementation - Phase 5.1

Supports Replicate's free tier models via their OpenAI-compatible chat completions API.
Replicate offers free token-limited access to various open-source models including
Llama, Mistral, and other popular OSS models.

Authentication: REPLICATE_API_TOKEN environment variable.
API Format: OpenAI-compatible /v1/models/{owner}/{model}/chat/completions
"""

import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx

from .antigravity_auth_base import BaseAuthProvider


class ReplicateProvider(BaseAuthProvider):
    """
    Provider for Replicate's free tier LLM API.
    
    Replicate hosts open-source models with a free tier (typically 100-500 requests/day
    depending on the model). This provider implements the OpenAI-compatible chat
    completions interface for supported models.
    """
    
    provider_name = "replicate"
    default_base_url = "https://api.replicate.com/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        **kwargs
    ):
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)
        self.api_key = api_key or os.getenv("REPLICATE_API_TOKEN") or os.getenv("REPLICATE_API_KEY")
        if not self.api_key:
            raise ValueError("Replicate API token required. Set REPLICATE_API_TOKEN.")
        
        self.client = httpx.AsyncClient(
            base_url=base_url or self.default_base_url,
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json",
                "Prefer": "wait",  # Synchronous response for non-streaming
            },
            timeout=timeout,
        )
    
    def _format_model_path(self, model_id: str) -> str:
        """
        Ensure model ID is in owner/model format required by Replicate.
        Some configs may use shorthand names that need expansion.
        """
        if "/" not in model_id:
            # Common free tier model mappings for Phase 5.1
            model_map = {
                "llama-3.1-8b": "meta/meta-llama-3.1-8b-instruct",
                "llama-3.1-70b": "meta/meta-llama-3.1-70b-instruct",
                "llama-3.1-405b": "meta/meta-llama-3.1-405b-instruct",
                "mistral-7b": "mistralai/mistral-7b-instruct-v0.3",
                "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct-v0.1",
                "gemma-2-9b": "google/gemma-2-9b-it",
                "gemma-2-27b": "google/gemma-2-27b-it",
                "deepseek-coder-33b": "deepseek-ai/deepseek-coder-33b-instruct",
            }
            if model_id.lower() in model_map:
                return model_map[model_id.lower()]
        return model_id
    
    async def chat_completions(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """
        Create chat completion via Replicate's OpenAI-compatible endpoint.
        
        Endpoint: POST /v1/models/{owner}/{model}/chat/completions
        """
        model_path = self._format_model_path(model)
        
        payload = {
            "model": model_path,
            "messages": messages,
            "stream": stream,
        }
        
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
            
        # Replicate-specific parameters
        if "repetition_penalty" in kwargs:
            payload["repetition_penalty"] = kwargs["repetition_penalty"]
        if "seed" in kwargs:
            payload["seed"] = kwargs["seed"]
        
        endpoint = f"models/{model_path}/chat/completions"
        
        try:
            if stream:
                # Streaming request
                async with self.client.stream(
                    "POST", 
                    endpoint, 
                    json=payload,
                    headers={"Accept": "text/event-stream"}
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                yield chunk
                            except json.JSONDecodeError:
                                continue
            else:
                # Non-streaming request
                response = await self.client.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise Exception(f"Replicate rate limit exceeded: {e.response.text}")
            elif e.response.status_code == 401:
                raise Exception(f"Replicate authentication failed: {e.response.text}")
            elif e.response.status_code == 422:
                raise Exception(f"Replicate validation error (model may not support chat): {e.response.text}")
            else:
                raise Exception(f"Replicate API error {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            raise Exception(f"Network error connecting to Replicate: {e}")
    
    async def validate_model(self, model_id: str) -> bool:
        """
        Validate that a model exists and supports chat completions on Replicate.
        """
        model_path = self._format_model_path(model_id)
        try:
            response = await self.client.get(f"models/{model_path}")
            if response.status_code == 200:
                data = response.json()
                # Check if model supports chat completions
                # Replicate models with chat capability typically have specific fields
                latest_version = data.get("latest_version", {})
                # Check openapi_schema for chat completion support
                schema = latest_version.get("openapi_schema", {})
                paths = schema.get("paths", {})
                chat_endpoint = f"/v1/models/{model_path}/chat/completions"
                return chat_endpoint in paths or "/chat/completions" in str(paths)
            return False
        except Exception:
            return False
    
    async def list_free_models(self) -> List[Dict[str, Any]]:
        """
        List models available on Replicate's free tier.
        This is used by the discovery mechanism to validate free availability.
        """
        # Replicate doesn't provide a specific "free tier" endpoint,
        # but we can check pricing or maintain a curated list
        # For Phase 5.1, we return commonly available free models
        return [
            {
                "id": "meta/meta-llama-3.1-8b-instruct",
                "name": "Llama 3.1 8B Instruct",
                "context_length": 8192,
                "pricing": {"input": 0, "output": 0},  # Free tier
            },
            {
                "id": "meta/meta-llama-3.1-70b-instruct",
                "name": "Llama 3.1 70B Instruct",
                "context_length": 8192,
                "pricing": {"input": 0, "output": 0},
            },
            {
                "id": "mistralai/mistral-7b-instruct-v0.3",
                "name": "Mistral 7B Instruct v0.3",
                "context_length": 32768,
                "pricing": {"input": 0, "output": 0},
            },
        ]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Replicate API connectivity and auth."""
        try:
            response = await self.client.get("account")
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy",
                    "type": data.get("type", "unknown"),
                    "username": data.get("username", "unknown"),
                }
            elif response.status_code == 401:
                return {"status": "unhealthy", "error": "authentication_failed"}
            else:
                return {"status": "unhealthy", "error": f"http_{response.status_code}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def close(self):
        """Cleanup HTTP client."""
        await self.client.aclose()
