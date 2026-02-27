"""
DeepInfra provider implementation for OpenAI-compatible API.
Supports free tier models including Llama 3.3, Qwen2.5 Coder, and Gemma.
"""

import os
from typing import Any, Dict, List, Optional, AsyncGenerator
import httpx


class DeepInfraProvider:
    """DeepInfra API provider supporting free tier open source models."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPINFRA_API_KEY")
        self.base_url = base_url or "https://api.deepinfra.com/v1/openai"
        
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send chat completion request to DeepInfra.
        
        Args:
            model: Model identifier (e.g., 'meta-llama/Llama-3.3-70B-Instruct')
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            JSON response from DeepInfra API
            
        Raises:
            ValueError: If API key is not configured
            httpx.HTTPError: If the request fails
        """
        if not self.api_key:
            raise ValueError("DEEPINFRA_API_KEY not configured")
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
            
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
            
    async def stream_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion from DeepInfra."""
        if not self.api_key:
            raise ValueError("DEEPINFRA_API_KEY not configured")
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs
        }
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        yield line[6:]
                        
    def get_free_tier_models(self) -> List[Dict[str, Any]]:
        """
        Return list of available free tier models on DeepInfra.
        Based on research of DeepInfra's free tier offerings.
        """
        return [
            {
                "id": "meta-llama/Llama-3.3-70B-Instruct",
                "name": "Llama 3.3 70B Instruct",
                "context_window": 128000,
                "coding_score": 0.85,
                "chat_score": 0.88,
            },
            {
                "id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "name": "Llama 3.1 8B Instruct",
                "context_window": 128000,
                "coding_score": 0.72,
                "chat_score": 0.75,
                "speed": "fast",
            },
            {
                "id": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "name": "Qwen2.5 Coder 32B",
                "context_window": 128000,
                "coding_score": 0.87,
                "chat_score": 0.80,
            },
            {
                "id": "google/gemma-2-27b-it",
                "name": "Gemma 2 27B IT",
                "context_window": 8192,
                "coding_score": 0.78,
                "chat_score": 0.82,
            },
            {
                "id": "microsoft/WizardLM-2-8x22B",
                "name": "WizardLM 2 8x22B",
                "context_window": 65536,
                "coding_score": 0.84,
                "chat_score": 0.86,
            },
        ]
        
    def validate_config(self) -> tuple[bool, Optional[str]]:
        """
        Validate provider configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.api_key:
            return False, "DEEPINFRA_API_KEY environment variable not set"
        return True, None
        
    def get_rate_limits(self) -> Dict[str, int]:
        """Return default rate limits for DeepInfra free tier."""
        return {
            "rpm": 30,  # Requests per minute
            "daily": 1000,  # Daily requests
            "tpm": 100000,  # Tokens per minute
        }
