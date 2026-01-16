"""
Puter.js Provider for LLM-API-Key-Proxy

Provides free access to 500+ models via puter-free-chatbot Vercel API wrapper.
"""

import httpx
import logging
from typing import List, Optional
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class PuterProvider(ProviderInterface):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.api_base = "https://puter-free-chatbot.vercel.app"
        self.models_endpoint = "/api/models"
        self.chat_endpoint = "/api/chat"

    async def get_models(
        self,
        api_key: str,
        client: httpx.AsyncClient,
        model_filter: Optional[List[str]] = None,
    ) -> List[str]:
        try:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            response = await client.get(
                f"{self.api_base}{self.models_endpoint}", headers=headers
            )
            response.raise_for_status()

            data = response.json()

            if data.get("models") and isinstance(data["models"], list):
                models = data["models"]
                if model_filter:
                    models = [
                        m
                        for m in models
                        if any(f.lower() in m.lower() for f in model_filter)
                    ]
                return models

            return []

        except httpx.RequestError as e:
            lib_logger.error(f"Failed to fetch Puter models: {e}")
            return []
        except Exception as e:
            lib_logger.error(f"Unexpected error fetching Puter models: {e}")
            return []

    async def completion(
        self,
        api_key: str,
        client: httpx.AsyncClient,
        model: str,
        messages: List[dict],
        stream: bool = False,
        **kwargs,
    ) -> dict:
        try:
            # Handle model format: only strip 'puter/' prefix if present
            # We must preserve other prefixes like 'openai:openai/' or 'openrouter:'
            # as the backend expects the full model ID from get_models()
            model_name = model
            if model.lower().startswith("puter/"):
                model_name = model[6:]

            payload = {
                "message": messages[-1]["content"] if messages else "",
                "model": model_name,
                "stream": stream,
            }
            payload.update(kwargs)

            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            response = await client.post(
                f"{self.api_base}{self.chat_endpoint}", headers=headers, json=payload
            )
            response.raise_for_status()

            data = response.json()

            return {
                "id": data.get("id", "chatcmpl-xxx"),
                "object": "chat.completion",
                "created": data.get("created", 0),
                "model": data.get("model", model),
                "choices": data.get("choices", []),
                "usage": data.get(
                    "usage",
                    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                ),
            }

        except httpx.RequestError as e:
            lib_logger.error(f"Puter API request error: {e}")
            raise
        except Exception as e:
            lib_logger.error(f"Unexpected error in Puter provider: {e}")
            raise

    def supports_streaming(self) -> bool:
        return True

    def supports_embeddings(self) -> bool:
        return False
