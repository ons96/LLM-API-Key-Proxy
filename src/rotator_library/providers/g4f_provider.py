import os
import httpx
import logging
import json
from typing import List, Dict, Any, Optional, AsyncGenerator, Union

import litellm

from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class G4FProvider(ProviderInterface):
    """G4F (g4f) fallback provider implementation.

    Notes on configuration:
    - By default, we use https://g4f.dev (public endpoint).
    - Custom endpoints can be configured via G4F_*_API_BASE environment variables.
    - Some deployments expose OpenAI-compatible endpoints at either:
        - <base>/v1/... (OpenAI standard)
        - <base>/...    (non-standard)
      We attempt both when talking to custom endpoints.

    Credentials:
    - G4F may not require an API key.
    - The proxy still models "credentials" as strings, so users commonly set
      G4F_API_KEY="" to enable the provider (an "anonymous" credential).
    - If you do have multiple keys, you can define G4F_API_KEY_1, G4F_API_KEY_2, ...
      and the proxy will rotate between them.
    """

    provider_name = "g4f"
    provider_env_name = "g4f"

    # G4F is typically a fallback tier
    default_tier_priority = 5

    _DEFAULT_PUBLIC_BASE = "https://g4f.dev"

    def __init__(self):
        # API key (optional)
        self.api_key: Optional[str] = os.getenv("G4F_API_KEY") or None

        # Base URLs for different G4F-compatible endpoints
        self.main_api_base: Optional[str] = os.getenv("G4F_MAIN_API_BASE") or None
        self.groq_api_base: Optional[str] = os.getenv("G4F_GROQ_API_BASE") or None
        self.grok_api_base: Optional[str] = os.getenv("G4F_GROK_API_BASE") or None
        self.gemini_api_base: Optional[str] = os.getenv("G4F_GEMINI_API_BASE") or None
        self.nvidia_api_base: Optional[str] = os.getenv("G4F_NVIDIA_API_BASE") or None

        # Log without leaking secrets
        lib_logger.info(
            "Initialized G4F provider (custom main base configured=%s)",
            bool(self._is_valid_base_url(self.main_api_base)),
        )

    @staticmethod
    def _is_valid_base_url(url: Optional[str]) -> bool:
        if not url:
            return False
        url = url.strip()
        if not url:
            return False
        # Common placeholder from templates/docs
        if "example.com" in url.lower():
            return False
        return True

    @staticmethod
    def _normalize_base_url(url: str) -> str:
        return url.strip().rstrip("/")

    @classmethod
    def _build_openai_url(cls, base_url: str, endpoint: str) -> str:
        """Build a URL for OpenAI-compatible endpoints.

        If base_url already ends with /v1, we avoid double-inserting it.
        """

        base = cls._normalize_base_url(base_url)
        endpoint = endpoint.lstrip("/")

        if base.endswith("/v1"):
            return f"{base}/{endpoint}"
        return f"{base}/v1/{endpoint}"

    def _select_base_url(self, proxy_model: str) -> str:
        """Select an endpoint base URL.

        If users configured provider-specific bases (groq/grok/gemini/nvidia),
        we apply lightweight heuristics based on the model name.

        Otherwise, use G4F_MAIN_API_BASE if set, else the public g4f.dev.
        """

        base_model = proxy_model
        if proxy_model.startswith("g4f/"):
            base_model = proxy_model.split("/", 1)[1]

        model_lower = base_model.lower()

        # Prefer specialized endpoints when configured
        if self._is_valid_base_url(self.gemini_api_base) and "gemini" in model_lower:
            return self._normalize_base_url(self.gemini_api_base)  # type: ignore[arg-type]
        if self._is_valid_base_url(self.grok_api_base) and "grok" in model_lower:
            return self._normalize_base_url(self.grok_api_base)  # type: ignore[arg-type]
        if self._is_valid_base_url(self.nvidia_api_base) and (
            "nvidia" in model_lower or model_lower.startswith("nv-")
        ):
            return self._normalize_base_url(self.nvidia_api_base)  # type: ignore[arg-type]
        if self._is_valid_base_url(self.groq_api_base) and any(
            kw in model_lower for kw in ("llama", "mixtral", "gemma")
        ):
            return self._normalize_base_url(self.groq_api_base)  # type: ignore[arg-type]

        if self._is_valid_base_url(self.main_api_base):
            return self._normalize_base_url(self.main_api_base)  # type: ignore[arg-type]

        return self._DEFAULT_PUBLIC_BASE

    @staticmethod
    def _strip_provider_prefix(proxy_model: str) -> str:
        return proxy_model.split("/", 1)[1] if proxy_model.startswith("g4f/") else proxy_model

    def _build_headers(self, credential_identifier: Optional[str]) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "LLM-API-Key-Proxy/G4FProvider",
        }

        token = credential_identifier or self.api_key
        if token:
            headers["Authorization"] = f"Bearer {token}"

        return headers

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Fetch available models from G4F.

        Priority:
        1. Fetch from official g4f.dev API
        2. Fetch from a configured custom endpoint (tries /v1/models then /models)
        3. Fall back to g4f library's ModelUtils
        4. Final fallback to a static model list
        """

        # 1) Official g4f.dev models endpoint
        try:
            response = await client.get(
                "https://g4f.dev/v1/models",
                timeout=15.0,
                headers={"User-Agent": "LLM-API-Key-Proxy/1.0"},
            )
            response.raise_for_status()

            models_data = response.json()
            models: List[str] = []

            if "data" in models_data:
                for model in models_data["data"]:
                    if isinstance(model, dict) and "id" in model:
                        model_id = model["id"]
                        skip_keywords = [
                            "flux",
                            "dall-e",
                            "midjourney",
                            "stable-diffusion",
                            "sdxl",
                            "imagen",
                            "suno",
                            "sora",
                            "veo",
                            "kling",
                            "whisper",
                            "tts",
                            "playai-tts",
                            "gpt-image",
                            "z-image",
                            "grok-imagine-video",
                            "midijourney",
                            "seedream",
                        ]
                        if not any(kw in model_id.lower() for kw in skip_keywords):
                            models.append(f"g4f/{model_id}")

            if models:
                lib_logger.info(f"Discovered {len(models)} models from g4f.dev API")
                return models

        except httpx.RequestError as e:
            lib_logger.debug(f"Failed to fetch models from g4f.dev: {e}")
        except Exception as e:
            lib_logger.debug(f"Error parsing g4f.dev models response: {e}")

        # 2) Custom endpoint if configured
        base_url = self._select_base_url("g4f")
        if self._is_valid_base_url(base_url) and "g4f.dev" not in base_url:
            headers = self._build_headers(api_key or None)

            # Try OpenAI-standard /v1/models first
            for models_url in (
                self._build_openai_url(base_url, "models"),
                f"{self._normalize_base_url(base_url)}/models",
            ):
                try:
                    response = await client.get(models_url, headers=headers, timeout=10.0)
                    response.raise_for_status()

                    models_data = response.json()
                    models: List[str] = []

                    if "data" in models_data:
                        for model in models_data["data"]:
                            if isinstance(model, dict) and "id" in model:
                                models.append(f"g4f/{model['id']}")
                    elif "models" in models_data:
                        for model in models_data["models"]:
                            if isinstance(model, str):
                                models.append(f"g4f/{model}")
                            elif isinstance(model, dict) and "name" in model:
                                models.append(f"g4f/{model['name']}")

                    if models:
                        lib_logger.info(
                            f"Discovered {len(models)} models from custom G4F API ({models_url})"
                        )
                        return models

                except httpx.HTTPStatusError as e:
                    # Try the next variant only on 404; otherwise surface error
                    if e.response.status_code != 404:
                        raise
                except httpx.RequestError as e:
                    lib_logger.debug(f"Failed to fetch G4F models from custom API: {e}")
                    break
                except Exception as e:
                    lib_logger.debug(f"Error parsing custom G4F API response: {e}")
                    break

        # 3) g4f library model list
        try:
            from g4f.models import ModelUtils

            all_model_names = list(ModelUtils.convert.keys())
            excluded_keywords = [
                "flux",
                "dall-e",
                "midjourney",
                "stable-diffusion",
                "sdxl",
                "playground",
                "imagen",
                "whisper",
                "tts",
                "suno",
                "audio",
                "music",
                "video",
                "image",
            ]

            chat_models: List[str] = []
            for model_name in all_model_names:
                model_lower = model_name.lower()
                if not any(kw in model_lower for kw in excluded_keywords):
                    chat_models.append(f"g4f/{model_name}")

            lib_logger.info(f"Discovered {len(chat_models)} chat models from g4f library")
            return chat_models

        except ImportError as e:
            lib_logger.warning(f"g4f library not available: {e}")
        except Exception as e:
            lib_logger.warning(f"Error discovering g4f models from library: {e}")

        # 4) Static fallback
        static_models = [
            "g4f/gpt-4o",
            "g4f/gpt-4o-mini",
            "g4f/gpt-4",
            "g4f/gpt-3.5-turbo",
            "g4f/claude-3.5-sonnet",
            "g4f/claude-3-haiku",
            "g4f/claude-sonnet-4",
            "g4f/gemini-2.5-pro",
            "g4f/gemini-2.5-flash",
            "g4f/gemini-2.0-flash",
            "g4f/llama-3.3-70b-versatile",
            "g4f/llama-3.1-405b-instruct",
            "g4f/deepseek-v3",
            "g4f/deepseek-r1",
            "g4f/qwq-32b",
            "g4f/grok",
            "g4f/mistral-large",
            "g4f/command-r-plus",
        ]

        lib_logger.info(f"Using fallback static model list: {len(static_models)} models")
        return static_models

    def has_custom_logic(self) -> bool:
        return True

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        proxy_model = kwargs.get("model", "")
        if not proxy_model:
            raise ValueError("'model' is required")

        credential_identifier = kwargs.get("credential_identifier")
        stream = bool(kwargs.get("stream", False))

        base_url = self._select_base_url(proxy_model)
        api_url = self._build_openai_url(base_url, "chat/completions")

        # G4F endpoints expect model WITHOUT the proxy provider prefix
        upstream_model = self._strip_provider_prefix(proxy_model)

        request_data: Dict[str, Any] = {
            "model": upstream_model,
            "messages": kwargs.get("messages", []),
            "stream": stream,
        }

        # Optional OpenAI-like parameters
        for key in (
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "tools",
            "tool_choice",
        ):
            if key in kwargs and kwargs[key] is not None:
                request_data[key] = kwargs[key]

        headers = self._build_headers(credential_identifier)

        if stream:
            return self._handle_streaming_completion(
                client=client,
                api_url=api_url,
                headers=headers,
                request_data=request_data,
                proxy_model=proxy_model,
            )

        return await self._handle_non_streaming_completion(
            client=client,
            api_url=api_url,
            headers=headers,
            request_data=request_data,
            proxy_model=proxy_model,
        )

    async def _handle_non_streaming_completion(
        self,
        client: httpx.AsyncClient,
        api_url: str,
        headers: Dict[str, str],
        request_data: Dict[str, Any],
        proxy_model: str,
    ) -> litellm.ModelResponse:
        response = await client.post(api_url, headers=headers, json=request_data)
        response.raise_for_status()

        response_data = response.json()
        return self._convert_g4f_response(response_data, proxy_model=proxy_model)

    async def _handle_streaming_completion(
        self,
        client: httpx.AsyncClient,
        api_url: str,
        headers: Dict[str, str],
        request_data: Dict[str, Any],
        proxy_model: str,
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        async with client.stream(
            "POST",
            api_url,
            headers=headers,
            json=request_data,
        ) as response:
            # Preload body if error, so response.text is available to error handling/logging
            if response.status_code >= 400:
                try:
                    await response.aread()
                except Exception:
                    pass

            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue

                if line.startswith("data: "):
                    line_data = line[6:]
                    if line_data.strip() == "[DONE]":
                        break

                    try:
                        chunk_data = json.loads(line_data)
                        yield self._convert_g4f_chunk(
                            chunk_data, proxy_model=proxy_model
                        )
                    except json.JSONDecodeError:
                        lib_logger.warning(
                            f"Failed to parse G4F streaming chunk: {line_data[:200]}"
                        )
                        continue

    def _convert_g4f_response(
        self, response_data: Dict[str, Any], proxy_model: str
    ) -> litellm.ModelResponse:
        try:
            created = response_data.get("created", 0)

            choices: List[litellm.Choices] = []
            for choice in response_data.get("choices", []) or []:
                message = choice.get("message", {}) if isinstance(choice, dict) else {}
                choice_obj = litellm.Choices(
                    index=choice.get("index", 0) if isinstance(choice, dict) else 0,
                    message=litellm.Message(
                        role=message.get("role", "assistant"),
                        content=message.get("content", ""),
                    ),
                    finish_reason=choice.get("finish_reason")
                    if isinstance(choice, dict)
                    else None,
                )
                choices.append(choice_obj)

            usage_data = response_data.get("usage") if isinstance(response_data, dict) else None
            usage = None
            if isinstance(usage_data, dict) and usage_data:
                usage = litellm.Usage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                )

            return litellm.ModelResponse(
                id=response_data.get("id", "g4f-completion"),
                object=response_data.get("object", "chat.completion"),
                created=created,
                model=proxy_model,
                choices=choices,
                usage=usage,
            )

        except Exception as e:
            lib_logger.error(f"Error converting G4F response: {e}")
            return litellm.ModelResponse(
                id="g4f-error",
                object="chat.completion",
                created=0,
                model=proxy_model,
                choices=[
                    litellm.Choices(
                        index=0,
                        message=litellm.Message(
                            role="assistant",
                            content=f"Error processing response: {str(e)}",
                        ),
                        finish_reason="error",
                    )
                ],
            )

    def _convert_g4f_chunk(
        self, chunk_data: Dict[str, Any], proxy_model: str
    ) -> litellm.ModelResponse:
        try:
            # OpenAI streaming format: {id, object, created, model, choices:[{delta:{...}, finish_reason, index}]}
            choices_in = (
                chunk_data.get("choices", []) if isinstance(chunk_data, dict) else []
            )
            if (
                choices_in
                and isinstance(choices_in, list)
                and isinstance(choices_in[0], dict)
            ):
                choice0 = choices_in[0]
                delta = (
                    choice0.get("delta", {})
                    if isinstance(choice0.get("delta"), dict)
                    else {}
                )
                index = choice0.get("index", 0)
                finish_reason = choice0.get("finish_reason")
            else:
                # Fallback for non-standard streaming formats
                delta = (
                    chunk_data.get("delta", {})
                    if isinstance(chunk_data.get("delta"), dict)
                    else {}
                )
                index = chunk_data.get("index", 0)
                finish_reason = chunk_data.get("finish_reason")

            chunk_out: Dict[str, Any] = {
                "id": chunk_data.get("id", "g4f-chunk"),
                "object": chunk_data.get("object", "chat.completion.chunk"),
                "created": chunk_data.get("created", 0),
                "model": proxy_model,
                "choices": [
                    {
                        "index": index,
                        "delta": delta,
                        "finish_reason": finish_reason,
                    }
                ],
            }

            # Preserve usage if the upstream provides it in the stream
            if isinstance(chunk_data, dict) and "usage" in chunk_data:
                chunk_out["usage"] = chunk_data["usage"]

            return litellm.ModelResponse(**chunk_out)

        except Exception as e:
            lib_logger.error(f"Error converting G4F chunk: {e}")
            return litellm.ModelResponse(
                id="g4f-chunk-error",
                object="chat.completion.chunk",
                created=0,
                model=proxy_model,
                choices=[
                    litellm.Choices(
                        index=0,
                        message=litellm.Message(role="assistant", content=""),
                        finish_reason="error",
                    )
                ],
            )

    async def aembedding(self, client: httpx.AsyncClient, **kwargs) -> litellm.EmbeddingResponse:
        lib_logger.info("G4F embeddings not supported - returning empty response")
        return litellm.EmbeddingResponse(
            id="g4f-embeddings-unsupported",
            object="list",
            data=[],
            model="g4f/embeddings-unsupported",
        )

    @staticmethod
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        error_message = str(error).lower()
        body_lower = (error_body or "").lower()
        combined = f"{error_message}\n{body_lower}"

        if "rate limit" in combined or "429" in combined:
            return {
                "retry_after": 60,
                "reason": "RATE_LIMITED",
                "reset_timestamp": None,
                "quota_reset_timestamp": None,
            }

        if "quota" in combined or "exhausted" in combined:
            return {
                "retry_after": 300,
                "reason": "QUOTA_EXHAUSTED",
                "reset_timestamp": None,
                "quota_reset_timestamp": None,
            }

        return None

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {credential_identifier}"}

    def get_credential_tier_name(self, credential: str) -> Optional[str]:
        return "free-tier"
