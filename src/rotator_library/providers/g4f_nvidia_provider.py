import json
from typing import Any, AsyncGenerator, List, Union

import httpx
import litellm

from ..model_definitions import ModelDefinitions
from ..timeout_config import TimeoutConfig
from .g4f_utils import build_g4f_headers, build_openai_payload, load_g4f_config, strip_provider_prefix
from .provider_interface import ProviderInterface


class G4FNvidiaProvider(ProviderInterface):
    """G4F NVIDIA-compatible endpoint.

    Environment:
      - G4F_NVIDIA_API_BASE
      - G4F_API_KEY (optional)
    """

    skip_cost_calculation = True

    def __init__(self):
        self.model_definitions = ModelDefinitions()
        self._config = load_g4f_config("G4F_NVIDIA_API_BASE", api_key=None)
        self._default_api_key: str | None = None

    def has_custom_logic(self) -> bool:
        return True

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        models: List[str] = []

        static_models = self.model_definitions.get_all_provider_models("g4f_nvidia")
        if static_models:
            models.extend(static_models)

        try:
            url = f"{self._config.api_base}/models"
            resp = await client.get(
                url,
                headers=build_g4f_headers(api_key),
                timeout=TimeoutConfig.non_streaming(),
            )
            resp.raise_for_status()

            ids = [
                m.get("id")
                for m in resp.json().get("data", [])
                if isinstance(m, dict)
            ]
            for mid in ids:
                if mid:
                    models.append(f"g4f_nvidia/{mid}")
        except Exception:
            pass

        seen = set()
        deduped: List[str] = []
        for m in models:
            if m not in seen:
                deduped.append(m)
                seen.add(m)
        return deduped

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs: Any
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        api_key = (
            kwargs.pop("credential_identifier", None)
            or kwargs.pop("api_key", None)
            or self._default_api_key
        )

        model = kwargs.get("model")
        if not model:
            raise ValueError("'model' is required")

        model_name = strip_provider_prefix(model)
        payload = build_openai_payload(**kwargs)
        payload["model"] = model_name

        url = f"{self._config.api_base}/chat/completions"
        headers = build_g4f_headers(api_key)

        if kwargs.get("stream"):

            async def gen() -> AsyncGenerator[litellm.ModelResponse, None]:
                async with client.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=payload,
                    timeout=TimeoutConfig.streaming(),
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if line.startswith(":"):
                            continue
                        if not line.startswith("data:"):
                            continue

                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        if isinstance(chunk, dict) and chunk.get("model") == model_name:
                            chunk["model"] = model

                        yield litellm.ModelResponse(**chunk)

            return gen()

        resp = await client.post(
            url,
            headers=headers,
            json=payload,
            timeout=TimeoutConfig.non_streaming(),
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("model") == model_name:
            data["model"] = model
        return litellm.ModelResponse(**data)
