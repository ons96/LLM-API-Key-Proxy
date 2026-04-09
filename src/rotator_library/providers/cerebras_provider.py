import httpx
import logging
import re
from typing import Any, Dict, List, Optional
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class CerebrasProvider(ProviderInterface):
    """
    Provider implementation for Cerebras Inference API.

    Cerebras offers extremely fast inference with a generous free tier.
    API Base: https://api.cerebras.ai/v1

    Free tier limits:
    - 1 million tokens per day
    - 64K context window
    - Rate limited

    Available models:
    - llama-3.1-8b (Llama 3.1 8B)
    - llama-3.3-70b (Llama 3.3 70B)
    - qwen-3-32b (Qwen 3 32B)
    - qwen-3-235b (Qwen 3 235B Instruct - preview)
    - gpt-oss-120b (GPT OSS 120B)
    - zai-glm-4.6 (ZAI GLM 4.6 - preview)
    """

    provider_name = "cerebras"
    provider_env_name = "cerebras"

    # High priority - very fast inference with good free tier
    default_tier_priority = 2

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from Cerebras API.
        """
        try:
            response = await client.get(
                "https://api.cerebras.ai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            if "data" in data:
                models = [f"cerebras/{model['id']}" for model in data.get("data", [])]
                if models:
                    lib_logger.info(
                        f"Discovered {len(models)} models from Cerebras API"
                    )
                    return models

        except httpx.RequestError as e:
            lib_logger.debug(f"Failed to fetch Cerebras models: {e}")
        except Exception as e:
            lib_logger.debug(f"Error parsing Cerebras models: {e}")

        # Fallback to known models (updated to match actual API response)
        static_models = [
            "cerebras/llama3.1-8b",
            "cerebras/llama-3.3-70b",
            "cerebras/qwen-3-32b",
            "cerebras/qwen-3-235b-a22b-instruct-2507",
            "cerebras/gpt-oss-120b",
            "cerebras/zai-glm-4.7",
            "cerebras/zai-glm-4.6",
        ]

        lib_logger.info(
            f"Using fallback Cerebras model list: {len(static_models)} models"
        )
        return static_models

    @staticmethod
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse Cerebras rate-limit/quota errors.

        Headers observed:
        - x-ratelimit-remaining-tokens-minute
        - x-ratelimit-reset-tokens-minute
        - x-ratelimit-remaining-requests-day
        - x-ratelimit-reset-requests-day
        """

        def _parse_seconds(val: Optional[str]) -> Optional[int]:
            if val is None:
                return None
            s = str(val).strip().lower().replace("s", "")
            try:
                return int(float(s))
            except (ValueError, TypeError):
                return None

        headers = None
        if isinstance(error, httpx.HTTPStatusError) and hasattr(error, "response"):
            headers = error.response.headers or {}

        retry_after = None
        reason = "RATE_LIMITED"

        if headers:
            candidates = [
                headers.get("retry-after"),
                headers.get("x-ratelimit-reset-tokens-minute"),
                headers.get("x-ratelimit-reset-requests-day"),
            ]
            for c in candidates:
                retry_after = _parse_seconds(c)
                if retry_after:
                    break

        body_text = (error_body or "").lower()
        if not retry_after:
            m = re.search(r"retry\s*after\s*(\d+)", body_text)
            if m:
                retry_after = _parse_seconds(m.group(1))

        if "tokens per minute" in body_text or "tpm" in body_text:
            reason = "TPM_LIMIT"
        elif "tokens per day" in body_text or "tpd" in body_text:
            reason = "TPD_LIMIT"
        elif "requests per minute" in body_text or "rpm" in body_text:
            reason = "RPM_LIMIT"
        elif "requests per day" in body_text or "rpd" in body_text:
            reason = "RPD_LIMIT"

        if retry_after:
            return {
                "retry_after": retry_after,
                "reason": reason,
                "quota_reset_timestamp": None,
            }
        return None
