import httpx
import logging
import re
from typing import Any, Dict, List, Optional
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False  # Ensure this logger doesn't propagate to root
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class GroqProvider(ProviderInterface):
    """
    Provider implementation for the Groq API.
    """

    @staticmethod
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse Groq rate-limit/quota errors.

        Groq returns headers like:
        - x-ratelimit-reset-tokens: "7.66s"
        - x-ratelimit-reset-requests: "2m59s"
        - retry-after: "60"
        Sometimes 413 is used for TPM overflow with a body mentioning TPM.
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
                headers.get("x-ratelimit-reset-tokens"),
                headers.get("x-ratelimit-reset-requests"),
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
        if "tpm" in body_text or "tokens per minute" in body_text:
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

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the Groq API.
        """
        try:
            response = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            return [f"groq/{model['id']}" for model in response.json().get("data", [])]
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to fetch Groq models: {e}")
            return []
