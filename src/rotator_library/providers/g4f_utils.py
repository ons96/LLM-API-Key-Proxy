import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse


@dataclass(frozen=True)
class G4FConfig:
    api_base: str
    api_key: Optional[str]
    allow_insecure_http: bool


def load_g4f_config(api_base_env: str, api_key: Optional[str]) -> G4FConfig:
    api_base = (os.getenv(api_base_env) or "").strip()
    if not api_base:
        raise ValueError(f"Environment variable {api_base_env} is required")

    allow_insecure_http = os.getenv("G4F_ALLOW_INSECURE_HTTP", "false").lower() == "true"

    parsed = urlparse(api_base)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Invalid {api_base_env}={api_base!r}. Must start with http:// or https://"
        )

    if parsed.scheme == "http" and not allow_insecure_http:
        raise ValueError(
            f"Refusing insecure {api_base_env}={api_base!r}. Set G4F_ALLOW_INSECURE_HTTP=true to override."
        )

    return G4FConfig(api_base=api_base.rstrip("/"), api_key=api_key, allow_insecure_http=allow_insecure_http)


def build_g4f_headers(api_key: Optional[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def strip_provider_prefix(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def build_openai_payload(**kwargs: Any) -> Dict[str, Any]:
    """Build OpenAI-compatible payload for /chat/completions.

    Drops internal-only and unsupported keys to keep payload stable.
    """

    payload: Dict[str, Any] = {}

    # Common OpenAI chat fields
    allowed = {
        "model",
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "stream",
        "stream_options",
        "tools",
        "tool_choice",
        "presence_penalty",
        "frequency_penalty",
        "n",
        "stop",
        "seed",
        "response_format",
        "logit_bias",
        "user",
        "reasoning_effort",
    }

    for k, v in kwargs.items():
        if k in allowed and v is not None:
            payload[k] = v

    return payload
