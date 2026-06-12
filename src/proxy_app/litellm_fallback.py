"""Build kwargs for `litellm.acompletion` when the adapter factory fails.

Pure helper module. No litellm import here so tests stay hermetic.
"""

from typing import Any, Dict, Optional


def build_litellm_fallback_kwargs(
    request_clean: Dict[str, Any],
    provider: str,
    model: str,
    api_key: Optional[str],
    api_base: Optional[str],
) -> Dict[str, Any]:
    """Build kwargs for `litellm.acompletion` when the adapter factory fails.

    Free providers like `kilo`, `aihubmix`, `zanity`, `paxsenix` are NOT
    known providers in litellm. Passing `model=f"{provider}/{model}"`
    directly causes litellm to parse `provider` as the upstream name,
    then fail with `BadRequestError: LLM Provider NOT provided` because
    it has no api_key, no api_base, and no provider class for that name.

    We solve this by:
      1. Stripping the provider prefix from `model` (litellm gets the
         raw upstream model id).
      2. Forcing `custom_llm_provider="openai"` so litellm uses the
         OpenAI-compatible code path.
      3. Passing through the resolved `api_key` and `api_base` from the
         provider config (env-resolved upstream).
      4. Falling back to the literal sentinel `"EMPTY"` if the env
         variable is unset — the upstream provider may accept no auth
         (e.g. a public demo endpoint) and litellm would otherwise
         refuse to dispatch.
    """
    kwargs: Dict[str, Any] = {
        **request_clean,
        "model": model,
        "api_key": api_key if api_key else "EMPTY",
        "custom_llm_provider": "openai",
    }
    if api_base:
        kwargs["api_base"] = api_base
    return kwargs
