"""Tests for `build_litellm_fallback_kwargs`.

Regression: when the adapter factory cannot create an adapter for a free
provider (`kilo`, `aihubmix`, `zanity`, `paxsenix`, ...), the gateway
falls back to `litellm.acompletion`. Previously this passed
`model="kilo/meta-llama/llama-3.3"` with no api_key and no api_base,
which made litellm raise
`litellm.BadRequestError: LLM Provider NOT provided` because "kilo" is
not a known litellm provider and there was nothing to dispatch with.

The fix: strip the provider prefix from the model, force
`custom_llm_provider="openai"`, and pass through the env-resolved
api_key / api_base. We also fall back to the literal "EMPTY" api_key
when the upstream endpoint accepts no auth.
"""

import pytest

from src.proxy_app.litellm_fallback import build_litellm_fallback_kwargs


class TestBuildLitellmFallbackKwargs:
    def test_strips_provider_prefix_from_model(self):
        """The model string must be the bare upstream id, not 'provider/model'."""
        request_clean = {"model": "kilo/meta-llama/llama-3.3-70b-instruct:free"}
        kwargs = build_litellm_fallback_kwargs(
            request_clean=request_clean,
            provider="kilo",
            model="meta-llama/llama-3.3-70b-instruct:free",
            api_key="sk-kilo-xxx",
            api_base="https://api.kilo.ai/v1",
        )
        assert kwargs["model"] == "meta-llama/llama-3.3-70b-instruct:free"
        # Original request_clean must not be mutated.
        assert request_clean["model"] == "kilo/meta-llama/llama-3.3-70b-instruct:free"

    def test_forces_openai_provider(self):
        """Without custom_llm_provider=openai, litellm can't route kilo/aihubmix."""
        kwargs = build_litellm_fallback_kwargs(
            request_clean={"model": "aihubmix/glm-4-flash"},
            provider="aihubmix",
            model="glm-4-flash",
            api_key="sk-ahub-xxx",
            api_base="https://aihubmix.com/v1",
        )
        assert kwargs["custom_llm_provider"] == "openai"

    def test_passes_api_key_and_api_base(self):
        kwargs = build_litellm_fallback_kwargs(
            request_clean={"model": "kilo/x"},
            provider="kilo",
            model="x",
            api_key="sk-kilo",
            api_base="https://api.kilo.ai/v1",
        )
        assert kwargs["api_key"] == "sk-kilo"
        assert kwargs["api_base"] == "https://api.kilo.ai/v1"

    def test_empty_api_key_falls_back_to_empty_sentinel(self):
        """When env is unset, upstream may accept no auth — use sentinel."""
        kwargs = build_litellm_fallback_kwargs(
            request_clean={"model": "kilo/x"},
            provider="kilo",
            model="x",
            api_key=None,
            api_base="https://api.kilo.ai/v1",
        )
        assert kwargs["api_key"] == "EMPTY"

    def test_none_api_base_omitted(self):
        """Don't pass api_base=None to litellm — let it default."""
        kwargs = build_litellm_fallback_kwargs(
            request_clean={"model": "kilo/x"},
            provider="kilo",
            model="x",
            api_key="sk-kilo",
            api_base=None,
        )
        assert "api_base" not in kwargs

    def test_preserves_other_request_fields(self):
        """messages, max_tokens, temperature, etc. must be carried through."""
        request_clean = {
            "model": "kilo/x",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        kwargs = build_litellm_fallback_kwargs(
            request_clean=request_clean,
            provider="kilo",
            model="x",
            api_key="sk-kilo",
            api_base="https://api.kilo.ai/v1",
        )
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert kwargs["max_tokens"] == 1024
        assert kwargs["temperature"] == 0.7

    @pytest.mark.parametrize(
        "provider,model,api_base",
        [
            ("kilo", "meta-llama/llama-3.3-70b-instruct:free", "https://api.kilo.ai/v1"),
            ("aihubmix", "glm-4-flash", "https://aihubmix.com/v1"),
            ("zanity", "gpt-4o", "https://api.zanity.com/v1"),
            ("paxsenix", "claude-3-5-sonnet", "https://api.paxsenix.com/v1"),
            ("groq", "llama-3.1-8b-instant", "https://api.groq.com/openai/v1"),
        ],
    )
    def test_realistic_provider_matrix(self, provider, model, api_base):
        """Smoke test across all free providers that have triggered this bug."""
        kwargs = build_litellm_fallback_kwargs(
            request_clean={"model": f"{provider}/{model}"},
            provider=provider,
            model=model,
            api_key=f"sk-{provider}-test",
            api_base=api_base,
        )
        assert kwargs["model"] == model
        assert kwargs["api_key"] == f"sk-{provider}-test"
        assert kwargs["api_base"] == api_base
        assert kwargs["custom_llm_provider"] == "openai"
