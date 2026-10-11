import json
import pytest
from unittest.mock import patch
from urllib.error import HTTPError

from provider_adapter import OpenAICompatibleAdapter, ProviderError, ProviderRequest
from router_core import Capability, Deployment


DEPLOYMENT = Deployment("demo", Capability.FAST_GENERAL, 1000, provider="demo", model="demo-model")


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.closed = False

    def read(self):
        return json.dumps(self.payload).encode()

    def __iter__(self):
        return iter([b"data: {\"choices\": []}\n", b"data: [DONE]\n"])

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        self.closed = True


def test_openai_compatible_adapter_completes_without_real_network():
    adapter = OpenAICompatibleAdapter("demo")
    with patch.dict("os.environ", {"DEMO_BASE_URL": "https://provider.invalid", "DEMO_API_KEY": "secret"}):
        adapter = OpenAICompatibleAdapter("demo")
        with patch("provider_adapter.request.urlopen", return_value=FakeResponse({"choices": [{"message": {"content": "ok"}}]})):
            result = adapter.complete(DEPLOYMENT, ProviderRequest([{"role": "user", "content": "hello"}], {}))
            assert result.content == "ok"
            assert result.message == {"content": "ok"}


def test_openai_compatible_adapter_streams_sse_data_lines():
    with patch.dict("os.environ", {"DEMO_BASE_URL": "https://provider.invalid"}):
        adapter = OpenAICompatibleAdapter("demo")
        with patch("provider_adapter.request.urlopen", return_value=FakeResponse({})):
            provider_request = ProviderRequest([{"role": "user", "content": "hello"}], {})
            assert list(adapter.stream(DEPLOYMENT, provider_request)) == ['{"choices": []}', "[DONE]"]


def test_openai_compatible_stream_close_failure_does_not_mask_success():
    class CloseFailingResponse(FakeResponse):
        def close(self):
            raise RuntimeError("close failed")

    with patch.dict("os.environ", {"DEMO_BASE_URL": "https://provider.invalid"}):
        adapter = OpenAICompatibleAdapter("demo")
        with patch("provider_adapter.request.urlopen", return_value=CloseFailingResponse({})):
            provider_request = ProviderRequest([{"role": "user", "content": "hello"}], {})
            assert list(adapter.stream(DEPLOYMENT, provider_request)) == ['{"choices": []}', "[DONE]"]


def test_openai_compatible_adapter_requires_endpoint():
    with patch.dict("os.environ", {}, clear=True):
        adapter = OpenAICompatibleAdapter("missing")
        try:
            adapter.complete(DEPLOYMENT, ProviderRequest([{"role": "user", "content": "hello"}], {}))
        except ProviderError as error:
            assert "base URL" in str(error)
        else:
            raise AssertionError("missing endpoint should fail")


def test_openai_compatible_adapter_converts_http_errors():
    with patch.dict("os.environ", {"DEMO_BASE_URL": "https://provider.invalid"}, clear=True):
        adapter = OpenAICompatibleAdapter("demo")
        with patch("provider_adapter.request.urlopen", side_effect=HTTPError(
            "https://provider.invalid", 429, "rate limited", {}, None
        )):
            with pytest.raises(ProviderError) as raised:
                adapter.complete(DEPLOYMENT, ProviderRequest([{"role": "user", "content": "hello"}], {}))
            assert "request failed" in str(raised.value)
            assert raised.value.error_class == "rate_limit"
            assert raised.value.status_code == 429


def test_openai_compatible_adapter_reads_retry_and_quota_headers():
    headers = {"Retry-After": "3", "X-RateLimit-Reset": "9"}
    with patch.dict("os.environ", {"DEMO_BASE_URL": "https://provider.invalid"}, clear=True):
        adapter = OpenAICompatibleAdapter("demo")
        with patch("provider_adapter.request.urlopen", side_effect=HTTPError(
            "https://provider.invalid", 429, "rate limited", headers, None
        )):
            with pytest.raises(ProviderError) as raised:
                adapter.complete(DEPLOYMENT, ProviderRequest([{"role": "user", "content": "hello"}], {}))
    assert raised.value.retry_after_seconds == 3
    assert raised.value.quota_reset_at is not None


def test_openai_compatible_adapter_reads_epoch_reset_from_generic_header():
    reset_at = 1_800_000_000
    headers = {"Retry-After": "3", "RateLimit-Reset": str(reset_at)}
    with patch.dict("os.environ", {"DEMO_BASE_URL": "https://provider.invalid"}, clear=True):
        adapter = OpenAICompatibleAdapter("demo")
        with patch("provider_adapter.request.urlopen", side_effect=HTTPError(
            "https://provider.invalid", 429, "rate limited", headers, None
        )):
            with pytest.raises(ProviderError) as raised:
                adapter.complete(DEPLOYMENT, ProviderRequest([{"role": "user", "content": "hello"}], {}))
    assert raised.value.quota_reset_at == reset_at


@pytest.mark.parametrize(
    ("exception", "error_class"),
    [(TimeoutError(), "timeout"), (OSError("connection reset"), "network_error")],
)
def test_openai_compatible_adapter_classifies_transport_errors(exception, error_class):
    with patch.dict("os.environ", {"DEMO_BASE_URL": "https://provider.invalid"}, clear=True):
        adapter = OpenAICompatibleAdapter("demo")
        with patch("provider_adapter.request.urlopen", side_effect=exception):
            with pytest.raises(ProviderError) as raised:
                adapter.complete(DEPLOYMENT, ProviderRequest([{"role": "user", "content": "hello"}], {}))
    assert raised.value.error_class == error_class


def test_provider_request_preserves_messages_tools_and_standard_options():
    provider_request = ProviderRequest.from_payload({
        "model": "auto",
        "messages": [
            {"role": "system", "content": "Be concise"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "done"},
        ],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": "auto",
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "stream": True,
        "context_tokens": 123,
    })

    payload = provider_request.to_payload("real-model", stream=False)

    assert payload["model"] == "real-model"
    assert payload["stream"] is False
    assert payload["messages"][1]["tool_calls"][0]["id"] == "call-1"
    assert payload["tools"][0]["function"]["name"] == "lookup"
    assert payload["tool_choice"] == "auto"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["temperature"] == 0
    assert "context_tokens" not in payload


def test_http_adapter_serializes_full_request_to_upstream():
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["payload"] = json.loads(req.data)
        return FakeResponse({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    with patch.dict("os.environ", {"DEMO_BASE_URL": "https://provider.invalid/v1", "DEMO_API_KEY": "secret"}):
        adapter = OpenAICompatibleAdapter("demo")
        provider_request = ProviderRequest.from_payload({
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "auto",
            "response_format": {"type": "json_object"},
            "max_tokens": 12,
        })
        with patch("provider_adapter.request.urlopen", side_effect=fake_urlopen):
            result = adapter.complete(DEPLOYMENT, provider_request)

    assert result.content == "ok"
    assert captured["url"] == "https://provider.invalid/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["payload"]["model"] == "demo-model"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["payload"]["tools"][0]["function"]["name"] == "lookup"
    assert captured["payload"]["tool_choice"] == "auto"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["max_tokens"] == 12
    assert captured["payload"]["stream"] is False


def test_deployment_base_url_can_supply_endpoint_without_url_environment():
    deployment = Deployment(
        "demo",
        Capability.FAST_GENERAL,
        1000,
        provider="demo",
        model="demo-model",
        base_url="https://metadata.invalid/v1",
    )
    with patch.dict("os.environ", {}, clear=True):
        adapter = OpenAICompatibleAdapter("demo", base_url=deployment.base_url)
    with patch("provider_adapter.request.urlopen", return_value=FakeResponse({"choices": [{"message": {"content": "ok"}}]})):
        assert adapter.complete(deployment, ProviderRequest([{"role": "user", "content": "hello"}], {})).content == "ok"
