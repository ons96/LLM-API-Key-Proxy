#!/usr/bin/env python3
"""The provider's own status must survive the hop through the gateway.

A provider that does not carry the requested model answers 404. If the
gateway turns that into a 500, clients read it as a transient fault: they
retry a permanent error and finally report something unrelated ("high
demand"). These tests pin the status-forwarding behaviour.
"""

import pytest


def _load_main():
    from proxy_app import main as main_module

    return main_module


def _load_adapter():
    from proxy_app.provider_adapter import OpenAICompatibleAdapter

    return OpenAICompatibleAdapter


def _error_with_status(code):
    return type("ProviderError", (Exception,), {"status_code": code})("boom")


def test_adapter_keeps_upstream_404():
    """A missing model must stay a 404, not become a 500."""

    adapter_cls = _load_adapter()
    adapter = adapter_cls("nvidia", "key", "https://example.invalid/v1")

    err = adapter._convert_error(_error_with_status(404))
    assert err.status_code == 404


@pytest.mark.parametrize("code", [400, 401, 403, 429])
def test_adapter_keeps_client_error_statuses(code):
    adapter_cls = _load_adapter()
    adapter = adapter_cls("nvidia", "key", "https://example.invalid/v1")

    err = adapter._convert_error(_error_with_status(code))
    assert err.status_code == code


def test_adapter_keeps_status_from_nested_response():
    adapter_cls = _load_adapter()
    adapter = adapter_cls("nvidia", "key", "https://example.invalid/v1")

    class Response:
        status_code = 404

    err = adapter._convert_error(type("ProviderError", (Exception,), {"response": Response()})("gone"))
    assert err.status_code == 404


def test_adapter_still_guesses_429_without_a_status():
    """Providers that attach no status keep the keyword-based behaviour."""

    adapter_cls = _load_adapter()
    adapter = adapter_cls("nvidia", "key", "https://example.invalid/v1")

    err = adapter._convert_error(Exception("rate_limit exceeded"))
    assert err.status_code == 429


def test_adapter_still_guesses_401_without_a_status():
    adapter_cls = _load_adapter()
    adapter = adapter_cls("nvidia", "key", "https://example.invalid/v1")

    err = adapter._convert_error(Exception("invalid api_key"))
    assert err.status_code == 401


def test_adapter_defaults_to_500_without_a_status():
    adapter_cls = _load_adapter()
    adapter = adapter_cls("nvidia", "key", "https://example.invalid/v1")

    err = adapter._convert_error(Exception("something odd happened"))
    assert err.status_code == 500


def test_404_from_provider_is_forwarded():
    main = _load_main()

    class ProviderError(Exception):
        status_code = 404

    status, detail = main._upstream_error_status(ProviderError("404 page not found"))
    assert status == 404
    assert "404 page not found" in detail


def test_404_from_nested_response_is_forwarded():
    main = _load_main()

    class Response:
        status_code = 404

    class ProviderError(Exception):
        response = Response()

    status, _detail = main._upstream_error_status(ProviderError("model not found"))
    assert status == 404


def test_auth_and_rate_limit_statuses_are_forwarded():
    main = _load_main()

    for code in (400, 401, 403, 429):
        err = type("ProviderError", (Exception,), {"status_code": code})("boom")
        status, _detail = main._upstream_error_status(err)
        assert status == code


def test_statusless_error_has_no_status():
    main = _load_main()

    status, detail = main._upstream_error_status(ValueError("no status here"))
    assert status is None
    assert detail == "no status here"


def test_non_client_error_status_is_not_forwarded():
    """5xx is a gateway-side fault; the caller keeps its own default."""

    main = _load_main()

    err = type("ProviderError", (Exception,), {"status_code": 503})("upstream down")
    status, _detail = main._upstream_error_status(err)
    assert status is None


def test_non_numeric_status_is_ignored():
    main = _load_main()

    err = type("ProviderError", (Exception,), {"status_code": "not-a-number"})("boom")
    status, _detail = main._upstream_error_status(err)
    assert status is None


@pytest.mark.parametrize("code", [400, 401, 403, 404, 429])
def test_string_status_is_coerced(code):
    main = _load_main()

    err = type("ProviderError", (Exception,), {"status_code": str(code)})("boom")
    status, _detail = main._upstream_error_status(err)
    assert status == code


# --- 402 payment required ---------------------------------------------------
#
# Cerebras answers "Payment required to access this resource" for a key with
# no credit. Flattening that to 500 made the gateway classify it as transient
# and retry a permanent condition, so clients reported "high demand".


@pytest.mark.parametrize("code", [402])
def test_adapter_keeps_payment_required(code):
    adapter_cls = _load_adapter()
    adapter = adapter_cls("cerebras", "key", "https://example.invalid/v1")

    err = adapter._convert_error(_error_with_status(code))
    assert err.status_code == 402


def test_adapter_maps_payment_required_text_to_402():
    """Providers that do not attach a status still must not look transient."""
    adapter_cls = _load_adapter()
    adapter = adapter_cls("cerebras", "key", "https://example.invalid/v1")

    err = adapter._convert_error(
        Exception("Payment required to access this resource. Visit your billing tab.")
    )
    assert err.status_code == 402


@pytest.mark.parametrize("code", [402])
def test_main_forwards_payment_required(code):
    main = _load_main()
    status, _detail = main._upstream_error_status(_error_with_status(code))
    assert status == code


def test_error_status_helper_keeps_402_from_text_wrapped_error():
    """A wrapped HTTPException carries the status, so it must be forwarded."""
    from fastapi import HTTPException

    main = _load_main()
    status, detail = main._upstream_error_status(
        HTTPException(status_code=402, detail="Payment required")
    )
    assert status == 402
    assert "Payment required" in detail


def test_every_adapter_forwards_a_404():
    """Regression: each adapter class had its own keyword-only error mapping.

    A real upstream 404 (Gemini reports a retired model this way) was
    flattened to 500 by GeminiAdapter/GroqAdapter/G4FAdapter/TogetherAdapter/
    KiloAdapter, so the router retried it and Codex printed "high demand".
    """
    from proxy_app import provider_adapter as pa

    adapter_classes = [
        (pa.OpenAICompatibleAdapter, 3),
        (pa.GroqAdapter, 2),
        (pa.GeminiAdapter, 2),
        (pa.G4FAdapter, 2),
        (pa.TogetherAdapter, 2),
        (pa.KiloAdapter, 2),
    ]
    for adapter_cls, arity in adapter_classes:
        args = ("probe", "key", "https://example.invalid/v1")[:arity]
        adapter = adapter_cls(*args)
        err = adapter._convert_error(_error_with_status(404))
        assert err.status_code == 404, f"{adapter_cls.__name__} flattened 404"


def test_forwardable_upstream_status_helper():
    from proxy_app import provider_adapter as pa

    assert pa.forwardable_upstream_status(_error_with_status(404)) == 404
    assert pa.forwardable_upstream_status(_error_with_status(402)) == 402
    # Not a status we forward; callers keep their own heuristic.
    assert pa.forwardable_upstream_status(_error_with_status(500)) is None
    assert pa.forwardable_upstream_status(Exception("plain")) is None
