from integrations.opencode_headers import router_headers


def test_opencode_headers_are_bounded_and_signal_only():
    headers = router_headers("session", "verify", "run-tests", True)
    assert headers == {
        "X-Session-Id": "session",
        "X-Router-Phase": "verify",
        "X-Router-Action": "run-tests",
        "X-Router-Verification": "failed",
    }
    assert len(router_headers("x" * 1000)["X-Session-Id"]) == 256
