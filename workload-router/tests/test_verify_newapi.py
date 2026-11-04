"""Provider-free checks for the real-binary verifier's startup gate."""

import pytest

from newapi_catalog import CatalogError
from tools.verify_newapi import wait_entrypoint


class Process:
    """Only the exit status is relevant to the boot probe."""

    def __init__(self, result: int | None) -> None:
        self.result = result

    def poll(self) -> int | None:
        return self.result


def test_failed_authorization_cannot_accept_a_healthy_listener(monkeypatch):
    monkeypatch.setattr(
        "tools.verify_newapi.get_json",
        lambda *_args: (200, {"service": "newapi-sidecar"}),
    )
    with pytest.raises(CatalogError, match="before successful authorization"):
        wait_entrypoint(Process(1), ("127.0.0.1", 8003), fail_closed=True)


@pytest.mark.parametrize("result", [0, 1])
def test_valid_startup_cannot_accept_an_early_exit(monkeypatch, result):
    def unavailable(*_args):
        raise ConnectionRefusedError

    monkeypatch.setattr("tools.verify_newapi.get_json", unavailable)
    with pytest.raises(CatalogError, match="exited unexpectedly"):
        wait_entrypoint(Process(result), ("127.0.0.1", 8003), fail_closed=False)


def test_nonzero_authorization_exit_is_fail_closed(monkeypatch):
    def unavailable(*_args):
        raise ConnectionRefusedError

    monkeypatch.setattr("tools.verify_newapi.get_json", unavailable)
    wait_entrypoint(Process(1), ("127.0.0.1", 8003), fail_closed=True)


def test_startup_requires_the_expected_service_health(monkeypatch):
    monkeypatch.setattr(
        "tools.verify_newapi.get_json", lambda *_args: (200, {"service": "other"})
    )
    with pytest.raises(CatalogError, match="unexpected sidecar health"):
        wait_entrypoint(Process(None), ("127.0.0.1", 8003), fail_closed=False)


def test_startup_probe_has_a_deadline(monkeypatch):
    clock = iter((0.0, 11.0))
    monkeypatch.setattr("tools.verify_newapi.time.monotonic", lambda: next(clock))
    with pytest.raises(CatalogError, match="startup timed out"):
        wait_entrypoint(Process(None), ("127.0.0.1", 8003), fail_closed=False)
