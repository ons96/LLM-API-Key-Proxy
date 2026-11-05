"""Provider-free checks for the real-binary verifier's startup gate."""

import subprocess

import pytest

from newapi_catalog import CatalogError
from tools.verify_newapi import fixture_command, wait_entrypoint


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


@pytest.mark.parametrize("code", [0, 1])
def test_fixture_command_requires_the_expected_exit_status(tmp_path, monkeypatch, code):
    def run(argv, **kwargs):
        kwargs["stdout"].write(b'{"ok":true}')
        kwargs["stderr"].write(b"private diagnostic")
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["timeout"] == 45
        assert kwargs["umask"] == 0o077
        return subprocess.CompletedProcess(argv, code)

    monkeypatch.setattr("tools.verify_newapi.subprocess.run", run)
    if code:
        with pytest.raises(CatalogError, match="fixture command failed"):
            fixture_command(tmp_path, {}, "check", ["fake"])
    else:
        assert fixture_command(tmp_path, {}, "check", ["fake"]) == b'{"ok":true}'


def test_fixture_command_hides_timeout_output(tmp_path, monkeypatch):
    def run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["fake"], 45, output="sensitive-value")

    monkeypatch.setattr("tools.verify_newapi.subprocess.run", run)
    with pytest.raises(CatalogError, match="could not complete") as error:
        fixture_command(tmp_path, {}, "check", ["fake"])
    assert "sensitive-value" not in str(error.value)


def test_fixture_command_rejects_excess_output(tmp_path, monkeypatch):
    def run(argv, **kwargs):
        kwargs["stdout"].write(b"x" * (1024 * 1024 + 1))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr("tools.verify_newapi.subprocess.run", run)
    with pytest.raises(CatalogError, match="output exceeds limit"):
        fixture_command(tmp_path, {}, "check", ["fake"])


def test_fixture_command_rejects_credential_echo_without_exposing_it(
    tmp_path, monkeypatch
):
    def run(argv, **kwargs):
        kwargs["stdout"].write(b"generated-sensitive-value")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr("tools.verify_newapi.subprocess.run", run)
    with pytest.raises(CatalogError, match="logged a generated credential") as error:
        fixture_command(
            tmp_path, {"NEWAPI_API_KEY": "generated-sensitive-value"}, "check", ["fake"]
        )
    assert "generated-sensitive-value" not in str(error.value)
