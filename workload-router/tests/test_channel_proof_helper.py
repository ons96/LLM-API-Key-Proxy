"""Channel-key proof helper contract, exercised without a live New-API."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[1] / "tools" / "newapi_channel_proof.py"
SECRET = b"12345678901234567890"
PROOF = "proof-token-abcdef0123456789abcdef"


class FakeNewAPI:
    """Loopback stand-in for the two management endpoints the helper uses."""

    def __init__(self, *, proof: str = PROOF, accept: bool = True) -> None:
        self.requests: list[tuple[str, object, dict]] = []
        self.proof = proof
        self.accept = accept
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # keep the test output quiet
                pass

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) or b"{}"
                outer.requests.append((self.path, self.headers, json.loads(raw)))
                if not outer.accept:
                    payload = {"success": False, "message": "no"}
                elif self.path == "/api/verify":
                    payload = {"success": True, "data": {"proof_token": outer.proof}}
                else:
                    payload = {"success": True, "data": {"key": "sk-test"}}
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return "http://127.0.0.1:%d" % self.server.server_address[1]

    def __enter__(self) -> "FakeNewAPI":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake_api():
    with FakeNewAPI() as api:
        yield api


@pytest.fixture
def secret_file(tmp_path: Path) -> Path:
    path = tmp_path / "totp-seed"
    path.write_text(base64.b32encode(SECRET).decode())
    path.chmod(0o600)
    return path


def run_helper(args, *, stdin: bytes = b"7\n", env=None, timeout: int = 25):
    environment = {**os.environ, "NEWAPI_PROOF_ADMIN_TOKEN": "admin-token", **(env or {})}
    return subprocess.run(
        [sys.executable, str(HELPER), *args],
        input=stdin,
        capture_output=True,
        timeout=timeout,
        env=environment,
        start_new_session=True,
    )


def totp_args(api: FakeNewAPI, secret: Path) -> list[str]:
    return [
        "--mode=totp",
        "--totp-secret-file=" + str(secret),
        "--base-url=" + api.base_url,
        "--admin-user-id=1",
    ]


def test_totp_mode_returns_exactly_one_proof_token(fake_api, secret_file):
    result = run_helper(totp_args(fake_api, secret_file))

    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout == (PROOF + "\n").encode()
    assert result.stderr == b""

    (path, headers, body), = fake_api.requests
    assert path == "/api/verify"
    assert headers.get("Authorization") == "Bearer admin-token"
    assert headers.get("New-API-User") == "1"
    assert body["method"] == "2fa"
    assert body["scope"] == "channel.key.read"
    assert body["context"] == {"channel_id": 7}
    assert len(body["code"]) == 6 and body["code"].isdigit()


@pytest.mark.parametrize("stdin", [b"", b"abc\n", b"-1\n", b"0\n", b"7\n8\n", b"7"])
def test_invalid_channel_id_fails_before_any_request(fake_api, secret_file, stdin):
    result = run_helper(totp_args(fake_api, secret_file), stdin=stdin)

    assert result.returncode == 1
    assert result.stdout == b""
    assert fake_api.requests == []


def test_management_rejection_fails_closed(secret_file):
    with FakeNewAPI(accept=False) as api:
        result = run_helper(totp_args(api, secret_file))

    assert result.returncode == 1
    assert result.stdout == b""
    assert len(api.requests) == 1


@pytest.mark.parametrize("proof", ["", "two\nlines", "bad\x00token"])
def test_malformed_proof_is_rejected(secret_file, proof):
    with FakeNewAPI(proof=proof) as api:
        result = run_helper(totp_args(api, secret_file))

    assert result.returncode == 1
    assert result.stdout == b""


def test_shared_seed_file_is_refused(fake_api, secret_file):
    secret_file.chmod(0o644)

    result = run_helper(totp_args(fake_api, secret_file))

    assert result.returncode == 1
    assert result.stdout == b""
    assert fake_api.requests == []


def test_missing_admin_token_fails_closed(fake_api, secret_file):
    result = run_helper(totp_args(fake_api, secret_file), env={"NEWAPI_PROOF_ADMIN_TOKEN": ""})

    assert result.returncode == 1
    assert result.stdout == b""
    assert fake_api.requests == []


def test_remote_base_url_is_refused(fake_api, secret_file):
    args = totp_args(fake_api, secret_file)
    args[2] = "--base-url=http://203.0.113.10:8001"

    result = run_helper(args)

    assert result.returncode == 1
    assert result.stdout == b""
    assert fake_api.requests == []


def test_interactive_mode_fails_without_a_terminal(fake_api):
    result = run_helper(["--mode=interactive", "--base-url=" + fake_api.base_url,
                         "--admin-user-id=1"])

    assert result.returncode == 1
    assert result.stdout == b""
    assert fake_api.requests == []


def test_sidecar_startup_path_reaches_the_helper(fake_api, secret_file, monkeypatch):
    """The real wrapper used by serve() must accept this helper unchanged."""
    from newapi_server import channel_key_proof_from_command

    monkeypatch.setenv("NEWAPI_PROOF_ADMIN_TOKEN", "admin-token")
    command = " ".join([sys.executable, str(HELPER), *totp_args(fake_api, secret_file)])

    assert channel_key_proof_from_command(command)(7) == PROOF
