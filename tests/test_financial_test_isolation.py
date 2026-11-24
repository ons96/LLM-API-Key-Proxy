"""Self-tests for the Stage 1 verifier, not production financial eligibility."""

import ctypes
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from financial_test_bootstrap import GUARD, IsolationError

if GUARD is None:
    raise RuntimeError("use python -I -S tests/support/financial_test_bootstrap.py")


@pytest.mark.parametrize("path", [
    "/repo/.env", "/scratch/.env", "/scratch/.env.local",
    "/home/operator/.secrets/provider", "/home/operator/.ssh/id_ed25519",
    "/tmp/llm_proxy_telemetry.db", "/tmp/llm_task_telemetry.db",
    "/dev/shm/telemetry.db", "/repo/data/rate_counters.json",
    "/scratch/tmp/llm_proxy_telemetry.db", "/tmp/../tmp/llm_proxy_telemetry.db",
])
def test_credential_and_default_storage_reads_are_blocked(path):
    with GUARD.expect_blocked(), pytest.raises(IsolationError):
        Path(path).read_bytes()


@pytest.mark.parametrize("path", ["/repo/README.md", "/tmp/llm_proxy_telemetry.db"])
def test_forbidden_writes_are_blocked(path):
    with GUARD.expect_blocked(), pytest.raises(IsolationError):
        Path(path).write_text("synthetic")


@pytest.mark.parametrize("path", [
    "/tmp/llm_proxy_telemetry.db", "file:/tmp/llm_proxy_telemetry.db?mode=ro",
    "file://provider.invalid/scratch/test.db?mode=ro",
    "file:/scratch/test.db?vfs=unix-dotfile", "file:/scratch/test.db?mode=ro&mode=rw",
    "/repo/data/active_days.db",
])
def test_sqlite_cannot_bypass_file_guard(path):
    with GUARD.expect_blocked(), pytest.raises(IsolationError):
        sqlite3.connect(path, uri=path.startswith("file:"))


@pytest.mark.parametrize("address", [
    ("127.0.0.1", 8000), ("127.0.0.1", 8001), ("localhost", 8000),
    ("192.0.2.1", 443), ("provider.invalid", 443),
])
def test_unregistered_connections_are_blocked_before_dns_or_connect(address):
    with GUARD.expect_blocked(), pytest.raises(IsolationError):
        socket.create_connection(address, timeout=0.1)


def test_low_level_socket_connect_is_guarded():
    with socket.socket() as client:
        with GUARD.expect_blocked(), pytest.raises(IsolationError):
            client.connect(("127.0.0.1", 8000))


def test_reverse_lookup_never_delegates_to_native_dns():
    import _socket

    assert socket.gethostbyaddr("127.0.0.1") == ("localhost", [], ["127.0.0.1"])
    with GUARD.expect_blocked(), pytest.raises(IsolationError):
        _socket.gethostbyaddr("127.0.0.1")


def test_unix_socket_connection_is_not_a_loopback_exception():
    with socket.socket(socket.AF_UNIX) as client:
        with GUARD.expect_blocked(), pytest.raises(IsolationError):
            client.connect("/scratch/nonexistent.sock")


def test_network_bind_and_udp_are_restricted():
    with socket.socket() as listener:
        with GUARD.expect_blocked(), pytest.raises(IsolationError):
            listener.bind(("0.0.0.0", 0))
    with GUARD.expect_blocked(), pytest.raises(IsolationError):
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


@pytest.mark.parametrize("command, kwargs", [
    (["/bin/sh", "-c", "true"], {}),
    ([sys.executable, "-c", "pass"], {"shell": True}),
    ([sys.executable, "-c", "pass"], {"pass_fds": (1,)}),
    ([sys.executable, "-I", "-c", "pass"], {}),
])
def test_unsupported_subprocesses_are_blocked(command, kwargs):
    with GUARD.expect_blocked(), pytest.raises(IsolationError):
        subprocess.run(command, **kwargs)


def test_direct_process_and_native_launches_are_blocked():
    for launch in (lambda: os.system("true"), lambda: os.fork(),
                   lambda: os.posix_spawn(sys.executable, [sys.executable], {}),
                   lambda: ctypes.CDLL(None)):
        with GUARD.expect_blocked(), pytest.raises(IsolationError):
            launch()


def test_synthetic_environment_and_namespace_are_established():
    assert os.getpid() == 1
    assert os.environ["HOME"] == "/scratch/home"
    assert os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert not any(name in os.environ for name in (
        "OPENAI_API_KEY", "VPS_GATEWAY_API_KEY", "EXA_API_KEY", "PI_SESSION_ID"
    ))
    assert not Path("/repo/.env").exists()
    assert not Path("/repo/config/router_config.yaml").exists()
    assert not Path("/repo/data").exists()
    assert not Path("/home").exists()
    assert os.statvfs("/repo").f_flag & os.ST_RDONLY
    assert not os.statvfs("/scratch").f_flag & os.ST_RDONLY
    assert not any(name.startswith(("proxy_app", "src.proxy_app", "rotator_library"))
                   for name in sys.modules)


def test_permitted_disposable_database_and_python_child(tmp_path):
    database = tmp_path / "synthetic.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE fixture (value TEXT)")
        connection.execute("INSERT INTO fixture VALUES (?)", ("synthetic",))
        row = connection.execute("SELECT value FROM fixture").fetchone()
        assert row == ("synthetic",)
    result = subprocess.run([
        sys.executable, "-c",
        "import os; print(os.environ['HOME']); print('child permitted')",
    ], capture_output=True, text=True, timeout=10)
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute("SELECT value FROM fixture").fetchone()
        assert row == ("synthetic",)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "/scratch/home\nchild permitted\n"


def test_python_child_has_guard_and_no_site_startup():
    code = """
import sys
from financial_test_bootstrap import GUARD, IsolationError
assert sys.flags.isolated and sys.flags.no_site
with GUARD.expect_blocked():
    try:
        open('/repo/.env')
    except IsolationError:
        pass
    else:
        raise AssertionError('unguarded child')
print('child denial verified')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "child denial verified\n"


def test_swallowed_child_violation_still_fails():
    code = """
import socket, sys
try:
    socket.create_connection(('127.0.0.1', 8000))
except PermissionError:
    pass
sys.exit(0)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 1
    assert "unexpected isolation denial" in result.stderr


def test_registered_http_fixture_and_child_then_revocation():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"synthetic response")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    address = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert address in GUARD.active_endpoints()
        code = (
            "from urllib.request import urlopen; "
            f"print(urlopen('http://127.0.0.1:{address[1]}').read().decode())"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert result.stdout == "synthetic response\n"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert address not in GUARD.active_endpoints()
    with GUARD.expect_blocked(), pytest.raises(IsolationError):
        socket.create_connection(address)


def test_unregistered_owned_socket_is_not_automatically_permitted():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with GUARD.expect_blocked(), pytest.raises(IsolationError):
            socket.create_connection(listener.getsockname())
        GUARD.register_endpoint(listener)
        with socket.create_connection(listener.getsockname(), timeout=1):
            pass
        with socket.create_connection(
            ("localhost", listener.getsockname()[1]), timeout=1
        ):
            pass


def test_symlink_and_descriptor_targets_are_revalidated(tmp_path):
    source = Path("/repo/tests/test_financial_test_isolation.py")
    link = tmp_path / "source.py"
    link.symlink_to(source)
    assert link.read_text() == source.read_text()
    with GUARD.expect_blocked(), pytest.raises(IsolationError):
        link.write_text("must not overwrite snapshot")
    descriptor = os.open(source, os.O_RDONLY)
    try:
        with GUARD.expect_blocked(), pytest.raises(IsolationError):
            os.fdopen(descriptor, "wb", closefd=False)
    finally:
        os.close(descriptor)
    with GUARD.expect_blocked(), pytest.raises(IsolationError):
        os.open("/repo/.env", os.O_RDONLY)


def test_relative_symlink_and_synthetic_secret_fd_are_permitted(tmp_path):
    source = tmp_path / "synthetic-secret"
    source.write_text("fixture only")
    link = tmp_path / "alias"
    link.symlink_to(source.name)
    descriptor = os.open(link, os.O_RDONLY)
    with os.fdopen(descriptor, "r") as stream:
        assert stream.read() == "fixture only"


def test_denial_is_enforced_in_other_threads():
    failures = []

    def worker():
        try:
            with GUARD.expect_blocked(), pytest.raises(IsolationError):
                socket.create_connection(("127.0.0.1", 8000))
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive() and not failures


def test_child_cannot_replace_environment_or_disable_bootstrap(tmp_path):
    code = """
import os, sys
assert sys.flags.isolated and sys.flags.no_site
assert os.environ['HOME'] == '/scratch/home'
assert 'LD_PRELOAD' not in os.environ
assert '/host' not in sys.path
print('environment constrained')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=10, env={
                                "HOME": "/host", "PYTHONPATH": "/host",
                                "LD_PRELOAD": "/host/library.so",
                            })
    assert result.returncode == 0, result.stderr
    assert result.stdout == "environment constrained\n"


def test_original_popen_cannot_skip_child_bootstrap():
    original = subprocess.Popen.__mro__[1]
    with GUARD.expect_blocked(), pytest.raises(IsolationError):
        original([sys.executable, "-c", "pass"])


def test_unexpected_access_is_sticky_even_if_exception_is_swallowed():
    child = """
from financial_test_bootstrap import GUARD
try:
    open('/repo/.env')
except PermissionError:
    pass
assert GUARD.unexpected == 1
"""
    result = subprocess.run([sys.executable, "-c", child], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 1
    assert 'unexpected isolation denial' in result.stderr
