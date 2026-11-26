"""Provider-free pytest runner. Invoke with an existing Python using -I -S.

Linux namespaces/chroot protect host resources even in native code and children.
The audit guard adds precise Python I/O denial and disposable-endpoint admission.
This is test infrastructure, not production financial admission or a hostile-code VM.
"""

from __future__ import annotations

import argparse
import atexit
from contextlib import contextmanager
import ctypes
import errno
import json
import os
from pathlib import Path
import runpy
import shutil
import socket
import socketserver
import stat
import subprocess
import sys
import tempfile
import threading
from typing import Any, Iterator
import weakref
from urllib.parse import parse_qsl, unquote, urlsplit


LEGACY_TESTS = (
    "test_routing_integrity_369.py",
    "test_alias_stream_fallback.py",
    "test_counter_persistence_377.py",
    "test_quota_transient_429.py",
    "test_task_telemetry.py",
)
BOOTSTRAP = "/repo/tests/support/financial_test_bootstrap.py"
SAFE_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/scratch/home",
    "USER": "test",
    "LOGNAME": "test",
    "TMPDIR": "/scratch/tmp",
    "XDG_CONFIG_HOME": "/scratch/home/config",
    "XDG_CACHE_HOME": "/scratch/home/cache",
    "XDG_DATA_HOME": "/scratch/home/data",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    "SKIP_LAUNCHER": "true",
    "SKIP_REAL_VIRTUAL_MODELS": "true",
    "SKIP_REAL_DEAD_PROVIDERS": "true",
    "TELEMETRY_DB_PATH": "/scratch/passive-telemetry.db",
    "TASK_TELEMETRY_DB_PATH": "/scratch/task-telemetry.db",
    "PENALTY_DB_PATH": "/scratch/penalty.db",
}


class IsolationError(PermissionError):
    """A forbidden test operation was rejected before its side effect."""


class Guard:
    """Fail-closed Python I/O policy installed before third-party imports."""

    def __init__(self, inherited_endpoints: list[list] | None = None) -> None:
        self.endpoints: weakref.WeakValueDictionary[
            tuple[str, int], socket.socket
        ] = weakref.WeakValueDictionary()
        self.inherited_endpoints = {tuple(row) for row in inherited_endpoints or []}
        self.local = threading.local()
        self.unexpected = 0
        self.expected = 0
        self.denial_sites: set[tuple[str, str, int]] = set()

    def deny(self, category: str) -> None:
        """Record even swallowed violations, without logging paths or credentials."""
        if getattr(self.local, "expecting", False):
            self.expected += 1
        else:
            self.unexpected += 1
            frame = sys._getframe(2)
            if len(self.denial_sites) < 20:
                self.denial_sites.add((category, frame.f_code.co_filename,
                                       frame.f_lineno))
        raise IsolationError(errno.EACCES, "test isolation: " + category)

    @contextmanager
    def expect_blocked(self) -> Iterator[None]:
        """Only negative self-tests should acknowledge an intentional denial."""
        before = self.expected
        old = getattr(self.local, "expecting", False)
        self.local.expecting = True
        try:
            yield
        finally:
            self.local.expecting = old
            if self.expected == before:
                raise AssertionError("negative self-test did not reach the guard")

    def path(self, value: object, *, writing: bool = False) -> None:
        """Validate Python file/SQLite access; the chroot is the native backstop."""
        if isinstance(value, int):
            # subprocess/socket.makefile wrap anonymous pipes and owned sockets.
            mode = os.fstat(value).st_mode
            if value not in (0, 1, 2) and not (
                stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)
            ):
                # Revalidate the live descriptor target, not a cached fd number.
                target = os.readlink(f"/proc/self/fd/{value}")
                if target.endswith(" (deleted)"):
                    target = target.removesuffix(" (deleted)")
                self.path(target, writing=writing)
            return
        raw = os.fsdecode(value)
        absolute = os.path.abspath(raw)
        resolved = os.path.realpath(absolute)
        parts = Path(resolved).parts
        disposable = resolved == "/scratch" or resolved.startswith("/scratch/")
        if (
            any(part == ".env" or part.startswith(".env.") for part in parts)
            or (not disposable and any(
                part in {".ssh", ".secrets", ".aws", ".gnupg"} for part in parts
            ))
            or any(path.startswith(("/repo/data/", "/repo/logs/", "/dev/shm/"))
                   or path in {"/repo/data", "/repo/logs", "/dev/shm"}
                   for path in (absolute, resolved))
            or any(path in {
                "/tmp/llm_proxy_telemetry.db", "/tmp/llm_task_telemetry.db",
                "/scratch/tmp/llm_proxy_telemetry.db",
                "/scratch/tmp/llm_task_telemetry.db",
            } for path in (absolute, resolved))
        ):
            self.deny("credentials or default persistent storage")
        roots = ("/scratch",) if writing else (
            "/scratch", "/repo", "/site-packages", "/usr/lib", "/lib", "/lib64"
        )
        devices = {"/dev/null", "/dev/random", "/dev/urandom"}
        if resolved in devices or any(
            resolved == root or resolved.startswith(root + "/") for root in roots
        ):
            return
        self.deny("filesystem write" if writing else "filesystem read")

    def active_endpoints(self) -> set[tuple[str, int]]:
        """Closed listeners are removed from admission, including port reuse."""
        active = set(self.inherited_endpoints)
        for address, listener in list(self.endpoints.items()):
            try:
                if listener.fileno() >= 0 and listener.getsockname() == address:
                    active.add(address)
            except OSError:
                pass  # Another test thread may have just closed the listener.
        return active

    def destination(self, address: object) -> None:
        """No DNS or blanket loopback exception: exact registered TCP endpoints."""
        if not isinstance(address, tuple) or len(address) != 2:
            self.deny("unregistered network destination")
        host, port = address
        if host == "localhost":
            host = "127.0.0.1"
        if (host, port) not in self.active_endpoints():
            self.deny("unregistered network destination")

    def register_endpoint(self, listener: socket.socket) -> None:
        """Admit an owned, listening loopback TCP socket created in the sandbox."""
        address = listener.getsockname()
        if (
            listener.family != socket.AF_INET
            or listener.type != socket.SOCK_STREAM
            or address[0] != "127.0.0.1"
            or not listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)
        ):
            self.deny("endpoint is not a disposable TCP listener")
        self.endpoints[address] = listener

    def audit(self, event: str, args: tuple) -> None:
        """Cover low-level Python APIs as well as ordinary client libraries."""
        if event == "open":
            self.path(args[0], writing=bool(args[2] & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            )))
        elif event == "sqlite3.connect":
            if args[0] != ":memory:":
                database = args[0]
                if str(database).startswith("file:"):
                    uri = urlsplit(str(database))
                    options = parse_qsl(uri.query, keep_blank_values=True)
                    if (uri.netloc or uri.fragment or len(options) != 1
                            or options[0][0] != "mode"
                            or options[0][1] not in {"ro", "rw", "rwc"}):
                        self.deny("unsupported SQLite URI")
                    database = unquote(uri.path)
                self.path(database, writing=True)
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.truncate"}:
            self.path(args[0], writing=True)
            if len(args) > 1 and event in {"os.remove", "os.rmdir", "os.mkdir"}:
                if args[-1] not in (-1, None):
                    self.deny("directory-relative mutation")
        elif event in {"os.rename", "os.link", "os.symlink"}:
            if event == "os.symlink":
                target = os.path.join(os.path.dirname(os.fspath(args[1])), args[0])
                self.path(target)
            else:
                self.path(args[0], writing=True)
            self.path(args[1], writing=True)
            if any(value not in (-1, None) for value in args[2:]):
                self.deny("directory-relative mutation")
        elif event == "socket.__new__":
            # family=-1 reconstructs an already-owned descriptor; it cannot dial.
            if args[1:4] == (-1, -1, -1):
                return
            if args[1] not in {socket.AF_INET, socket.AF_INET6, socket.AF_UNIX}:
                self.deny("unsupported socket family")
            if args[1] != socket.AF_UNIX and args[2] != socket.SOCK_STREAM:
                self.deny("non-TCP network socket")
        elif event in {"socket.connect", "socket.getaddrinfo"}:
            self.destination(args[1] if event == "socket.connect" else args[:2])
        elif event == "socket.bind":
            if args[1] != ("127.0.0.1", 0):
                self.deny("non-disposable listener bind")
        elif event in {"socket.sendto", "socket.sendmsg", "socket.gethostbyname"}:
            self.deny("unregistered network operation")
        elif event == "socket.gethostbyaddr":
            self.deny("DNS lookup")
        elif event == "subprocess.Popen":
            if not getattr(self.local, "launching", False):
                self.deny("unbootstrapped subprocess")
        elif event in {
            "os.system", "os.exec", "os.posix_spawn", "os.fork", "os.forkpty",
            "ctypes.dlopen", "ctypes.dlsym", "ctypes.call_function",
        }:
            self.deny("unguarded process or native-code launch")

    def install(self) -> None:
        """Install once, before dotenv, pytest, conftests or application modules."""
        original_activate = socketserver.TCPServer.server_activate
        original_getaddrinfo = socket.getaddrinfo
        guard = self

        def getaddrinfo(
            host: str, port: int, *args: Any, **kwargs: Any
        ) -> list[tuple]:
            guard.destination((host, port))
            # Avoid even a libc DNS attempt for an admitted localhost spelling.
            if host == "localhost":
                host = "127.0.0.1"
            return original_getaddrinfo(host, port, *args, **kwargs)

        socket.getaddrinfo = getaddrinfo

        def gethostbyaddr(host: str) -> tuple[str, list[str], list[str]]:
            # TCPServer.getfqdn must not delegate reverse lookup to libc/DNS.
            if host not in {"127.0.0.1", "::1"}:
                guard.deny("DNS lookup")
            return "localhost", [], [host]

        socket.gethostbyaddr = gethostbyaddr

        def activate(server: socketserver.TCPServer) -> None:
            original_activate(server)
            # This explicit adapter registers only successfully created listeners.
            guard.register_endpoint(server.socket)

        socketserver.TCPServer.server_activate = activate
        original_popen = subprocess.Popen

        class GuardedPopen(original_popen):
            def __init__(self, args: list | tuple, **kwargs: Any) -> None:
                if (
                    not isinstance(args, (list, tuple)) or len(args) < 2
                    or args[0] != sys.executable or kwargs.get("shell")
                    or kwargs.get("executable") or kwargs.get("pass_fds")
                    or kwargs.get("preexec_fn")
                ):
                    guard.deny("unsupported subprocess")
                if args[1] != "-c":
                    guard.path(args[1])
                    if not str(args[1]).endswith(".py"):
                        guard.deny("unsupported Python invocation")
                if kwargs.get("cwd") is not None:
                    guard.path(kwargs["cwd"])
                environment = dict(kwargs.pop("env", os.environ))
                for key in list(environment):
                    if key.startswith(("LD_", "PYTHON")):
                        environment.pop(key)
                environment.update(SAFE_ENV)
                environment["FINANCIAL_TEST_ENDPOINTS"] = json.dumps(
                    sorted(guard.active_endpoints())
                )
                kwargs.setdefault("stdin", subprocess.DEVNULL)
                kwargs["close_fds"] = True
                guard.local.launching = True
                try:
                    super().__init__(
                        [sys.executable, "-I", "-S", BOOTSTRAP, "--child", *args[1:]],
                        env=environment, **kwargs
                    )
                finally:
                    guard.local.launching = False

        subprocess.Popen = GuardedPopen
        sys.addaudithook(self.audit)


GUARD: Guard | None = None


def _mount(source: str | None, target: Path, kind: str | None, flags: int,
           data: str | None = None) -> None:
    """Use the installed Linux namespace/mount facilities, no new dependency."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                          ctypes.c_ulong, ctypes.c_char_p]
    encoded = [value.encode() if value is not None else None
               for value in (source, str(target), kind)]
    if libc.mount(*encoded, flags, data.encode() if data else None) != 0:
        raise OSError(ctypes.get_errno(), "sandbox mount failed")


def _bind_readonly(source: Path, root: Path, destination: str) -> None:
    target = root / destination.lstrip("/")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        target.mkdir(exist_ok=True)
    else:
        target.touch()
    _mount(str(source), target, None, 4096)  # MS_BIND
    _mount(None, target, None, 4096 | 32 | 1 | 2)  # remount, readonly, nosuid


def _prepare_root(root: Path, repo: Path, packages: Path, parent_net: int) -> None:
    """Expose code/dependencies only; host configs, homes and storage are absent."""
    if os.getpid() != 1 or os.stat("/proc/self/ns/net").st_ino == parent_net:
        raise RuntimeError("fresh network/PID namespace required")
    _mount(None, Path("/"), None, 16384 | (1 << 18))  # recursive private mounts
    _mount("tmpfs", root, "tmpfs", 2 | 4, "size=256m,mode=0700")
    for directory in ("src", "tests", "workload-router"):
        for source in (repo / directory).rglob("*.py"):
            if source.is_symlink() or any(part.startswith(".") for part in
                                          source.relative_to(repo).parts):
                continue
            destination = root / "repo" / source.relative_to(repo)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    # Pytest config is plain text; provider/config/data files are never copied.
    shutil.copyfile(repo / "pytest.ini", root / "repo/pytest.ini")
    # Prevent the legacy credit helper falling back to a host-specific catalog.
    (root / "repo/config").mkdir()
    (root / "repo/config/providers_database.yaml").write_text("providers: {}\n")
    python = Path(sys.executable).resolve()
    _bind_readonly(python, root, "/usr/bin/python3.12")
    for source, destination in (
        (Path(sys.base_prefix) / "lib/python3.12", "/usr/lib/python3.12"),
        (Path("/usr/lib/x86_64-linux-gnu"), "/usr/lib/x86_64-linux-gnu"),
        (Path("/lib/x86_64-linux-gnu"), "/lib/x86_64-linux-gnu"),
        (Path("/lib64/ld-linux-x86-64.so.2"), "/lib64/ld-linux-x86-64.so.2"),
        (packages, "/site-packages"),
    ):
        _bind_readonly(source, root, destination)
    for device in ("null", "random", "urandom"):
        _bind_readonly(Path("/dev") / device, root, "/dev/" + device)
    (root / "scratch").mkdir()
    _mount("tmpfs", root / "scratch", "tmpfs", 2 | 4, "size=256m,mode=0700")
    for name in ("home", "tmp", "pytest"):
        (root / "scratch" / name).mkdir()
    (root / "tmp").symlink_to("/scratch/tmp")
    (root / "proc").mkdir()
    _mount("proc", root / "proc", "proc", 1 | 2 | 4 | 8)
    manifest = {"parent_net": parent_net,
                "net": os.stat("/proc/self/ns/net").st_ino}
    (root / "isolation.json").write_text(json.dumps(manifest))
    # Bring up only private loopback, before dropping all setup privileges.
    subprocess.run(["/usr/sbin/ip", "link", "set", "lo", "up"], check=True)
    _mount(None, root, "tmpfs", 32 | 1 | 2 | 4)
    os.chroot(root)
    os.chdir("/repo")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "no-new-privileges failed")
    # Drop user-namespace root's capabilities; it cannot undo the readonly root.
    header = (ctypes.c_uint32 * 2)(0x20080522, 0)
    data = (ctypes.c_uint32 * 6)()
    if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        raise OSError(ctypes.get_errno(), "capability drop failed")


def _install_guard(*, child: bool = False) -> Guard:
    global GUARD
    manifest = json.loads(Path("/isolation.json").read_text())
    current_net = os.stat("/proc/self/ns/net").st_ino
    if current_net != manifest["net"] or current_net == manifest["parent_net"]:
        raise RuntimeError("sandbox identity mismatch")
    sys.dont_write_bytecode = True
    sys.executable = "/usr/bin/python3.12"
    sys.path[:0] = ["/repo/tests/support", "/repo", "/repo/src",
                   "/repo/workload-router", "/site-packages"]
    inherited = (
        json.loads(os.environ.get("FINANCIAL_TEST_ENDPOINTS", "[]")) if child else []
    )
    guard = Guard(inherited)
    guard.install()
    GUARD = guard
    sys.modules["financial_test_bootstrap"] = sys.modules[__name__]

    def final_check() -> None:
        # Run after later import-time exit handlers, including child sys.exit(0).
        if guard.unexpected:
            print("unexpected isolation denial", file=sys.stderr, flush=True)
            os._exit(1)

    atexit.register(final_check)
    # LiteLLM calls dotenv at import. Suppress that, not the financial logic.
    import dotenv
    dotenv.load_dotenv = lambda *args, **kwargs: False
    return guard


def _run_tests(suite: str) -> int:
    os.environ["FINANCIAL_TEST_SUITE"] = suite
    guard = _install_guard()
    import pytest
    if suite == "sidecar":
        os.chdir("/repo/workload-router")
        paths = ["tests"]
    elif suite == "legacy":
        paths = ["tests/" + name for name in LEGACY_TESTS]
    elif suite == "financial":
        paths = [
            "tests/test_financial_test_isolation.py",
            "tests/test_financial_safety.py",
        ]
    else:
        paths = ["tests/test_financial_test_isolation.py"]
    # No plugin discovery, persisted pytest cache, or host temporary base.
    code = pytest.main([*paths, "-q", "-o", "addopts=", "-p", "no:cacheprovider",
                        "-p", "pytest_asyncio.plugin", "--basetemp=/scratch/pytest"])
    print(f"isolation: unexpected_denials={guard.unexpected} "
          f"expected_denials={guard.expected}")
    if guard.denial_sites:
        print("isolation denial sites: " + json.dumps(sorted(guard.denial_sites)))
    return 1 if guard.unexpected else int(code)


def main() -> int:
    """Run a fixed local suite, or a guarded Python child inside the same sandbox."""
    sys.dont_write_bytecode = True
    if not sys.flags.isolated or not sys.flags.no_site:
        raise RuntimeError("invoke Python with -I -S; unsafe startup is rejected")
    if sys.argv[1:2] == ["--child"]:
        guard = _install_guard(child=True)
        sys.argv = sys.argv[2:]
        if sys.argv[0] == "-c":
            code = sys.argv[1]
            sys.argv = ["-c", *sys.argv[2:]]
            exec(compile(code, "<synthetic-child>", "exec"), {"__name__": "__main__"})
        else:
            runpy.run_path(sys.argv[0], run_name="__main__")
        return 1 if guard.unexpected else 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",
                        choices=("isolation", "legacy", "sidecar", "financial"),
                        default="isolation")
    parser.add_argument("--worker", nargs=4, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ.clear()
    os.environ.update(SAFE_ENV)
    os.umask(0o077)
    if args.worker:
        root, repo, packages, parent_net = args.worker
        _prepare_root(Path(root), Path(repo), Path(packages), int(parent_net))
        # Reload from the snapshot so tracebacks never reopen host source paths.
        sys.path.insert(0, "/repo/tests/support")
        import financial_test_bootstrap
        return financial_test_bootstrap._run_tests(args.suite)
    if (sys.platform != "linux" or sys.version_info[:2] != (3, 12)
            or os.uname().machine != "x86_64"):
        raise RuntimeError("this verifier requires Linux x86_64 and Python 3.12")
    repo = Path(__file__).resolve().parents[2]
    packages = repo / "venv/lib/python3.12/site-packages"
    if not packages.is_dir() or not Path("/usr/bin/unshare").exists():
        raise RuntimeError("existing venv and unshare required; nothing is installed")
    parent_net = os.stat("/proc/self/ns/net").st_ino
    # Only this owned empty host directory is created; test state lives in tmpfs.
    with tempfile.TemporaryDirectory(
        prefix="financial-tests-", dir="/tmp"
    ) as directory:
        return subprocess.call([
            "/usr/bin/unshare", "--user", "--map-root-user", "--mount", "--net",
            "--pid", "--fork", "--kill-child", sys.executable, "-I", "-S",
            str(Path(__file__).resolve()), "--suite", args.suite, "--worker",
            directory, str(repo), str(packages), str(parent_net),
        ], stdin=subprocess.DEVNULL, close_fds=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print("financial test verifier failed: " + str(error), file=sys.stderr)
        raise SystemExit(1)
