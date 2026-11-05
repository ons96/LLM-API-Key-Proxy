"""Budgeted live free-provider checks through a disposable New-API instance.

Default: catalog preflight only. --execute allows at most two 128-token calls
per target (four total); --opencode permits one target and three 512-token calls.
--failover injects local 429s and allows one 128-token fallback per provider.
Existing keys stay in the parent process; an inherited
socket feeds a loopback bridge inside a private network namespace. New-API stores
only generated bridge credentials, never real provider keys. No VPS is accessed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import hmac
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import sqlite3
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from urllib import error, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from newapi_benchmark import collect
from newapi_catalog import (
    CatalogError,
    _request,
    load_provider_metadata,
    valid_model_id,
)
from newapi_runtime import Runtime, prepare, strict_json, validate_key
from newapi_server import Server
from provider_adapter import ProviderError, _NoRedirect
from quota_ledger import QuotaLedger, Route
from router_state import RouterState
from tools.verify_newapi import (
    bootstrap,
    enroll_local_proof,
    fixture_command,
    private_namespace,
    relay_token,
    running,
)
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_FRAME = 1024 * 1024
MAX_REQUEST_BYTES = 16384
OUTPUT_CAP = 128
CALLS_PER_TARGET = 2
TOOL_OUTPUT_CAP = 512
TOOL_CALLS = 3
SUPPORTED_PROVIDERS = {"groq", "openrouter", "mistral"}


@dataclass
class Target:
    """One explicitly selected, currently verified free route; key stays in memory."""

    provider: str
    model: str
    endpoint: str
    key: str = field(repr=False)
    context: int
    rpm: int

    def public(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "context": self.context,
            "rpm": self.rpm,
        }


def private_key(path: Path) -> str:
    """Read only a bounded owner-only regular file under the existing secret store."""
    path = path.expanduser()
    if not path.resolve().is_relative_to(Path.home() / ".secrets"):
        raise ValueError("key file must stay in the existing private secret store")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
            or info.st_size > 4096
        ):
            raise ValueError("key file must be owner-only, regular and bounded")
        return validate_key(stream.read(4097).decode("ascii").strip())


def free_model(provider: str, model: str, catalog: list[dict], canonical: dict) -> dict:
    """Fail closed on unlisted, paid, unknown-context or non-tool models."""
    matches = [
        row for row in catalog if isinstance(row, dict) and row.get("id") == model
    ]
    if len(matches) != 1:
        raise ValueError("selected model is absent or ambiguous in the live catalog")
    row = matches[0]
    if provider == "groq":
        if canonical.get("free_tier") != 1 or "tools" not in str(
            canonical.get("capabilities", "")
        ):
            raise ValueError("Groq model needs canonical free/tool metadata")
        context = canonical.get("context_window")
        if row.get("active", True) is not True:
            raise ValueError("selected Groq model is inactive")
    elif provider == "openrouter":
        prices = row.get("pricing")
        parameters = row.get("supported_parameters")
        if not model.endswith(":free") or not isinstance(prices, dict) or not prices:
            raise ValueError(
                "OpenRouter requires an explicit free model and live prices"
            )
        try:
            if any(
                float(prices.get(name, "nan")) != 0 for name in ("prompt", "completion")
            ) or any(float(value) != 0 for value in prices.values()):
                raise ValueError("selected model has nonzero or unknown live pricing")
        except (TypeError, ValueError):
            raise ValueError(
                "selected model has nonzero or unknown live pricing"
            ) from None
        if (
            not isinstance(parameters, list)
            or any(not isinstance(value, str) for value in parameters)
            or not {"tools", "tool_choice"} <= set(parameters)
        ):
            raise ValueError("selected model has no verified tool support")
        context = row.get("context_length")
    elif provider == "mistral":
        capabilities = row.get("capabilities")
        if (
            canonical.get("free_tier") != 1
            or "tools" not in str(canonical.get("capabilities", ""))
            or not isinstance(capabilities, dict)
            or capabilities.get("completion_chat") is not True
            or capabilities.get("function_calling") is not True
            or row.get("deprecation") is not None
        ):
            raise ValueError(
                "Mistral needs canonical free/tools and active live chat/tool support"
            )
        context = row.get("max_context_length")
        stored_context = canonical.get("context_window")
        if type(stored_context) is not int or stored_context <= OUTPUT_CAP:
            raise ValueError("canonical Mistral context is required")
        if type(context) is int:
            context = min(context, stored_context)
    else:
        raise ValueError("provider needs an approved free-only validation policy")
    if type(context) is not int or context <= OUTPUT_CAP:
        raise ValueError("verified model context is required")
    return {"context": context}


def load_targets(database: Path, plan: Path) -> tuple[list[Target], list[dict]]:
    """Catalog calls only: resolve canonical endpoints and private file references."""
    if not plan.is_file() or plan.stat().st_size > MAX_REQUEST_BYTES:
        raise ValueError("target plan must be a bounded file")
    raw = strict_json(plan.read_bytes())
    rows = raw.get("targets")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 2:
        raise ValueError("select one or two explicit targets")
    names = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row)
            not in ({"provider", "model", "key_file"}, {"provider", "model", "key_env"})
            or not isinstance(row["provider"], str)
            or row["provider"] not in SUPPORTED_PROVIDERS
            or not valid_model_id(row["model"])
            or not isinstance(row.get("key_file", row.get("key_env")), str)
        ):
            raise ValueError(
                "targets require an approved provider, model and one private key source"
            )
        names.append(row["provider"])
    if len(set(names)) != len(names):
        raise ValueError("one target per provider is required")
    metadata = {row["name"]: row for row in load_provider_metadata(database, names)}
    connection = sqlite3.connect(
        database.expanduser().resolve().as_uri() + "?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    targets, reports = [], []
    opener = request.build_opener(request.ProxyHandler({}), _NoRedirect())
    try:
        for row in rows:
            provider = row["provider"]
            record = metadata[provider]
            report = {"provider": provider, "model": row["model"], "catalog_only": True}
            try:
                funding = connection.execute(
                    "SELECT enabled, free_tier, free_one_time, rate_limit_rpm FROM providers WHERE key_name=?",
                    (provider,),
                ).fetchone()
                if (
                    not funding
                    or funding["enabled"] != 1
                    or funding["free_tier"] != 1
                    or funding["free_one_time"]
                ):
                    raise ValueError(
                        "enabled renewable free provider metadata required"
                    )
                if not record["base_url"].startswith("https://"):
                    raise ValueError("live provider endpoints must use HTTPS")
                if "key_env" in row:
                    if row["key_env"] != record["env_var"]:
                        raise ValueError(
                            "key_env must match the canonical credential variable"
                        )
                    key = validate_key(os.environ.get(row["key_env"], ""))
                else:
                    key = private_key(Path(row["key_file"]))
                data = _request(
                    opener,
                    record["base_url"] + "/models",
                    "GET",
                    {"Authorization": "Bearer " + key},
                )
                catalog = data.get("data")
                if not isinstance(catalog, list):
                    raise ValueError("live model catalog must be a list")
                stored = connection.execute(
                    "SELECT m.context_window, m.free_tier, m.capabilities FROM models m "
                    "JOIN providers p ON m.provider_id=p.id WHERE p.key_name=? AND m.model_id=?",
                    (provider, row["model"]),
                ).fetchone()
                verified = free_model(
                    provider, row["model"], catalog, dict(stored) if stored else {}
                )
                rpm = funding["rate_limit_rpm"]
                if type(rpm) is not int or rpm <= 0:
                    raise ValueError("canonical request-rate bound required")
                target = Target(
                    provider,
                    row["model"],
                    record["base_url"],
                    key,
                    verified["context"],
                    min(rpm, 4),
                )
                targets.append(target)
                report.update(
                    ok=True, catalog_models=len(catalog), context=target.context
                )
            except (CatalogError, OSError, ValueError) as exc:
                report.update(ok=False, error=str(exc))
            reports.append(report)
    finally:
        connection.close()
    return targets, reports


def frame_send(channel: socket.socket, data: dict) -> None:
    """Write one bounded JSON message on the inherited local socket."""
    raw = json.dumps(data, allow_nan=False).encode()
    if len(raw) > MAX_FRAME:
        raise ValueError("bridge frame exceeds limit")
    channel.sendall(struct.pack("!I", len(raw)) + raw)


def frame_read(channel: socket.socket) -> dict:
    """Read exactly one bounded JSON object without accepting oversized frames."""

    def exact(length: int) -> bytes:
        pieces = bytearray()
        while len(pieces) < length:
            block = channel.recv(length - len(pieces))
            if not block:
                raise EOFError
            pieces.extend(block)
        return bytes(pieces)

    length = struct.unpack("!I", exact(4))[0]
    if length > MAX_FRAME:
        raise ValueError("bridge frame exceeds limit")
    return strict_json(exact(length))


class Broker:
    """Only allow selected models, bounded calls and output; never retry a target error."""

    def __init__(
        self, targets: list[Target], *, tool_loop: bool = False, failover: bool = False
    ) -> None:
        self.targets = {row.provider: row for row in targets}
        self.output_cap = TOOL_OUTPUT_CAP if tool_loop else OUTPUT_CAP
        self.call_limit = (
            1 if failover else TOOL_CALLS if tool_loop else CALLS_PER_TARGET
        )
        self.body_limit = 65536 if tool_loop else MAX_REQUEST_BYTES
        self.calls = {row.provider: 0 for row in targets}
        self.blocked: set[str] = set()
        self.events: list[dict] = []
        self.opener = request.build_opener(request.ProxyHandler({}), _NoRedirect())

    def dispatch(self, channel: socket.socket, message: dict) -> None:
        name, payload = message.get("provider"), message.get("payload")
        if (
            not isinstance(name, str)
            or name not in self.targets
            or not isinstance(payload, dict)
        ):
            raise ValueError("bridge target is not allowlisted")
        target = self.targets[name]
        limits: list[int] = [
            payload[key]
            for key in ("max_tokens", "max_completion_tokens")
            if key in payload
        ]
        if (
            payload.get("model") != target.model
            or not limits
            or any(type(n) is not int or not 1 <= n <= self.output_cap for n in limits)
            or type(payload.get("n", 1)) is not int
            or payload.get("n", 1) != 1
            or type(payload.get("best_of", 1)) is not int
            or payload.get("best_of", 1) != 1
            or not isinstance(payload.get("stream", False), bool)
            or len(json.dumps(payload, allow_nan=False).encode()) > self.body_limit
        ):
            raise ValueError("bridge request violates model/token/body budget")
        if name in self.blocked or self.calls[name] >= self.call_limit:
            frame_send(channel, {"status": 429, "content_type": "application/json"})
            frame_send(
                channel,
                {"body": '{"error":{"message":"live verification budget exhausted"}}'},
            )
            frame_send(channel, {"end": True})
            return
        self.calls[name] += 1
        started = time.monotonic()
        event = {
            "provider": name,
            "model": target.model,
            "output_cap": max(limits),
            "stream": payload.get("stream", False),
            "timestamp": time.time(),
        }
        self.events.append(event)
        req = request.Request(
            target.endpoint + "/chat/completions",
            json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + target.key,
            },
        )
        header_sent = False
        try:
            with self.opener.open(req, timeout=30) as response:
                content_type = response.headers.get("Content-Type", "")
                frame_send(
                    channel, {"status": response.status, "content_type": content_type}
                )
                header_sent = True
                total = 0
                while True:
                    raw = (
                        response.readline(65537)
                        if "event-stream" in content_type
                        else response.read(65537)
                    )
                    if not raw:
                        break
                    total += len(raw)
                    if (
                        len(raw) > 65536
                        or total > MAX_FRAME
                        or time.monotonic() - started > 40
                    ):
                        raise ValueError("live response exceeds byte/time budget")
                    if target.key.encode() in raw:
                        raise ValueError("live provider echoed its credential")
                    if (
                        not payload.get("stream", False)
                        and "event-stream" in content_type
                    ):
                        raise ValueError("live provider forced unrequested streaming")
                    frame_send(channel, {"body": raw.decode("utf-8")})
                event["status"] = response.status
        except error.HTTPError as exc:
            event.update(
                status=exc.code,
                error="rate_limit" if exc.code == 429 else "provider_rejection",
            )
            exc.close()
            self.blocked.add(name)
            frame_send(
                channel, {"status": exc.code, "content_type": "application/json"}
            )
            frame_send(
                channel,
                {"body": '{"error":{"message":"live provider rejected request"}}'},
            )
        except (OSError, error.URLError, ValueError):
            event.update(status=502, error="transport_or_response_failure")
            self.blocked.add(name)
            if not header_sent:
                frame_send(channel, {"status": 502, "content_type": "application/json"})
                frame_send(
                    channel,
                    {"body": '{"error":{"message":"live provider transport failure"}}'},
                )
        finally:
            event["latency_ms"] = (time.monotonic() - started) * 1000
            frame_send(channel, {"end": True})

    def serve(self, channel: socket.socket) -> None:
        while True:
            try:
                self.dispatch(channel, frame_read(channel))
            except ValueError:
                frame_send(channel, {"status": 400, "content_type": "application/json"})
                frame_send(
                    channel,
                    {
                        "body": '{"error":{"message":"invalid live verification request"}}'
                    },
                )
                frame_send(channel, {"end": True})
            except (EOFError, OSError):
                return


def live_opencode(directory: Path, executable: Path, runtime: Runtime) -> dict:
    """Allow only the generated read fixture and cap all real client generations."""
    home = directory / "client-home"
    home.mkdir()
    work = directory / "client-work"
    work.mkdir()
    # OpenCode uses "/" for non-Git worktrees; anchor the relative read permission.
    subprocess.run(
        ["git", "init", "--quiet", str(work)],
        check=True,
        env={"PATH": os.environ["PATH"], "HOME": str(home)},
    )
    fixture = work / "proof.txt"
    marker = "hello-newapi-live-proof"
    fixture.write_text(marker + "\n")
    key_file = directory / "client.key"
    key = secrets.token_urlsafe(24)
    key_file.write_text(key)
    key_file.chmod(0o600)
    models = directory / "models.json"
    models.write_text("{}")
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "OPENCODE_TEST_HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_DATA_HOME": str(home / ".local/share"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_STATE_HOME": str(home / ".local/state"),
        "OPENCODE_MODELS_PATH": str(models),
        "OPENCODE_DISABLE_MODELS_FETCH": "true",
        "OPENCODE_DISABLE_AUTOUPDATE": "true",
        "OPENCODE_DISABLE_DEFAULT_PLUGINS": "true",
        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "true",
        "OPENCODE_DISABLE_CLAUDE_CODE": "true",
        "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
        "OPENCODE_DISABLE_CHANNEL_DB": "true",
        "OPENCODE_DISABLE_LSP_DOWNLOAD": "true",
    }
    with running(
        Server(
            ("127.0.0.1", 0), runtime, key, workers=2, default_output=TOOL_OUTPUT_CAP
        )
    ) as server:
        config = directory / "client.json"
        config.write_text(
            json.dumps(
                {
                    "provider": {
                        "sidecar": {
                            "name": "Live free check",
                            "npm": "@ai-sdk/openai-compatible",
                            "options": {
                                "baseURL": f"http://127.0.0.1:{server.server_address[1]}/v1",
                                "apiKey": "{file:" + str(key_file) + "}",
                            },
                            "models": {
                                "auto-coding": {
                                    "name": "Free coding",
                                    "tool_call": True,
                                    "limit": {
                                        "context": runtime.ledger.routes()[
                                            0
                                        ].context_window,
                                        "output": TOOL_OUTPUT_CAP,
                                    },
                                }
                            },
                        }
                    },
                    "model": "sidecar/auto-coding",
                    "small_model": "sidecar/auto-coding",
                    "permission": {
                        "*": "deny",
                        "read": {"*": "deny", "proof.txt": "allow"},
                    },
                }
            )
        )
        env["OPENCODE_CONFIG"] = str(config)
        output = fixture_command(
            directory,
            env,
            "opencode-live-loop",
            [
                str(executable),
                "run",
                "--pure",
                "--dir",
                str(work),
                "--format",
                "json",
                "--model",
                "sidecar/auto-coding",
                "--title",
                "live-free-proof",
                "Use read to inspect "
                + str(fixture)
                + ". Reply with exactly its content. Do not use any other file or tool.",
            ],
        )
    events = [json.loads(line) for line in output.splitlines() if line.strip()]
    reads = [
        event
        for event in events
        if event.get("type") == "tool_use"
        and event.get("part", {}).get("tool") == "read"
        and event["part"].get("state", {}).get("status") == "completed"
    ]
    exact_read = len(reads) == 1 and reads[0]["part"]["state"].get("input", {}).get(
        "filePath"
    ) == str(fixture)
    content_ok = any(
        event.get("type") == "text"
        and event.get("part", {}).get("text", "").strip() == marker
        for event in events
    )
    version = (
        fixture_command(
            directory, env, "opencode-version", [str(executable), "--version"]
        )
        .decode()
        .strip()
    )
    return {
        "version": version,
        "read_tool_completed": exact_read,
        "content_ok": content_ok,
        "read_allowlist": "single disposable file",
        "passed": exact_read and content_ok,
    }


def verify_failover(
    directory: Path,
    base: str,
    relay: str,
    routes: list[Route],
    bridge_faults: set[str],
    bridge_attempts: list[str],
) -> list[dict]:
    """Use fresh ledgers per direction; never clear quota or cooldown history."""
    if len(routes) != 2 or len({route.provider_id for route in routes}) != 2:
        raise ValueError("failover requires two verified provider routes")
    results = []
    for index, primary in enumerate(routes):
        path = directory / ("failover-" + primary.provider_id + ".db")
        candidates = [
            replace(route, latency_ms=1 if route == primary else 100000)
            for route in routes
        ]
        backup = next(
            route for route in candidates if route.provider_id != primary.provider_id
        )
        ledger, state = QuotaLedger(path), RouterState(path)
        payload = {
            "model": "auto-coding" if index else "auto-chat",
            "stream": bool(index),
            "max_tokens": OUTPUT_CAP,
            "messages": [{"role": "user", "content": "Reply with exactly ok."}],
        }
        prepared = prepare(payload)
        before = len(bridge_attempts)
        bridge_faults.add(primary.provider_id)
        try:
            for route in candidates:
                ledger.configure(route)
            runtime = Runtime(ledger, state, base + "/v1", relay, explore=False)
            if runtime.ranked(prepared)[0].route.provider_id != primary.provider_id:
                raise CatalogError("controlled primary route was not ranked first")
            if payload["stream"]:
                source = runtime.open_stream(prepared)
                completion = collect(source)
                identity = source.identity
            else:
                result = runtime.complete(prepared)
                completion, identity = result.completion, result.decision_id
            attempts = bridge_attempts[before:]
            if attempts != [primary.provider_id, backup.provider_id]:
                raise CatalogError("hidden retry or unexpected failover order")
            decision = ledger.decision(identity)
            failure = ledger.db.execute(
                "SELECT * FROM routing_decisions WHERE status='failure'"
            ).fetchone()
            if (
                failure is None
                or failure["error_class"] != "rate_limit"
                or failure["input_tokens"] != failure["reserved_input"]
                or failure["output_tokens"] != failure["reserved_output"]
            ):
                raise CatalogError("injected failure was not conservatively accounted")
            content = completion.content
            if (
                not isinstance(content, str)
                or content.strip().lower().strip(".!") != "ok"
                or decision["status"] != "success"
                or decision["usage_estimated"]
            ):
                raise CatalogError("live fallback content or final usage is missing")
            results.append(
                {
                    "primary": primary.provider_id,
                    "fallback": backup.provider_id,
                    "stream": payload["stream"],
                    "attempt_order": attempts,
                    "fault": "local_injected_429",
                    "hidden_retries": 0,
                    "failed_reservation_retained": True,
                    "input_tokens": decision["input_tokens"],
                    "output_tokens": decision["output_tokens"],
                }
            )
        finally:
            bridge_faults.discard(primary.provider_id)
            state.close()
            ledger.close()
        # A new connection must still see the failed route's quota and cooldown.
        restored = QuotaLedger(path)
        try:
            cooldown = restored.capacity(
                primary, prepared.work.input_tokens, OUTPUT_CAP
            )
            if cooldown.feasible or cooldown.reason != "cooldown":
                raise CatalogError(
                    "failover cooldown did not survive reopening the ledger"
                )
            persisted_failure = restored.decision(failure["id"])
            if persisted_failure != dict(failure):
                raise CatalogError(
                    "failed reservation did not survive reopening the ledger"
                )
            results[-1]["cooldown_persisted"] = True
            results[-1]["reservation_persisted"] = True
        finally:
            restored.close()
    return results


def child_verify(
    directory: Path,
    binary: Path,
    channel: socket.socket,
    parent_inode: int,
    opencode: Path | None = None,
    failover: bool = False,
) -> dict:
    """The gateway has no real credentials or external network access."""
    private_namespace(parent_inode)
    targets = frame_read(channel)["targets"]
    output_cap = TOOL_OUTPUT_CAP if opencode else OUTPUT_CAP
    call_limit = TOOL_CALLS if opencode else CALLS_PER_TARGET
    body_limit = 65536 if opencode else MAX_REQUEST_BYTES
    bridge_lock = threading.Lock()
    bridge_keys: dict[str, str] = {}
    bridge_faults: set[str] = set()
    bridge_attempts: list[str] = []

    class Bridge(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def do_POST(self):
            parts = self.path.split("/")
            if len(parts) != 5 or parts[2:] != ["v1", "chat", "completions"]:
                self.send_error(404)
                return
            values = self.headers.get_all("Authorization", [])
            expected = bridge_keys.get(parts[1])
            if (
                expected is None
                or len(values) != 1
                or not hmac.compare_digest(
                    values[0].encode(), ("Bearer " + expected).encode()
                )
            ):
                self.send_error(401)
                return
            length = self.headers.get("Content-Length", "")
            if not length.isdecimal() or not 0 < int(length) <= body_limit:
                self.send_error(413)
                return
            raw = self.rfile.read(int(length))
            try:
                broker_request = strict_json(raw)
            except ValueError:
                self.send_error(400)
                return
            with bridge_lock:
                bridge_attempts.append(parts[1])
                if parts[1] in bridge_faults:
                    self.send_response(429)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Retry-After", "120")
                    self.end_headers()
                    self.wfile.write(
                        b'{"error":{"message":"local injected rate limit"}}'
                    )
                    return
                frame_send(channel, {"provider": parts[1], "payload": broker_request})
                first = frame_read(channel)
                self.send_response(first["status"])
                self.send_header("Content-Type", first["content_type"])
                self.end_headers()
                while True:
                    part = frame_read(channel)
                    if part.get("end"):
                        break
                    self.wfile.write(part.get("body", "").encode())
                    self.wfile.flush()

    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(directory),
        "SQLITE_PATH": str(directory / "newapi.db"),
        "PORT": "8002",
        "SESSION_SECRET": secrets.token_urlsafe(32),
        "CRYPTO_SECRET": secrets.token_urlsafe(32),
        "GOMAXPROCS": "2",
        "GOMEMLIMIT": "160MiB",
    }
    report: dict = {"live_provider_test": True, "results": []}
    ledger = state = process = None
    try:
        with running(ThreadingHTTPServer(("127.0.0.1", 0), Bridge)) as bridge, (
            directory / "newapi.log"
        ).open("xb") as log:
            process = subprocess.Popen(
                [str(binary)],
                cwd=directory,
                env=env,
                stdout=log,
                stderr=log,
                umask=0o077,
            )
            base = "http://127.0.0.1:8002"
            for _ in range(200):
                try:
                    with request.urlopen(base + "/api/setup", timeout=0.2):
                        break
                except OSError:
                    if process.poll() is not None:
                        raise CatalogError("disposable New-API exited") from None
                    time.sleep(0.05)
            else:
                raise CatalogError("disposable New-API startup timed out")
            admin, password = bootstrap(base)
            enroll_local_proof(admin, password)
            admin.call("PUT", "/api/option/", {"key": "RetryTimes", "value": "0"})
            ledger, state = QuotaLedger(directory / "sidecar.db"), RouterState(
                directory / "sidecar.db"
            )
            routes = []
            for index, target in enumerate(targets, 1):
                route = Route(
                    target["provider"],
                    target["provider"],
                    target["model"],
                    "live-bridge-" + str(index),
                    "sc.live." + target["provider"],
                    index,
                    quota_type="rate_limited",
                    context_window=target["context"],
                    output_limit=output_cap,
                    input_price=0,
                    output_price=0,
                    stream=True,
                    tool_call=True,
                    limits={"rpm": target["rpm"], "rpd": call_limit},
                )
                endpoint = f"http://127.0.0.1:{bridge.server_address[1]}/{target['provider']}/v1"
                bridge_key = secrets.token_urlsafe(24)
                bridge_keys[target["provider"]] = bridge_key
                route = admin.create_route(route, endpoint, bridge_key)
                ledger.configure(route)
                routes.append(route)
            report["audit_digest"] = admin.audit(routes)
            token_id, relay = relay_token(admin, [route.alias for route in routes])
            admin.audit_relay(relay, token_id)
            runtime = Runtime(
                ledger,
                state,
                base + "/v1",
                relay,
                explore=False,
                timeout=40,
                deadline=45,
            )
            if failover:
                report["failover"] = verify_failover(
                    directory, base, relay, routes, bridge_faults, bridge_attempts
                )
                report["passed"] = len(report["failover"]) == 2
                return report
            if opencode is not None:
                if len(routes) != 1:
                    raise ValueError("live tool loop must use one verified target")
                report["opencode"] = live_opencode(directory, opencode, runtime)
                decisions = [
                    dict(row)
                    for row in ledger.db.execute("SELECT * FROM routing_decisions")
                ]
                report["usage"] = [
                    {
                        name: row[name]
                        for name in (
                            "status",
                            "input_tokens",
                            "output_tokens",
                            "usage_estimated",
                            "latency_ms",
                            "ttft_ms",
                        )
                    }
                    for row in decisions
                ]
                report["diagnostics"] = ledger.diagnostics()
                report["passed"] = (
                    report["opencode"]["passed"]
                    and 2 <= len(decisions) <= TOOL_CALLS
                    and all(
                        row["status"] == "success" and row["usage_estimated"] == 0
                        for row in decisions
                    )
                )
                return report
            for route in routes:
                for stream in (False, True):
                    payload = {
                        "model": "auto-coding" if stream else "auto-chat",
                        "stream": stream,
                        "max_tokens": OUTPUT_CAP,
                        "messages": [
                            {"role": "user", "content": "Reply with exactly ok."}
                        ],
                    }
                    if route.provider_id == "groq":
                        payload["reasoning_effort"] = "low"
                    prepared = prepare(payload)
                    try:
                        if stream:
                            source = runtime.open_stream(prepared, routes=[route])
                            completion = collect(source)
                            identity = source.identity
                        else:
                            result = runtime.complete(prepared, routes=[route])
                            completion, identity = result.completion, result.decision_id
                        decision = ledger.decision(identity)
                        text = completion.message.get("content") or ""
                        report["results"].append(
                            {
                                "provider": route.provider_id,
                                "model": route.model_id,
                                "profile": payload["model"],
                                "stream": stream,
                                "transport_ok": True,
                                "content_ok": text.strip().lower().strip(".!") == "ok",
                                "usage_estimated": bool(decision["usage_estimated"]),
                                "input_tokens": decision["input_tokens"],
                                "output_tokens": decision["output_tokens"],
                                "ttft_ms": decision["ttft_ms"],
                                "reasoning_ttft_ms": decision["reasoning_ttft_ms"],
                                "latency_ms": decision["latency_ms"],
                            }
                        )
                        if (
                            not report["results"][-1]["content_ok"]
                            or decision["usage_estimated"]
                        ):
                            break
                    except ProviderError as exc:
                        report["results"].append(
                            {
                                "provider": route.provider_id,
                                "stream": stream,
                                "transport_ok": False,
                                "error_class": exc.error_class,
                            }
                        )
                        break
            report["diagnostics"] = ledger.diagnostics()
            report["passed"] = len(report["results"]) == len(routes) * 2 and all(
                row.get("transport_ok")
                and row.get("content_ok")
                and not row.get("usage_estimated", True)
                for row in report["results"]
            )
    except (CatalogError, ValueError, OSError) as exc:
        report.update(passed=False, error=str(exc))
    finally:
        if state is not None:
            state.close()
        if ledger is not None:
            ledger.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-db", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--artifacts", type=Path, default=Path("/tmp/opencode"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--opencode",
        type=Path,
        help="one live read-only tool loop instead of chat/SSE smokes",
    )
    parser.add_argument(
        "--failover",
        action="store_true",
        help="two directions of injected-429 fallback to live providers",
    )
    parser.add_argument("--bridge-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--namespace-parent-inode", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.umask(0o077)
    if args.bridge_fd is not None:
        with socket.socket(fileno=args.bridge_fd) as channel:
            report = child_verify(
                args.directory,
                args.binary,
                channel,
                args.namespace_parent_inode,
                args.opencode,
                args.failover,
            )
        (args.directory / "live-report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        return 0
    if not args.provider_db or not args.targets:
        parser.error("--provider-db and --targets are required")
    targets, catalogs = load_targets(args.provider_db, args.targets)
    report = {"catalogs": catalogs, "generation_requests": 0, "executed": False}
    if args.opencode and len(targets) != 1:
        raise ValueError("select exactly one verified target for a live tool loop")
    if args.failover and (len(targets) != 2 or args.opencode):
        raise ValueError(
            "failover requires exactly two verified targets and no tool-loop mode"
        )
    if args.execute and targets and all(row["ok"] for row in catalogs):
        if not args.binary or not args.binary.is_file() or not args.artifacts.is_dir():
            raise ValueError("existing binary and artifact parent required")
        directory = Path(tempfile.mkdtemp(prefix="newapi-live-", dir=args.artifacts))
        parent, child = socket.socketpair()
        broker = Broker(
            targets, tool_loop=args.opencode is not None, failover=args.failover
        )
        thread = threading.Thread(target=broker.serve, args=(parent,), daemon=True)
        thread.start()
        frame_send(parent, {"targets": [target.public() for target in targets]})
        namespace_env = {
            "PATH": os.environ["PATH"],
            "HOME": str(directory),
            "LANG": "C.UTF-8",
        }
        arguments = [
            "unshare",
            "--user",
            "--map-root-user",
            "--net",
            sys.executable,
            str(Path(__file__).resolve()),
            "--binary",
            str(args.binary.resolve()),
            "--bridge-fd",
            str(child.fileno()),
            "--directory",
            str(directory),
            "--namespace-parent-inode",
            str(os.stat("/proc/self/ns/net").st_ino),
            *(["--failover"] if args.failover else []),
            *(["--opencode", str(args.opencode.resolve())] if args.opencode else []),
        ]
        process = None
        try:
            with (directory / "child.out").open("xb") as stdout, (
                directory / "child.err"
            ).open("xb") as stderr:
                process = subprocess.Popen(
                    arguments,
                    env=namespace_env,
                    pass_fds=(child.fileno(),),
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                try:
                    process.wait(timeout=150)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=3)
            report.update(
                executed=True,
                artifact_dir=str(directory),
                broker=broker.events,
                generation_requests=sum(broker.calls.values()),
            )
            for output in (directory / "child.out", directory / "child.err"):
                data = output.read_bytes()
                if any(target.key.encode() in data for target in targets):
                    raise ValueError("isolated child exposed a provider credential")
            path = directory / "live-report.json"
            if process.returncode != 0 or not path.is_file():
                report.update(
                    passed=False, error="isolated live check exited before its report"
                )
            else:
                report["integration"] = json.loads(path.read_text())
                report["passed"] = report["integration"].get("passed") is True
            if any(target.key in json.dumps(report) for target in targets):
                raise ValueError("live report contains a provider credential")
        finally:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            child.close()
            parent.shutdown(socket.SHUT_RDWR)
            parent.close()
            thread.join(timeout=3)
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return (
        0
        if all(row["ok"] for row in catalogs) and report.get("passed", not args.execute)
        else 1
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CatalogError, ValueError, OSError):
        print(
            json.dumps(
                {
                    "passed": False,
                    "error": "live preflight configuration or transport failure",
                }
            )
        )
        raise SystemExit(1)
