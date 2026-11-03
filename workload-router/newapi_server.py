"""Separate opt-in HTTP boundary; the legacy gateway entry point is unchanged."""

from __future__ import annotations

import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import threading
import time

from newapi_runtime import PROFILE_ALIASES, Prepared, Runtime, prepare, strict_json, validate_key
from provider_adapter import ProviderError
from quota_ledger import QuotaLedger, token_count
from router_state import RouterState


LOG = logging.getLogger(__name__)


class Server(ThreadingHTTPServer):
    """Bound worker creation, not merely the number of handlers doing dispatch."""

    daemon_threads = False
    request_queue_size = 16

    def __init__(self, address: tuple[str, int], runtime: Runtime, client_key: str, *,
                 workers: int = 8, default_output: int = 2048, compress: bool = False) -> None:
        if address[0] not in {"127.0.0.1", "localhost"}:
            raise ValueError("sidecar must bind loopback; terminate TLS at a separate proxy")
        self.runtime, self.client_key = runtime, validate_key(client_key)
        if not 1 <= token_count(workers, "workers") <= 32:
            raise ValueError("workers must be between 1 and 32")
        self.default_output = token_count(default_output, "default_output")
        if not 1 <= self.default_output <= 131072 or not isinstance(compress, bool):
            raise ValueError("invalid default output or compression flag")
        self.compress = compress
        self.slots = threading.BoundedSemaphore(token_count(workers, "workers"))
        super().__init__(address, Handler)

    def process_request(self, sock, address) -> None:
        if not self.slots.acquire(blocking=False):
            try:
                sock.settimeout(0.2)
                sock.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\nRetry-After: 1\r\n\r\n")
            except OSError:
                pass
            finally:
                self.shutdown_request(sock)
            return
        try:
            super().process_request(sock, address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, sock, address) -> None:
        try:
            super().process_request_thread(sock, address)
        finally:
            self.slots.release()

    def handle_error(self, _request, _address) -> None:
        # Never print request bodies, exception strings, or credentials.
        LOG.error("sidecar handler failed")


class Handler(BaseHTTPRequestHandler):
    """Responses mirror the legacy handler without importing its global DB/config."""

    server: Server
    protocol_version = "HTTP/1.1"
    max_body_bytes = 1024 * 1024

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, _format: str, *args: object) -> None:
        return

    def send_json(self, status: int, payload: dict, headers: dict[str, str] | None = None) -> None:
        encoded = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.close_connection = True
        self.wfile.write(encoded)

    def authorized(self) -> bool:
        values = self.headers.get_all("Authorization", [])
        if len(values) != 1 or not hmac.compare_digest(values[0].encode("utf-8"),
                                                      ("Bearer " + self.server.client_key).encode("ascii")):
            self.send_json(401, {"error": "invalid sidecar token"})
            return False
        return True

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self.send_json(200, {"status": "ok", "service": "newapi-sidecar"})
            return
        if not self.authorized():
            return
        if self.path == "/v1/models":
            self.send_json(200, {"object": "list", "data": [
                {"id": name, "object": "model", "created": 0, "owned_by": "sidecar"}
                for name in PROFILE_ALIASES]})
        elif self.path == "/v1/router/diagnostics":
            self.send_json(200, {**self.server.runtime.ledger.diagnostics(),
                                 "free_only": self.server.runtime.free_only})
        else:
            self.send_json(404, {"error": "not found"})

    def read_request(self) -> Prepared | None:
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) != 1 or not lengths[0].isdigit():
            raise ValueError("one Content-Length required")
        size = int(lengths[0])
        if size > self.max_body_bytes:
            self.send_json(413, {"error": "request body exceeds 1 MiB"})
            return None
        if size <= 0:
            raise ValueError("empty request body")
        raw = self.rfile.read(size)
        if len(raw) != size:
            raise ValueError("truncated request body")
        payload = strict_json(raw)
        return prepare(payload, self.headers.get("X-Session-Id", ""),
                       self.headers.get("X-Route-Step", ""),
                       default_output=self.server.default_output, compress=self.server.compress)

    def do_POST(self) -> None:  # noqa: N802
        if not self.authorized():
            return
        if self.path not in {"/v1/chat/completions", "/v1/router/route"}:
            self.send_json(404, {"error": "not found"})
            return
        committed = False
        stream = None
        try:
            prepared = self.read_request()
            if prepared is None:
                return
            runtime = self.server.runtime
            if self.path == "/v1/router/route":
                candidates = runtime.ranked(prepared, explore=False)
                self.send_json(200, {"profile": prepared.work.profile, "step": prepared.work.step,
                                     "baseline_tokens": prepared.baseline_tokens,
                                     "compressed_tokens": prepared.compressed_tokens,
                                     "candidates": [{"route_id": row.route.route_id, "score": row.score,
                                                     "failure_probability": row.failure_probability,
                                                     "quota_pressure": row.pressure} for row in candidates]})
            elif prepared.work.streaming:
                stream = runtime.open_stream(prepared)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Connection", "close")
                self.send_header("X-Router-Deployment", stream.route.route_id)
                self.send_header("X-Router-Decision", stream.identity)
                self.end_headers()
                committed = True
                self.close_connection = True
                for value in stream:
                    if value != "[DONE]":
                        event = json.loads(value)
                        event["model"] = prepared.model
                        value = json.dumps(event, allow_nan=False, separators=(",", ":"))
                    self.wfile.write(("data: " + value + "\n\n").encode("utf-8"))
                    self.wfile.flush()
            else:
                result = runtime.complete(prepared)
                body = {"id": result.decision_id, "object": "chat.completion", "created": int(time.time()),
                        "model": prepared.model, "choices": [{"index": 0, "message": result.completion.message,
                                                               "finish_reason": result.completion.finish_reason}],
                        "router": {"deployment": result.route.route_id, "attempts": len(result.attempts)}}
                if result.completion.usage is not None:
                    body["usage"] = result.completion.usage
                self.send_json(200, body, {"X-Router-Deployment": result.route.route_id,
                                           "X-Router-Decision": result.decision_id})
        except (ValueError, TypeError, UnicodeError, RecursionError):
            if not committed:
                self.send_json(400, {"error": "invalid JSON, model, step or request limits"})
        except ProviderError as exc:
            if not committed:
                headers = {"X-Router-Error-Class": exc.error_class}
                if exc.retry_after_seconds:
                    headers["Retry-After"] = str(max(1, int(exc.retry_after_seconds)))
                self.send_json(503, {"error": "no feasible upstream", "error_class": exc.error_class}, headers)
        except sqlite3.Error:
            if not committed:
                self.send_json(503, {"error": "sidecar state unavailable"})
        except OSError:
            self.close_connection = True
        finally:
            self.close_connection = True
            if stream is not None:
                stream.close()


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name, "1" if default else "0").lower()
    if value not in {"0", "1", "false", "true"}:
        raise ValueError("invalid boolean environment setting")
    return value in {"1", "true"}


def channel_key_proof_from_command(command: str):
    """Build a proof callback backed by an operator-controlled step-up helper.

    The helper receives only the channel ID on stdin and must print one proof
    token. It owns the interactive/authenticated 2FA flow; the sidecar never
    stores a password, backup code, or generated enrollment secret.
    """
    argv = shlex.split(command)
    if not argv or any(not isinstance(part, str) or not part for part in argv):
        raise ValueError("invalid channel proof command")

    def proof(channel_id: int) -> str:
        try:
            result = subprocess.run(
                argv,
                input=str(token_count(channel_id, "channel_id")) + "\n",
                text=True,
                capture_output=True,
                check=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            raise RuntimeError("channel proof helper failed") from None
        value = result.stdout.strip()
        if not value or "\n" in value or "\r" in value:
            raise RuntimeError("channel proof helper returned invalid proof")
        return validate_key(value)

    return proof


def serve() -> None:
    """Require a fresh management audit before serving private-alias requests."""
    from newapi_catalog import AdminClient

    path = Path(os.environ["SIDECAR_DB"]).expanduser().resolve()
    ledger = QuotaLedger(path)
    state = None
    try:
        base_url = os.environ["NEWAPI_BASE_URL"]
        proof_command = os.environ.get("NEWAPI_CHANNEL_KEY_PROOF_COMMAND")
        key_proof = (channel_key_proof_from_command(proof_command)
                     if proof_command else None)
        admin = AdminClient(base_url, os.environ["NEWAPI_ADMIN_TOKEN"],
                            int(os.environ["NEWAPI_ADMIN_USER_ID"]), key_proof=key_proof)
        admin.audit(ledger.routes())
        admin.audit_relay(os.environ["NEWAPI_API_KEY"], int(os.environ["NEWAPI_RELAY_TOKEN_ID"]))
        state = RouterState(path)
        runtime = Runtime(ledger, state, base_url, os.environ["NEWAPI_API_KEY"],
                          free_only=env_flag("SIDECAR_FREE_ONLY", True))
        with Server((os.environ.get("SIDECAR_HOST", "127.0.0.1"), int(os.environ["SIDECAR_PORT"])),
                    runtime, os.environ["SIDECAR_API_KEY"],
                    compress=env_flag("SIDECAR_COMPRESS_JSON", False)) as server:
            server.serve_forever()
    finally:
        if state is not None:
            state.close()
        ledger.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    serve()
