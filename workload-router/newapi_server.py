"""Separate opt-in HTTP boundary; the legacy gateway entry point is unchanged."""

from __future__ import annotations

import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import re
import shlex
import signal
import sqlite3
import subprocess
import threading
import time
from urllib.parse import parse_qs, unquote

from newapi_runtime import (LEGACY_ALIASES, MODEL_ALIASES, Prepared, Runtime,
                            chat_payload_from_responses, embedding_inputs, prepare,
                            responses_object, responses_usage, strict_json, validate_key)
from provider_adapter import ProviderError
from quota_ledger import QuotaLedger, token_count
from router_state import RouterState


LOG = logging.getLogger(__name__)

MODEL_DESCRIPTIONS = {
    "auto-coding": "Coding and agentic tasks",
    "auto-code": "Compatibility alias for auto-coding",
    "auto-chat": "General chat and tool-bearing requests",
}
COMMAND_PATTERNS = (
    r"^[/!](?:model|switch)\s+(\S+)",
    r"^[/!](?:use)\s+(\S+)",
)
COMMAND_SHORTCUTS = {
    "/uncensored": "uncensored-chat", "/coding": "coding-elite",
    "/fast": "chat-fast", "/smart": "chat-smart", "/rp": "chat-rp",
    "!uncensored": "uncensored-chat", "!coding": "coding-elite",
    "!fast": "chat-fast", "!smart": "chat-smart", "!rp": "chat-rp",
}


def model_listing() -> dict:
    descriptions = dict(MODEL_DESCRIPTIONS)
    for name, (profile, step) in LEGACY_ALIASES.items():
        descriptions[name] = f"Legacy gateway alias; routes as {profile} / {step}"
    return {"models": list(MODEL_ALIASES), "descriptions": descriptions}


def command_response(requested: str, remaining: str, original: str) -> dict:
    if requested in MODEL_ALIASES:
        return {"command_found": True, "requested_model": requested,
                "canonical_model": requested, "remaining_message": remaining,
                "valid": True}
    return {"command_found": True, "requested_model": requested, "valid": False,
            "error": "Invalid model. Available: " + ", ".join(MODEL_ALIASES),
            "remaining_message": original}


def parse_model_command(message: object) -> dict:
    """Parse the legacy `/model` and shortcut forms without mutating a session."""
    if not isinstance(message, str) or len(message) > 8192:
        raise ValueError("message must be a string of at most 8192 characters")
    stripped = message.strip()
    for pattern in COMMAND_PATTERNS:
        match = re.match(pattern, stripped, re.IGNORECASE)
        if match:
            return command_response(match.group(1).lower(),
                                    stripped[match.end():].strip(), stripped)
    lower = stripped.lower()
    for shortcut, model in COMMAND_SHORTCUTS.items():
        if lower.startswith(shortcut):
            return command_response(model, stripped[len(shortcut):].strip(), stripped)
    return {"command_found": False, "remaining_message": stripped}


def query_int(query: dict[str, list[str]], name: str, default: int,
              minimum: int, maximum: int) -> int:
    values = query.get(name)
    if values is None:
        return default
    if len(values) != 1 or not values[0].isdigit():
        raise ValueError(f"{name} must be one integer")
    value = int(values[0])
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


class Server(ThreadingHTTPServer):
    """Bound worker creation, not merely the number of handlers doing dispatch."""

    daemon_threads = False
    request_queue_size = 16

    def __init__(self, address: tuple[str, int], runtime: Runtime, client_key: str, *,
                 workers: int = 8, default_output: int = 2048, compress: bool = False,
                 embedding_aliases: frozenset[str] = frozenset()) -> None:
        if address[0] not in {"127.0.0.1", "localhost"}:
            raise ValueError("sidecar must bind loopback; terminate TLS at a separate proxy")
        self.runtime, self.client_key = runtime, validate_key(client_key)
        if not 1 <= token_count(workers, "workers") <= 32:
            raise ValueError("workers must be between 1 and 32")
        self.default_output = token_count(default_output, "default_output")
        if not 1 <= self.default_output <= 131072 or not isinstance(compress, bool):
            raise ValueError("invalid default output or compression flag")
        if not isinstance(embedding_aliases, frozenset) or any(
                not isinstance(name, str) or not name or len(name) > 512
                for name in embedding_aliases):
            raise ValueError("invalid embedding aliases")
        self.compress, self.embedding_aliases = compress, embedding_aliases
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
    server_version = "newapi-sidecar"
    sys_version = ""
    max_body_bytes = 1024 * 1024

    def setup(self) -> None:
        super().setup()
        self.response_committed = False
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
        self.response_committed = True
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
        path = self.path.split("?", 1)[0]
        if path == "/healthz":
            self.send_json(200, {"status": "ok", "service": "newapi-sidecar"})
            return
        if not self.authorized():
            return
        if path == "/v1/models":
            self.send_json(200, {"object": "list", "data": [
                {"id": name, "object": "model", "created": 0, "owned_by": "sidecar",
                 **({"deprecated": True} if name in LEGACY_ALIASES else {})}
                for name in MODEL_ALIASES]})
        elif path.startswith("/v1/models/"):
            model_id = unquote(path[len("/v1/models/"):])
            if not model_id or len(model_id) > 512:
                self.send_json(404, {"error": "not found"})
                return
            known = model_id in MODEL_ALIASES
            owner = "sidecar" if known else (
                model_id.split("/", 1)[0] if "/" in model_id else "unknown")
            self.send_json(200, {
                "id": model_id, "object": "model", "created": int(time.time()),
                "owned_by": owner,
                **({"deprecated": True} if model_id in LEGACY_ALIASES else {}),
            })
        elif path == "/v1/providers":
            self.send_json(200, sorted({route.provider_id
                                        for route in self.server.runtime.ledger.routes()}))
        elif path == "/v1/tps-stats":
            try:
                query = parse_qs(self.path.partition("?")[2], keep_blank_values=True)
                days = query_int(query, "days", 7, 1, 365)
                min_samples = query_int(query, "min_samples", 5, 1, 10000)
                export = query.get("export")
                if export not in (None, ["false"], ["0"]):
                    raise ValueError("export is not supported")
                stats = self.server.runtime.ledger.tps_stats(days, min_samples)
                self.send_json(200, {"object": "list", "data": stats, "count": len(stats)})
            except ValueError as exc:
                self.send_json(400, {"error": str(exc)})
        elif path == "/v1/model/list":
            self.send_json(200, model_listing())
        elif path == "/v1/model/status":
            self.send_json(200, {"current_model": "auto-chat",
                                 "available_models": list(MODEL_ALIASES),
                                 "auto_routing_enabled": True})
        elif path == "/v1/router/diagnostics":
            self.send_json(200, {**self.server.runtime.ledger.diagnostics(),
                                 "free_only": self.server.runtime.free_only})
        else:
            self.send_json(404, {"error": "not found"})

    def method_not_allowed(self) -> None:
        self.send_json(405, {"error": "method not allowed"}, {"Allow": "GET, POST"})

    do_PUT = do_PATCH = do_DELETE = method_not_allowed

    def read_payload(self) -> dict | None:
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
        return strict_json(raw)

    def do_POST(self) -> None:  # noqa: N802
        if not self.authorized():
            return
        if self.path not in {"/v1/chat/completions", "/v1/router/route",
                             "/v1/responses", "/v1/embeddings",
                             "/v1/model/parse-command", "/v1/cost-estimate"}:
            self.send_json(404, {"error": "not found"})
            return
        stream = None
        try:
            payload = self.read_payload()
            if payload is None:
                return
            if self.path == "/v1/model/parse-command":
                self.send_json(200, parse_model_command(payload.get("message", "")))
                return
            runtime = self.server.runtime
            if self.path == "/v1/cost-estimate":
                model = payload.get("model")
                if not isinstance(model, str) or not model:
                    raise ValueError("'model' is required")
                result = runtime.ledger.estimate_cost(
                    model, payload.get("prompt_tokens", 0),
                    payload.get("completion_tokens", 0),
                    payload.get("cache_read_tokens", 0),
                    payload.get("cache_creation_tokens", 0))
                if result is None:
                    result = {"model": model, "cost": None, "currency": "USD",
                              "pricing": {}, "source": "unknown",
                              "error": "Pricing data not available for this model"}
                self.send_json(200, result)
                return
            if self.path == "/v1/embeddings":
                alias = payload.get("model")
                if not isinstance(alias, str) or alias not in self.server.embedding_aliases:
                    raise ValueError("model is not a configured embedding alias")
                route, decision, body = runtime.embed(alias, embedding_inputs(payload))
                body["model"] = alias
                self.send_json(200, body, {"X-Router-Deployment": route.route_id,
                                           "X-Router-Decision": decision})
                return
            if self.path == "/v1/responses":
                prepared = prepare(chat_payload_from_responses(payload),
                                   self.headers.get("X-Session-Id", ""),
                                   self.headers.get("X-Route-Step", ""),
                                   default_output=self.server.default_output,
                                   compress=self.server.compress)
                if prepared.work.streaming and prepared.work.tools:
                    raise ValueError("streaming tools are not supported by the responses shim")
                if prepared.work.streaming:
                    stream = runtime.open_stream(prepared)
                    self.send_responses_stream(prepared, stream)
                else:
                    result = runtime.complete(prepared)
                    self.send_json(200, responses_object(result.decision_id, prepared.model,
                                                         result.completion, int(time.time())),
                                   {"X-Router-Deployment": result.route.route_id,
                                    "X-Router-Decision": result.decision_id})
                return
            prepared = prepare(payload, self.headers.get("X-Session-Id", ""),
                               self.headers.get("X-Route-Step", ""),
                               default_output=self.server.default_output,
                               compress=self.server.compress)
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
                self.response_committed = True
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
            if not self.response_committed:
                self.send_json(400, {"error": "invalid JSON, model, step or request limits"})
        except ProviderError as exc:
            if not self.response_committed:
                headers = {"X-Router-Error-Class": exc.error_class}
                if exc.retry_after_seconds:
                    headers["Retry-After"] = str(max(1, int(exc.retry_after_seconds)))
                self.send_json(503, {"error": "no feasible upstream", "error_class": exc.error_class}, headers)
        except sqlite3.Error:
            if not self.response_committed:
                self.send_json(503, {"error": "sidecar state unavailable"})
        except OSError:
            self.close_connection = True
        finally:
            self.close_connection = True
            if stream is not None:
                stream.close()


    def send_responses_stream(self, prepared: Prepared, stream) -> None:
        """Translate the chat SSE stream into text-only Responses API events."""
        identity, item_id = stream.identity, "msg_" + stream.identity[:24]
        created, sequence = int(time.time()), 0
        base = {"id": "resp_" + identity, "object": "response", "created_at": created,
                "model": prepared.model}

        def emit(event: str, data: dict) -> None:
            nonlocal sequence
            payload = {"type": event, "sequence_number": sequence, **data}
            sequence += 1
            self.wfile.write(("event: " + event + "\n").encode("ascii"))
            self.wfile.write(("data: " + json.dumps(payload, allow_nan=False, separators=(",", ":"))
                              + "\n\n").encode("utf-8"))
            self.wfile.flush()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.send_header("X-Router-Deployment", stream.route.route_id)
        self.send_header("X-Router-Decision", identity)
        self.end_headers()
        self.response_committed = True
        self.close_connection = True
        emit("response.created", {"response": {**base, "status": "in_progress", "output": []}})
        emit("response.output_item.added", {"output_index": 0, "item": {
            "id": item_id, "type": "message", "status": "in_progress",
            "role": "assistant", "content": []}})
        emit("response.content_part.added", {"item_id": item_id, "output_index": 0,
                                              "content_index": 0,
                                              "part": {"type": "output_text", "text": "",
                                                       "annotations": []}})
        parts: list[str] = []
        try:
            for value in stream:
                if value == "[DONE]":
                    continue
                event = json.loads(value)
                for choice in event.get("choices", []):
                    delta = choice.get("delta") if isinstance(choice, dict) else None
                    text = delta.get("content") if isinstance(delta, dict) else None
                    if isinstance(text, str) and text:
                        parts.append(text)
                        emit("response.output_text.delta", {"item_id": item_id, "output_index": 0,
                                                            "content_index": 0, "delta": text})
        except (ProviderError, ValueError, TypeError):
            emit("response.failed", {"response": {**base, "status": "failed", "output": [],
                                                   "error": {"code": "upstream_error",
                                                             "message": "no usable output"}}})
            return
        text = "".join(parts)
        item = {"id": item_id, "type": "message", "status": "completed", "role": "assistant",
                "content": [{"type": "output_text", "text": text,
                              "annotations": [], "logprobs": []}]}
        emit("response.output_text.done", {"item_id": item_id, "output_index": 0,
                                           "content_index": 0, "text": text})
        emit("response.content_part.done", {"item_id": item_id, "output_index": 0,
                                            "content_index": 0, "part": item["content"][0]})
        emit("response.output_item.done", {"output_index": 0, "item": item})
        emit("response.completed", {"response": {**base, "status": "completed",
                                                   "output": [item], "output_text": text,
                                                   "usage": responses_usage(stream.usage) or {}}})


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


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit("missing environment variable: " + name)
    return value


def parse_embedding_aliases(value: str) -> frozenset[str]:
    """Embedding aliases are explicit; a typo must fail startup, not embed silently."""
    names = {part.strip() for part in value.split(",") if part.strip()}
    if any(len(name) > 512 or any(ord(char) < 33 for char in name) for name in names):
        raise ValueError("invalid embedding alias")
    return frozenset(names)


def _install_stop_handlers() -> None:
    """systemd sends SIGTERM; exit through the server's own close path."""
    if threading.current_thread() is not threading.main_thread():
        return

    def stop(_signum, _frame) -> None:
        raise SystemExit(0)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)


def serve() -> None:
    """Require a fresh management audit before serving private-alias requests."""
    from newapi_catalog import AdminClient

    path = Path(_env("SIDECAR_DB")).expanduser().resolve()
    ledger = QuotaLedger(path)
    state = None
    try:
        base_url = _env("NEWAPI_BASE_URL")
        proof_command = os.environ.get("NEWAPI_CHANNEL_KEY_PROOF_COMMAND")
        key_proof = (channel_key_proof_from_command(proof_command)
                     if proof_command else None)
        admin = AdminClient(base_url, _env("NEWAPI_ADMIN_TOKEN"),
                            int(_env("NEWAPI_ADMIN_USER_ID")), key_proof=key_proof)
        admin.audit(ledger.routes())
        admin.audit_relay(_env("NEWAPI_API_KEY"), int(_env("NEWAPI_RELAY_TOKEN_ID")))
        aliases = parse_embedding_aliases(os.environ.get("SIDECAR_EMBEDDING_ALIASES", ""))
        unknown = aliases - {route.alias for route in ledger.routes()}
        if unknown:
            raise SystemExit("embedding aliases are not configured routes: "
                             + ", ".join(sorted(unknown)))
        collided = aliases & set(MODEL_ALIASES)
        if collided:
            raise SystemExit("embedding aliases collide with chat aliases: "
                             + ", ".join(sorted(collided)))
        state = RouterState(path)
        runtime = Runtime(ledger, state, base_url, _env("NEWAPI_API_KEY"),
                          free_only=env_flag("SIDECAR_FREE_ONLY", True))
        _install_stop_handlers()
        with Server((os.environ.get("SIDECAR_HOST", "127.0.0.1"), int(_env("SIDECAR_PORT"))),
                    runtime, _env("SIDECAR_API_KEY"), embedding_aliases=aliases,
                    compress=env_flag("SIDECAR_COMPRESS_JSON", False)) as server:
            try:
                server.serve_forever()
            except (KeyboardInterrupt, SystemExit):
                LOG.info("sidecar stopping")
    finally:
        if state is not None:
            state.close()
        ledger.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    serve()
