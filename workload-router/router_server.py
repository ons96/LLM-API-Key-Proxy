"""Minimal local HTTP boundary for exercising the deterministic router."""

import json
import os
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from router_core import ROUTER_VERSION, HealthRegistry, RequestFeatures, estimate_context_tokens, normalize_phase, route_plan
from router_config import default_deployments, load_deployments
from provider_adapter import ProviderAdapter, ProviderError, ProviderRequest, adapters_from_environment, close_stream, execute, stream_execute, validate_stream_payload
from provider_metadata import load_provider_group_deployments
from router_state import RouterState
from stall_detector import StallState


def _load_runtime_deployments():
    """Prefer explicit config, with optional provider-manager metadata loading."""
    config_path = os.environ.get("ROUTER_DEPLOYMENTS")
    if config_path:
        return load_deployments(config_path)
    metadata_path = os.environ.get("ROUTER_PROVIDER_DB")
    if metadata_path:
        groups = [item.strip() for item in os.environ.get("ROUTER_PROVIDER_GROUPS", "").split(",") if item.strip()]
        free_only = os.environ.get("ROUTER_FREE_ONLY", "1").strip().lower() not in {"0", "false", "no"}
        try:
            return load_provider_group_deployments(metadata_path, groups or None, free_only=free_only)
        except (OSError, sqlite3.Error, ValueError):
            pass
    return default_deployments()


DEFAULT_DEPLOYMENTS = _load_runtime_deployments()


def _nonnegative_int(value: object, default: int = 0) -> int:
    """Parse bounded token counters without allowing negative routing hints."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _nonnegative_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, parsed)


def _continuation_header(handler: BaseHTTPRequestHandler) -> bool:
    return handler.headers.get("X-Router-Continuation", "").strip().lower() in {"1", "true", "yes"}


def _cache_usage(usage: object) -> tuple[int, int]:
    """Extract common provider cache counters without depending on one schema."""
    if not isinstance(usage, dict):
        return 0, 0
    details = usage.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    cached = usage.get("cached_tokens", details.get("cached_tokens", details.get("cache_read_input_tokens", 0)))
    written = usage.get("cache_write_tokens", details.get("cache_write_tokens", 0))
    return _nonnegative_int(cached), _nonnegative_int(written)


def _has_cache_fields(value: object) -> bool:
    """Return whether a payload contains an explicit cache observation."""
    if not isinstance(value, dict):
        return False
    if any(key in value for key in ("cached_tokens", "cache_write_tokens")):
        return True
    details = value.get("prompt_tokens_details")
    return isinstance(details, dict) and any(
        key in details for key in ("cached_tokens", "cache_read_input_tokens", "cache_write_tokens")
    )


class RouterHandler(BaseHTTPRequestHandler):
    """Expose route inspection without making an upstream provider call."""

    deployments = DEFAULT_DEPLOYMENTS
    state = RouterState(os.environ.get("ROUTER_STATE_DB", ":memory:"))
    health = HealthRegistry()
    max_body_bytes = 1_048_576
    adapters: dict[str, ProviderAdapter] = adapters_from_environment(DEFAULT_DEPLOYMENTS)
    model_aliases = {"auto", "auto-code", "auto-chat"}
    stall_states: dict[str, StallState] = {}

    @classmethod
    def group_names(cls) -> set[str]:
        return {deployment.chain_group for deployment in cls.deployments}

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path == "/healthz":
            self._send_json(200, {"status": "ok", "service": "ai-workload-router", "version": ROUTER_VERSION})
            return
        if self.path == "/v1/router/diagnostics":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "deployments": len(self.deployments),
                    "decisions": self.state.count(),
                    "successful_routes": self.state.outcome_count("success"),
                    "operational_failures": self.state.outcome_count("operational_failure"),
                    "semantic_failures": self.state.outcome_count("semantic_failure"),
                    "cache_hits": self.state.total_cache_hits(),
                },
            )
            return
        if self.path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": alias, "object": "model", "created": 0, "owned_by": "ai-workload-router"}
                        for alias in sorted(self.model_aliases)
                    ],
                },
            )
            return
        if self.path == "/v1/router/groups":
            groups = []
            for group in sorted(self.group_names()):
                chain = sorted(
                    (deployment for deployment in self.deployments if deployment.chain_group == group),
                    key=lambda item: (item.chain_priority, item.deployment_id),
                )
                groups.append(
                    {
                        "id": group,
                        "capability": chain[0].capability.value if chain else "",
                        "chain": [
                            {
                                "deployment": item.deployment_id,
                                "provider": item.provider,
                                "model": item.model,
                                "priority": item.chain_priority,
                            }
                            for item in chain
                        ],
                    }
                )
            self._send_json(200, {"object": "list", "data": groups})
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path == "/v1/chat/completions":
            self._handle_chat_completion()
            return
        if self.path == "/v1/router/outcome":
            self._handle_outcome()
            return
        if self.path != "/v1/router/route":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > self.max_body_bytes:
                self._send_json(413, {"error": "request body exceeds 1 MiB limit"})
                return
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a non-empty list")
            if any(not isinstance(item, dict) for item in messages):
                raise ValueError("messages entries must be objects")
            prompt = "\n".join(str(item.get("content", "")) for item in messages)
            phase = normalize_phase(self.headers.get("X-Router-Phase"))
            requested_group = payload.get("model_group", self.headers.get("X-Router-Group", ""))
            if requested_group is not None and not isinstance(requested_group, str):
                raise ValueError("model_group must be a string")
            requested_group = requested_group.strip() if requested_group else ""
            if requested_group and requested_group not in self.group_names():
                raise ValueError("unknown model_group")
            continuation = bool(payload.get("continuation", False)) or _continuation_header(self)
            features = RequestFeatures(
                prompt=prompt,
                context_tokens=_nonnegative_int(payload.get("context_tokens", estimate_context_tokens(messages)), 1),
                tools=bool(payload.get("tools")),
                structured_output=bool(payload.get("response_format")),
                phase=phase,
                model_group=requested_group,
                continuation=continuation,
                cached_tokens=_nonnegative_int(payload.get("cached_tokens")),
            )
            session_id = self.headers.get("X-Session-Id", "anonymous")
            phase = features.phase or "default"
            stall = self.stall_states.setdefault(session_id, StallState(phase=phase))
            action = str(self.headers.get("X-Router-Action", "route"))
            verification_failed = self.headers.get("X-Router-Verification", "") == "failed"
            stall.observe(action, verification_failed)
            escalation = stall.should_escalate()
            if escalation:
                stall.escalate()
            preferred = self.state.preferred(session_id, phase)
            prefix = prompt[:4096]
            cache_hints = self.state.cache_hints(session_id, prefix)
            plan = route_plan(features, self.deployments, preferred, self.health, cache_hints)
            deployment = plan.candidates[0]
            self.state.record(session_id, prompt, deployment.capability.value, deployment.deployment_id, deployment.provider, deployment.config_fingerprint)
            self.state.set_preferred(session_id, phase, deployment.deployment_id)
            if _has_cache_fields(payload):
                self.state.record_cache(
                    session_id,
                    deployment.deployment_id,
                    prefix,
                    _nonnegative_int(payload.get("cached_tokens")),
                    _nonnegative_int(payload.get("cache_write_tokens")),
                    _nonnegative_float(
                        payload.get("cache_ttl_seconds", deployment.cache_ttl_seconds),
                        deployment.cache_ttl_seconds,
                    ),
                )
            response = {
                "deployment": deployment.deployment_id,
                "model_group": deployment.chain_group,
                "capability": deployment.capability.value,
                "escalation": stall.escalation_level,
                "route_reason": plan.reason,
                "cache_saved_ms": round(plan.cache_saved_ms, 3),
                "chain": [item.deployment_id for item in plan.candidates],
            }
            self._send_json(
                200,
                response,
                {
                    "X-Router-Deployment": deployment.deployment_id,
                    "X-Router-Capability": deployment.capability.value,
                    "X-Router-Group": deployment.chain_group,
                    "X-Router-Reason": plan.reason,
                },
            )
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": f"invalid request: {error}"})
        except LookupError as error:
            self._send_json(503, {"error": str(error)})

    def _handle_chat_completion(self) -> None:
        """Execute through explicitly installed adapters; no fake provider is bundled."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > self.max_body_bytes:
                self._send_json(413, {"error": "request body exceeds 1 MiB limit"})
                return
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            messages = payload.get("messages")
            model = payload.get("model", "auto")
            if not isinstance(model, str) or model not in self.model_aliases:
                raise ValueError("model must be one of: auto, auto-code, auto-chat")
            if not isinstance(messages, list) or not messages or any(not isinstance(item, dict) for item in messages):
                raise ValueError("messages must be a non-empty list of objects")
            provider_request = ProviderRequest.from_payload(payload)
            prompt = "\n".join(str(item.get("content", "")) for item in messages)
            phase = normalize_phase(self.headers.get("X-Router-Phase"))
            if not phase and model == "auto-code":
                phase = "implement"
            requested_group = payload.get("model_group", self.headers.get("X-Router-Group", ""))
            if requested_group is not None and not isinstance(requested_group, str):
                raise ValueError("model_group must be a string")
            requested_group = requested_group.strip() if requested_group else ""
            if requested_group and requested_group not in self.group_names():
                raise ValueError("unknown model_group")
            continuation = bool(payload.get("continuation", False)) or _continuation_header(self)
            features = RequestFeatures(
                prompt=prompt,
                context_tokens=_nonnegative_int(payload.get("context_tokens", estimate_context_tokens(messages)), 1),
                tools=bool(payload.get("tools")),
                structured_output=bool(payload.get("response_format")),
                phase=phase,
                model_group=requested_group,
                continuation=continuation,
                cached_tokens=_nonnegative_int(payload.get("cached_tokens")),
            )
            session_id = self.headers.get("X-Session-Id", "anonymous")
            phase = features.phase or "default"
            preferred = self.state.preferred(session_id, phase)
            prefix = prompt[:4096]
            cache_hints = self.state.cache_hints(session_id, prefix)
            plan = route_plan(features, self.deployments, preferred, self.health, cache_hints)

            def record_provider_failure(deployment_id: str, error_class: str) -> None:
                self.state.record_outcome(session_id, deployment_id, "operational_failure", error_class)

            response_format = payload.get("response_format")
            if response_format is not None and not isinstance(response_format, dict):
                raise ValueError("response_format must be an object")
            if payload.get("stream") is True:
                try:
                    streamed = self._send_provider_stream(
                        features, prompt, model, provider_request, record_provider_failure, cache_hints, preferred
                    )
                    if streamed is not None:
                        deployment_id, usage = streamed
                        deployment = self._deployment(deployment_id)
                        self.state.record(session_id, prompt, deployment.capability.value, deployment.deployment_id, deployment.provider, deployment.config_fingerprint)
                        self.state.set_preferred(session_id, phase, deployment_id)
                        if _has_cache_fields(usage):
                            cached_tokens, cache_writes = _cache_usage(usage)
                            self.state.record_cache(session_id, deployment_id, prefix, cached_tokens, cache_writes, deployment.cache_ttl_seconds)
                        self.state.record_outcome(session_id, deployment_id, "success")
                except ProviderError as error:
                    # Completion-only adapters retain the compatibility stream
                    # contract; native adapters stream without buffering.
                    if error.error_class != "no_adapter":
                        raise
                    result = execute(
                        features,
                        self.deployments,
                        self.adapters,
                        self.health,
                        prompt,
                        provider_request,
                        record_provider_failure,
                        cache_hints,
                        preferred,
                    )
                    self._send_stream(
                        result.content,
                        model,
                        result.deployment_id,
                        result.message,
                        result.finish_reason,
                        result.usage,
                    )
                    self.state.record(session_id, prompt, self._deployment(result.deployment_id).capability.value, result.deployment_id, self._deployment(result.deployment_id).provider, self._deployment(result.deployment_id).config_fingerprint)
                    self.state.set_preferred(session_id, phase, result.deployment_id)
                    cached_tokens, cache_writes = _cache_usage(result.usage)
                    deployment = self._deployment(result.deployment_id)
                    if _has_cache_fields(result.usage):
                        self.state.record_cache(session_id, result.deployment_id, prefix, cached_tokens, cache_writes, deployment.cache_ttl_seconds)
                    self.state.record_outcome(session_id, result.deployment_id, "success")
                return
            result = execute(
                features,
                self.deployments,
                self.adapters,
                self.health,
                prompt,
                provider_request,
                record_provider_failure,
                cache_hints,
                preferred,
            )
            message = result.message or {"role": "assistant", "content": result.content}
            response = {
                "id": "router-local",
                "object": "chat.completion",
                "model": model,
                "choices": [{"index": 0, "message": message, "finish_reason": result.finish_reason or "stop"}],
                "router": {"deployment": result.deployment_id, "attempts": result.attempts},
            }
            if result.usage is not None:
                response["usage"] = result.usage
            deployment = self._deployment(result.deployment_id)
            self.state.record(session_id, prompt, deployment.capability.value, deployment.deployment_id, deployment.provider, deployment.config_fingerprint)
            self.state.set_preferred(session_id, phase, deployment.deployment_id)
            cached_tokens, cache_writes = _cache_usage(result.usage)
            if _has_cache_fields(result.usage):
                self.state.record_cache(session_id, result.deployment_id, prefix, cached_tokens, cache_writes, deployment.cache_ttl_seconds)
            self.state.record_outcome(session_id, result.deployment_id, "success")
            self._send_json(
                200,
                response,
                {
                    "X-Router-Deployment": result.deployment_id,
                    "X-Router-Group": deployment.chain_group,
                    "X-Router-Reason": plan.reason,
                },
            )
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": f"invalid request: {error}"})
        except (ProviderError, LookupError) as error:
            headers = {"X-Router-Error-Class": getattr(error, "error_class", "routing_failure")}
            retry_after = getattr(error, "retry_after_seconds", None)
            if retry_after is not None and retry_after > 0:
                headers["Retry-After"] = str(max(1, int(retry_after)))
            self._send_json(503, {"error": str(error)}, headers)

    def _handle_outcome(self) -> None:
        """Record an outcome and cooldown only after operational failures."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > self.max_body_bytes:
                self._send_json(413, {"error": "request body exceeds 1 MiB limit"})
                return
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            session_id = str(self.headers.get("X-Session-Id", "anonymous"))
            deployment = payload.get("deployment")
            outcome = payload.get("outcome")
            error_class = payload.get("error_class")
            if not isinstance(deployment, str) or not isinstance(outcome, str):
                raise ValueError("deployment and outcome are required strings")
            self.state.record_outcome(session_id, deployment, outcome, error_class)
            if outcome == "operational_failure":
                retry_after = _nonnegative_float(payload.get("retry_after_seconds"), 0.0)
                quota_reset_at = payload.get("quota_reset_at")
                try:
                    quota_reset_at = None if quota_reset_at is None else float(quota_reset_at)
                except (TypeError, ValueError):
                    quota_reset_at = None
                self.health.mark_failure(
                    deployment,
                    error_class=error_class if isinstance(error_class, str) else None,
                    retry_after_seconds=retry_after or None,
                    quota_reset_at=quota_reset_at,
                )
            elif outcome == "success":
                self.stall_states.setdefault(session_id, StallState()).mark_progress()
            self._send_json(200, {"recorded": outcome, "deployment": deployment})
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": f"invalid outcome: {error}"})

    def _send_json(self, status: int, payload: dict[str, object], headers: dict[str, str] | None = None) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_stream(
        self,
        content: object,
        model: str,
        deployment_id: str,
        message: dict[str, object] | None = None,
        finish_reason: str | None = None,
        usage: dict[str, object] | None = None,
    ) -> None:
        """Emit a compatibility SSE stream from an adapter completion.

        The current adapter protocol is completion-oriented, so this preserves
        the streaming wire contract while chunking after the provider returns.
        True upstream token streaming remains a future adapter capability.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Router-Deployment", deployment_id)
        self.end_headers()
        # Slice characters rather than words so code indentation and newlines
        # survive the compatibility streaming path unchanged.
        text = content if isinstance(content, str) else ""
        chunks = [text[index:index + 256] for index in range(0, len(text), 256)] or [""]
        for index, chunk in enumerate(chunks):
            delta = {"role": "assistant"} if index == 0 else {}
            if chunk or not message or not message.get("tool_calls"):
                delta["content"] = chunk
            if index == 0 and message:
                for field in ("tool_calls", "function_call"):
                    if field in message:
                        delta[field] = message[field]
            event = {"id": "router-local", "object": "chat.completion.chunk", "model": model,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
        if usage is not None:
            usage_event = {"id": "router-local", "object": "chat.completion.chunk", "model": model,
                           "choices": [], "usage": usage}
            self.wfile.write(f"data: {json.dumps(usage_event)}\n\n".encode("utf-8"))
        final = {"id": "router-local", "object": "chat.completion.chunk", "model": model,
                 "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason or "stop"}]}
        self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode("utf-8"))

    def _send_provider_stream(
        self,
        features: RequestFeatures,
        prompt: str,
        model: str,
        provider_request: ProviderRequest,
        on_failure,
        cache_hints,
        preferred,
    ) -> tuple[str, dict[str, object] | None] | None:
        deployment, payloads = stream_execute(
            features,
            self.deployments,
            self.adapters,
            self.health,
            prompt,
            provider_request,
            on_failure,
            cache_hints,
            preferred,
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Router-Deployment", deployment.deployment_id)
        self.send_header("X-Router-Group", deployment.chain_group)
        self.end_headers()
        usage = None
        completed = False
        try:
            for payload in payloads:
                if payload == "[DONE]":
                    self.wfile.write(b"data: [DONE]\n\n")
                    completed = True
                    break
                event = validate_stream_payload(payload)
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                event["model"] = model
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
                self.wfile.flush()
            if not completed:
                raise ProviderError("provider stream ended before DONE", error_class="invalid_response")
            self.wfile.flush()
        except ProviderError as error:
            on_failure(deployment.deployment_id, error.error_class)
            self.health.mark_failure(deployment.deployment_id, error_class=error.error_class)
            self.close_connection = True
            return None
        except (BrokenPipeError, ConnectionResetError):
            on_failure(deployment.deployment_id, "network_error")
            self.health.mark_failure(deployment.deployment_id, error_class="network_error")
            self.close_connection = True
            return None
        finally:
            close_stream(payloads)
        self.health.mark_success(deployment.deployment_id)
        return deployment.deployment_id, usage

    def _deployment(self, deployment_id: str):
        """Resolve a selected deployment for metadata persistence."""
        for deployment in self.deployments:
            if deployment.deployment_id == deployment_id:
                return deployment
        raise LookupError("selected deployment is not configured")

    def log_message(self, format: str, *args: object) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Run the local route-inspection server."""
    ThreadingHTTPServer((host, port), RouterHandler).serve_forever()


if __name__ == "__main__":
    serve(
        host=os.environ.get("ROUTER_HOST", "127.0.0.1"),
        port=int(os.environ.get("ROUTER_PORT", "8080")),
    )
