"""Minimal local HTTP boundary for exercising the deterministic router."""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from router_core import ROUTER_VERSION, HealthRegistry, RequestFeatures, estimate_context_tokens, normalize_phase, select_with_stickiness
from router_config import default_deployments, load_deployments
from provider_adapter import ProviderAdapter, ProviderError, ProviderRequest, adapters_from_environment, execute, stream_execute
from router_state import RouterState
from stall_detector import StallState


DEFAULT_DEPLOYMENTS = load_deployments(os.environ.get("ROUTER_DEPLOYMENTS"))


class RouterHandler(BaseHTTPRequestHandler):
    """Expose route inspection without making an upstream provider call."""

    deployments = DEFAULT_DEPLOYMENTS
    state = RouterState(os.environ.get("ROUTER_STATE_DB", ":memory:"))
    health = HealthRegistry()
    max_body_bytes = 1_048_576
    adapters: dict[str, ProviderAdapter] = adapters_from_environment(DEFAULT_DEPLOYMENTS)
    model_aliases = {"auto", "auto-code", "auto-chat"}
    stall_states: dict[str, StallState] = {}

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
            features = RequestFeatures(
                prompt=prompt,
                context_tokens=int(payload.get("context_tokens", estimate_context_tokens(messages))),
                tools=bool(payload.get("tools")),
                structured_output=bool(payload.get("response_format")),
                phase=phase,
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
            deployment = select_with_stickiness(features, self.deployments, preferred, self.health)
            self.state.record(session_id, prompt, deployment.capability.value, deployment.deployment_id, deployment.provider, deployment.config_fingerprint)
            self.state.set_preferred(session_id, phase, deployment.deployment_id)
            self.state.record_cache(
                session_id,
                deployment.deployment_id,
                prompt[:4096],
                int(payload.get("cached_tokens", 0)),
                int(payload.get("cache_write_tokens", 0)),
            )
            response = {
                "deployment": deployment.deployment_id,
                "capability": deployment.capability.value,
                "escalation": stall.escalation_level,
            }
            self._send_json(200, response, {"X-Router-Deployment": deployment.deployment_id, "X-Router-Capability": deployment.capability.value})
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
            features = RequestFeatures(
                prompt=prompt,
                context_tokens=int(payload.get("context_tokens", estimate_context_tokens(messages))),
                tools=bool(payload.get("tools")),
                structured_output=bool(payload.get("response_format")),
                phase=phase,
            )
            session_id = self.headers.get("X-Session-Id", "anonymous")

            def record_provider_failure(deployment_id: str, error_class: str) -> None:
                self.state.record_outcome(session_id, deployment_id, "operational_failure", error_class)

            response_format = payload.get("response_format")
            if response_format is not None and not isinstance(response_format, dict):
                raise ValueError("response_format must be an object")
            if payload.get("stream") is True:
                try:
                    deployment_id = self._send_provider_stream(
                        features, prompt, model, provider_request, record_provider_failure
                    )
                    if deployment_id is not None:
                        self.state.record_outcome(session_id, deployment_id, "success")
                except ProviderError as error:
                    # Completion-only adapters retain the compatibility stream
                    # contract; native adapters stream without buffering.
                    if "streaming adapter" not in str(error):
                        raise
                    result = execute(
                        features,
                        self.deployments,
                        self.adapters,
                        self.health,
                        prompt,
                        provider_request,
                        record_provider_failure,
                    )
                    self._send_stream(result.content, model, result.deployment_id)
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
            self.state.record_outcome(session_id, result.deployment_id, "success")
            self._send_json(200, response, {"X-Router-Deployment": result.deployment_id})
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": f"invalid request: {error}"})
        except (ProviderError, LookupError) as error:
            self._send_json(503, {"error": str(error)})

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
                self.health.mark_failure(deployment)
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

    def _send_stream(self, content: str, model: str, deployment_id: str) -> None:
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
        chunks = [content[index:index + 256] for index in range(0, len(content), 256)] or [""]
        for index, chunk in enumerate(chunks):
            delta = {"role": "assistant"} if index == 0 else {}
            delta["content"] = chunk
            event = {"id": "router-local", "object": "chat.completion.chunk", "model": model,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
        final = {"id": "router-local", "object": "chat.completion.chunk", "model": model,
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode("utf-8"))

    def _send_provider_stream(
        self,
        features: RequestFeatures,
        prompt: str,
        model: str,
        provider_request: ProviderRequest,
        on_failure,
    ) -> str | None:
        deployment, payloads = stream_execute(
            features, self.deployments, self.adapters, self.health, prompt, provider_request, on_failure
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Router-Deployment", deployment.deployment_id)
        self.end_headers()
        try:
            for payload in payloads:
                if payload == "[DONE]":
                    self.wfile.write(b"data: [DONE]\n\n")
                    continue
                try:
                    event = json.loads(payload)
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ProviderError("provider returned malformed SSE JSON", error_class="invalid_response") from exc
                if not isinstance(event, dict):
                    raise ProviderError("provider returned malformed SSE event", error_class="invalid_response")
                event.setdefault("model", model)
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
            self.wfile.flush()
        except ProviderError as error:
            on_failure(deployment.deployment_id, error.error_class)
            self.close_connection = True
            return None
        except (BrokenPipeError, ConnectionResetError):
            on_failure(deployment.deployment_id, "network_error")
            self.close_connection = True
            return None
        return deployment.deployment_id

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
