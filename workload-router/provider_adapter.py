"""Provider-neutral execution boundary with bounded operational failover."""

from dataclasses import dataclass
from copy import deepcopy
from email.utils import parsedate_to_datetime
from http.client import IncompleteRead
import json
import math
import time
import os
from urllib import error, request
from typing import Callable, Protocol

from router_core import CacheHint, Deployment, HealthRegistry, RequestFeatures, route_plan, warm_cache_wait_seconds


class ProviderError(Exception):
    """An operational provider error eligible for same-group failover."""

    _ERROR_CLASSES = {
        "provider_error",
        "timeout",
        "rate_limit",
        "upstream_failure",
        "provider_rejection",
        "network_error",
        "invalid_response",
        "empty_stream",
        "no_adapter",
        "http_error",
    }

    def __init__(
        self,
        message: str,
        *,
        error_class: str = "provider_error",
        status_code: int | None = None,
        deployment_id: str | None = None,
        retry_after_seconds: float | None = None,
        quota_reset_at: float | None = None,
    ) -> None:
        super().__init__(message)
        self.error_class = error_class if error_class in self._ERROR_CLASSES else "provider_error"
        self.status_code = status_code
        self.deployment_id = deployment_id
        self.retry_after_seconds = _finite_nonnegative(retry_after_seconds)
        self.quota_reset_at = _finite_nonnegative(quota_reset_at)


ROUTER_ONLY_FIELDS = {
    "model",
    "stream",
    "context_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "model_group",
    "continuation",
    "cache_ttl_seconds",
}


@dataclass(frozen=True)
class ProviderRequest:
    """Validated request data forwarded to an OpenAI-compatible provider.

    The router owns the model alias, stream flag, and telemetry-only fields.
    Every other JSON request field is retained so provider-specific standard
    options are not silently discarded at the transport boundary.
    """

    messages: list[dict[str, object]]
    options: dict[str, object]

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "ProviderRequest":
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list of objects")
        if any(not isinstance(message, dict) for message in messages):
            raise ValueError("messages entries must be objects")
        response_format = payload.get("response_format")
        if response_format is not None and not isinstance(response_format, dict):
            raise ValueError("response_format must be an object")
        tools = payload.get("tools")
        if tools is not None and (not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools)):
            raise ValueError("tools must be a list of objects")
        return cls(
            messages=[deepcopy(message) for message in messages],
            options=deepcopy({key: value for key, value in payload.items() if key not in ROUTER_ONLY_FIELDS and key != "messages"}),
        )

    def to_payload(self, model: str, stream: bool) -> dict[str, object]:
        """Build an upstream payload without exposing the router model alias."""
        payload = {"model": model, "messages": deepcopy(self.messages), "stream": stream}
        payload.update(deepcopy(self.options))
        return payload


@dataclass(frozen=True)
class ProviderCompletion:
    """Normalized completion data, including tool-call messages when present."""

    message: dict[str, object]
    finish_reason: str | None = None
    usage: dict[str, object] | None = None

    @property
    def content(self) -> object:
        return self.message.get("content", "")


class ProviderAdapter(Protocol):
    def complete(self, deployment: Deployment, provider_request: ProviderRequest) -> ProviderCompletion | str:
        """Complete a request through one concrete deployment."""

    def stream(self, deployment: Deployment, provider_request: ProviderRequest):
        """Yield provider-native SSE data payloads when supported."""


class OpenAICompatibleAdapter:
    """Small standard-library adapter for an OpenAI-compatible endpoint.

    Endpoint and credentials are supplied by environment variables named from
    the deployment provider, keeping secrets out of deployment profiles.
    """

    def __init__(
        self,
        provider: str,
        timeout: float = 30.0,
        env_prefix: str | None = None,
        base_url: str | None = None,
    ) -> None:
        prefix = (env_prefix or provider).upper().replace("-", "_")
        self.base_url = (base_url or os.environ.get(f"{prefix}_BASE_URL", "")).rstrip("/")
        self.api_key = os.environ.get(f"{prefix}_API_KEY", "")
        self.timeout = timeout

    def complete(self, deployment: Deployment, provider_request: ProviderRequest) -> ProviderCompletion:
        if not self.base_url:
            raise ProviderError(f"missing {deployment.provider} base URL")
        payload = json.dumps(provider_request.to_payload(deployment.model, stream=False)).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(f"{self.base_url}/chat/completions", data=payload, headers=headers)
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                body = json.loads(response.read())
        except error.HTTPError as exc:
            retry_after, quota_reset = _retry_hints(exc.headers)
            raise ProviderError(
                "OpenAI-compatible request failed",
                error_class=_http_error_class(exc.code),
                status_code=exc.code,
                retry_after_seconds=retry_after,
                quota_reset_at=quota_reset,
            ) from exc
        except error.URLError as exc:
            category = "timeout" if isinstance(exc.reason, TimeoutError) else "network_error"
            raise ProviderError("OpenAI-compatible request failed", error_class=category) from exc
        except TimeoutError as exc:
            raise ProviderError("OpenAI-compatible request timed out", error_class="timeout") from exc
        except (OSError, IncompleteRead) as exc:
            raise ProviderError("OpenAI-compatible request failed", error_class="network_error") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProviderError("provider returned invalid JSON", error_class="invalid_response") from exc
        try:
            choice = body["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict):
                raise TypeError("message must be an object")
            finish_reason = choice.get("finish_reason")
            usage = body.get("usage")
            return ProviderCompletion(
                message=dict(message),
                finish_reason=None if finish_reason is None else str(finish_reason),
                usage=usage if isinstance(usage, dict) else None,
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("provider returned an invalid completion", error_class="invalid_response") from exc

    def stream(self, deployment: Deployment, provider_request: ProviderRequest):
        """Yield upstream SSE data lines for providers that support streaming."""
        if not self.base_url:
            raise ProviderError(f"missing {deployment.provider} base URL")
        payload = json.dumps(provider_request.to_payload(deployment.model, stream=True)).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(f"{self.base_url}/chat/completions", data=payload, headers=headers)
        try:
            response = request.urlopen(req, timeout=self.timeout)
        except error.HTTPError as exc:
            retry_after, quota_reset = _retry_hints(exc.headers)
            raise ProviderError(
                "OpenAI-compatible stream failed",
                error_class=_http_error_class(exc.code),
                status_code=exc.code,
                retry_after_seconds=retry_after,
                quota_reset_at=quota_reset,
            ) from exc
        except error.URLError as exc:
            category = "timeout" if isinstance(exc.reason, TimeoutError) else "network_error"
            raise ProviderError("OpenAI-compatible stream failed", error_class=category) from exc
        except TimeoutError as exc:
            raise ProviderError("OpenAI-compatible stream timed out", error_class="timeout") from exc
        except OSError as exc:
            raise ProviderError("OpenAI-compatible stream failed", error_class="network_error") from exc
        try:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    validate_stream_payload(payload)
                    yield payload
                    if payload == "[DONE]":
                        return
        except TimeoutError as exc:
            raise ProviderError("OpenAI-compatible stream timed out", error_class="timeout") from exc
        except (OSError, IncompleteRead) as exc:
            raise ProviderError("OpenAI-compatible stream failed", error_class="network_error") from exc
        except UnicodeDecodeError as exc:
            raise ProviderError("provider returned invalid SSE encoding", error_class="invalid_response") from exc
        finally:
            close_stream(response)


def adapters_from_environment(deployments: list[Deployment]) -> dict[str, ProviderAdapter]:
    """Create adapters only for profiles with configured provider URLs."""
    adapters: dict[str, ProviderAdapter] = {}
    for deployment in deployments:
        if deployment.provider == "unconfigured":
            continue
        adapter = OpenAICompatibleAdapter(
            deployment.provider,
            env_prefix=deployment.env_prefix or None,
            base_url=deployment.base_url or None,
        )
        if adapter.base_url:
            adapters[deployment.deployment_id] = adapter
    return adapters


@dataclass(frozen=True)
class ExecutionResult:
    deployment_id: str
    content: object
    attempts: int
    message: dict[str, object] | None = None
    finish_reason: str | None = None
    usage: dict[str, object] | None = None
    failures: tuple[tuple[str, str], ...] = ()


def _http_error_class(status_code: int) -> str:
    """Map provider HTTP statuses to bounded operational telemetry classes."""
    if status_code == 429:
        return "rate_limit"
    if status_code in {408, 425}:
        return "timeout"
    if 500 <= status_code <= 599:
        return "upstream_failure"
    if 400 <= status_code <= 499:
        return "provider_rejection"
    return "http_error"


def _finite_nonnegative(value: object) -> float | None:
    """Discard non-finite provider hints before cooldown/header calculations."""
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return max(0.0, parsed) if math.isfinite(parsed) else None


def _retry_hints(headers) -> tuple[float | None, float | None]:
    """Parse bounded provider retry/quota hints without retaining response text."""
    if headers is None:
        return None, None
    retry_after: float | None = None
    raw_retry = headers.get("Retry-After")
    if raw_retry:
        try:
            retry_after = _finite_nonnegative(float(raw_retry))
        except (TypeError, ValueError):
            try:
                retry_after = max(0.0, parsedate_to_datetime(raw_retry).timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                retry_after = None
    quota_reset: float | None = None
    raw_reset = headers.get("X-RateLimit-Reset") or headers.get("RateLimit-Reset")
    if raw_reset:
        try:
            parsed = _finite_nonnegative(raw_reset)
            if parsed is not None:
                # X-RateLimit-Reset commonly uses Unix seconds (sometimes ms).
                # A past Unix timestamp must not become a decades-long delta.
                if parsed >= 1_000_000_000:
                    quota_reset = parsed / 1000 if parsed >= 1_000_000_000_000 else parsed
                else:
                    quota_reset = time.time() + parsed
        except (TypeError, ValueError, OverflowError):
            quota_reset = None
    return retry_after, quota_reset


def validate_stream_payload(payload: str) -> dict[str, object] | None:
    """Validate an SSE data event before committing downstream headers/data."""
    if payload == "[DONE]":
        return None
    try:
        event = json.loads(payload)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProviderError("provider returned malformed SSE JSON", error_class="invalid_response") from exc
    if not isinstance(event, dict) or "error" in event or not isinstance(event.get("choices"), list):
        raise ProviderError("provider returned malformed SSE event", error_class="invalid_response")
    return event


def _warm_wait(error: ProviderError, hint: CacheHint | None) -> float:
    """Respect the longer of quota reset and retry hints before cache retries."""
    delay = error.retry_after_seconds
    if error.quota_reset_at is not None:
        delay = max(delay or 0.0, error.quota_reset_at - time.time())
    return warm_cache_wait_seconds(error.error_class, delay, hint)


def _normalize_adapter_exception(exc: Exception) -> ProviderError:
    """Convert unexpected adapter failures into bounded failover categories."""
    if isinstance(exc, TimeoutError):
        return ProviderError("provider operation timed out", error_class="timeout")
    if isinstance(exc, OSError):
        return ProviderError("provider operation failed", error_class="network_error")
    if isinstance(exc, (TypeError, ValueError, UnicodeError, json.JSONDecodeError)):
        return ProviderError("provider returned an invalid response", error_class="invalid_response")
    return ProviderError("provider operation failed", error_class="provider_error")


def close_stream(payloads) -> None:
    """Close a provider iterator without masking the original failure."""
    if payloads is None or not hasattr(payloads, "close"):
        return
    try:
        payloads.close()
    except Exception:
        return


def _prime_stream(stream, deployment: Deployment, provider_request: ProviderRequest):
    """Create and prime an adapter stream while normalizing pre-header faults."""
    payloads = None
    try:
        payloads = iter(stream(deployment, provider_request))
        return payloads, next(payloads)
    except (ProviderError, StopIteration):
        close_stream(payloads)
        raise
    except Exception as exc:
        close_stream(payloads)
        raise _normalize_adapter_exception(exc) from exc


def _as_execution_result(
    deployment: Deployment,
    result: ProviderCompletion | str,
    attempts: int,
    failures: tuple[tuple[str, str], ...] = (),
) -> ExecutionResult:
    if isinstance(result, ProviderCompletion):
        return ExecutionResult(
            deployment_id=deployment.deployment_id,
            content=result.content,
            attempts=attempts,
            message=result.message,
            finish_reason=result.finish_reason,
            usage=result.usage,
            failures=failures,
        )
    return ExecutionResult(deployment.deployment_id, result, attempts, failures=failures)


def execute(
    features: RequestFeatures,
    deployments: list[Deployment],
    adapters: dict[str, ProviderAdapter],
    health: HealthRegistry,
    prompt: str,
    provider_request: ProviderRequest | None = None,
    on_failure: Callable[[str, str], None] | None = None,
    cache_hints: dict[str, CacheHint] | None = None,
    preferred_id: str | None = None,
) -> ExecutionResult:
    """Try the ordered same-group chain once each, then fail immediately."""
    plan = route_plan(features, deployments, preferred_id=preferred_id, health=health, cache_hints=cache_hints)
    ordered = plan.candidates
    group = ordered[0].chain_group
    attempts = 0
    failures: list[tuple[str, str]] = []
    last_error: ProviderError | None = None
    for deployment in ordered:
        adapter = adapters.get(deployment.deployment_id)
        if adapter is None:
            continue
        attempts += 1
        try:
            request_data = provider_request or ProviderRequest(
                messages=[{"role": "user", "content": prompt}], options={}
            )
            try:
                upstream_result = adapter.complete(deployment, request_data)
            except ProviderError:
                raise
            except Exception as exc:
                raise _normalize_adapter_exception(exc) from exc
            result = _as_execution_result(deployment, upstream_result, attempts, tuple(failures))
            health.mark_success(deployment.deployment_id)
            return result
        except ProviderError as error:
            last_error = error
            health.mark_failure(
                deployment.deployment_id,
                error_class=error.error_class,
                retry_after_seconds=error.retry_after_seconds,
                quota_reset_at=error.quota_reset_at,
            )
            error_class = error.error_class
            failures.append((deployment.deployment_id, error_class))
            if on_failure is not None:
                on_failure(deployment.deployment_id, error_class)
            wait_seconds = _warm_wait(
                error,
                (cache_hints or {}).get(deployment.deployment_id),
            )
            if wait_seconds:
                time.sleep(wait_seconds)
                attempts += 1
                try:
                    request_data = provider_request or ProviderRequest(
                        messages=[{"role": "user", "content": prompt}], options={}
                    )
                    try:
                        upstream_result = adapter.complete(deployment, request_data)
                    except ProviderError:
                        raise
                    except Exception as exc:
                        raise _normalize_adapter_exception(exc) from exc
                    result = _as_execution_result(
                        deployment,
                        upstream_result,
                        attempts,
                        tuple(failures),
                    )
                    health.mark_success(deployment.deployment_id)
                    return result
                except ProviderError as retry_error:
                    last_error = retry_error
                    health.mark_failure(
                        deployment.deployment_id,
                        error_class=retry_error.error_class,
                        retry_after_seconds=retry_error.retry_after_seconds,
                        quota_reset_at=retry_error.quota_reset_at,
                    )
                    failures.append((deployment.deployment_id, retry_error.error_class))
                    if on_failure is not None:
                        on_failure(deployment.deployment_id, retry_error.error_class)
    final_class = failures[-1][1] if failures else "no_adapter"
    raise ProviderError(
        f"all eligible deployments failed for {group}",
        error_class=final_class,
        retry_after_seconds=last_error.retry_after_seconds if last_error else None,
        quota_reset_at=last_error.quota_reset_at if last_error else None,
    )


def stream_execute(
    features: RequestFeatures,
    deployments: list[Deployment],
    adapters: dict[str, ProviderAdapter],
    health: HealthRegistry,
    prompt: str,
    provider_request: ProviderRequest | None = None,
    on_failure: Callable[[str, str], None] | None = None,
    cache_hints: dict[str, CacheHint] | None = None,
    preferred_id: str | None = None,
):
    """Return the first eligible provider stream; fail over before yielding."""
    plan = route_plan(features, deployments, preferred_id=preferred_id, health=health, cache_hints=cache_hints)
    ordered = plan.candidates
    group = ordered[0].chain_group
    last_error: ProviderError | None = None
    for deployment in ordered:
        adapter = adapters.get(deployment.deployment_id)
        stream = getattr(adapter, "stream", None) if adapter else None
        if stream is None:
            continue
        payloads = None
        try:
            request_data = provider_request or ProviderRequest(
                messages=[{"role": "user", "content": prompt}], options={}
            )
            payloads, first = _prime_stream(stream, deployment, request_data)
            if first == "[DONE]":
                raise ProviderError("provider returned an empty stream", error_class="empty_stream")
            validate_stream_payload(first)
            return deployment, _prepend_stream(first, payloads)
        except StopIteration:
            last_error = ProviderError("provider returned an empty stream", error_class="empty_stream")
            health.mark_failure(deployment.deployment_id, error_class="empty_stream")
            if on_failure is not None:
                on_failure(deployment.deployment_id, "empty_stream")
        except ProviderError as error:
            last_error = error
            health.mark_failure(
                deployment.deployment_id,
                error_class=error.error_class,
                retry_after_seconds=error.retry_after_seconds,
                quota_reset_at=error.quota_reset_at,
            )
            if on_failure is not None:
                on_failure(deployment.deployment_id, error.error_class)
            close_stream(payloads)
            wait_seconds = _warm_wait(
                error,
                (cache_hints or {}).get(deployment.deployment_id),
            )
            if wait_seconds:
                time.sleep(wait_seconds)
                try:
                    request_data = provider_request or ProviderRequest(
                        messages=[{"role": "user", "content": prompt}], options={}
                    )
                    payloads, first = _prime_stream(stream, deployment, request_data)
                    if first == "[DONE]":
                        raise ProviderError("provider returned an empty stream", error_class="empty_stream")
                    validate_stream_payload(first)
                    return deployment, _prepend_stream(first, payloads)
                except StopIteration:
                    last_error = ProviderError("provider returned an empty stream", error_class="empty_stream")
                    health.mark_failure(deployment.deployment_id, error_class="empty_stream")
                    if on_failure is not None:
                        on_failure(deployment.deployment_id, "empty_stream")
                except ProviderError as retry_error:
                    last_error = retry_error
                    health.mark_failure(
                        deployment.deployment_id,
                        error_class=retry_error.error_class,
                        retry_after_seconds=retry_error.retry_after_seconds,
                        quota_reset_at=retry_error.quota_reset_at,
                    )
                    if on_failure is not None:
                        on_failure(deployment.deployment_id, retry_error.error_class)
                    close_stream(payloads)
    raise ProviderError(
        f"no streaming adapter available for {group}",
        error_class=last_error.error_class if last_error else "no_adapter",
        retry_after_seconds=last_error.retry_after_seconds if last_error else None,
        quota_reset_at=last_error.quota_reset_at if last_error else None,
    )


def _prepend_stream(first: str, payloads):
    """Keep the upstream generator closeable after priming for safe failover."""
    try:
        yield first
        yield from payloads
    except ProviderError:
        raise
    except Exception as exc:
        raise _normalize_adapter_exception(exc) from exc
    finally:
        close_stream(payloads)
