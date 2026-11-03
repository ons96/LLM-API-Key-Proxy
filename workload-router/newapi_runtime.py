"""Opt-in, quota-accounted execution through private New-API aliases only."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import time
from typing import Callable, Iterator
from urllib.parse import urlsplit

from provider_adapter import (
    OpenAICompatibleAdapter, ProviderCompletion, ProviderError, ProviderRequest,
    _normalize_adapter_exception, close_stream, validate_stream_payload,
)
from quota_ledger import QuotaLedger, Route, nonnegative, token_count
from route_policy import STEPS, Scored, Weights, Work, compact_json_tools, rank_routes
from router_core import Capability, Deployment
from router_state import RouterState


LOG = logging.getLogger(__name__)
PROFILE_ALIASES = {"auto-coding": "auto-coding", "auto-code": "auto-coding", "auto-chat": "auto-chat"}


def strict_json(raw: bytes) -> dict:
    """Reject duplicate keys, constants and non-object request/catalog bodies."""
    def unique(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate JSON keys")
        return result

    def constant(_value):
        raise ValueError("non-finite JSON")

    payload = json.loads(raw, object_pairs_hook=unique, parse_constant=constant)
    if not isinstance(payload, dict):
        raise ValueError("JSON object required")
    return payload


def validate_endpoint(base_url: str) -> str:
    """Require TLS except on an explicit loopback interservice link."""
    if not isinstance(base_url, str) or len(base_url) > 2048 or any(char.isspace() for char in base_url):
        raise ValueError("invalid New-API endpoint")
    parsed = urlsplit(base_url)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("invalid New-API endpoint")
    parsed.port  # Validate malformed port spellings before any dispatch.
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("New-API requires HTTPS outside loopback")
    return base_url.rstrip("/")


def validate_key(value: str) -> str:
    """Reject empty, non-ASCII and header-injecting credentials without printing them."""
    if (not isinstance(value, str) or not 1 <= len(value) <= 4096
            or any(not 33 <= ord(char) <= 126 for char in value)):
        raise ValueError("invalid service token")
    return value


@dataclass(frozen=True)
class Prepared:
    work: Work
    request: ProviderRequest
    baseline_tokens: int
    compressed_tokens: int
    prefix: str
    model: str


@dataclass(frozen=True)
class Result:
    route: Route
    decision_id: str
    attempts: tuple[str, ...]
    completion: ProviderCompletion


def prepare(payload: dict, session_id: str = "", step: str = "", *,
            default_output: int = 2048, compress: bool = False) -> Prepared:
    """Reserve a conservative text bound, including tools, and forward the output cap."""
    ProviderRequest.from_payload(payload)
    model = payload.get("model")
    if not isinstance(model, str) or model not in PROFILE_ALIASES:
        raise ValueError("model must be auto-coding or auto-chat")
    if (not isinstance(session_id, str) or len(session_id) > 256
            or any(ord(char) < 32 for char in session_id)):
        raise ValueError("invalid session id")
    if not isinstance(step, str):
        raise ValueError("invalid route step")
    profile = PROFILE_ALIASES[model]
    step = step.lower().replace("_", "-") or ("implement" if profile == "auto-coding" else "chat")
    if step not in STEPS or not isinstance(payload.get("stream", False), bool):
        raise ValueError("invalid step or stream flag")
    for name in ("n", "best_of"):
        if name in payload and token_count(payload[name], name) != 1:
            raise ValueError("multiple generations require separate reservations")
    limits = [token_count(payload[name], name) for name in ("max_tokens", "max_completion_tokens") if name in payload]
    if len(set(limits)) > 1:
        raise ValueError("conflicting output token limits")
    output = limits[0] if limits else token_count(default_output, "default_output")
    if not 0 < output <= 131072:
        raise ValueError("output token limit must be between 1 and 131072")
    forwarded = dict(payload)
    if not limits:
        forwarded["max_tokens"] = output
    if forwarded.get("stream", False):
        options = forwarded.get("stream_options")
        if options is not None and not isinstance(options, dict):
            raise ValueError("stream_options must be an object")
        forwarded["stream_options"] = {**(options or {}), "include_usage": True}
    compacted, baseline, compact = compact_json_tools(forwarded, compress)
    tools = compacted.get("tools") or []
    web = "web_search_options" in compacted or any(
        str(tool.get("type", "")).startswith("web_search") for tool in tools
    )
    work = Work(profile, step, compact, output, bool(tools),
                compacted.get("response_format") is not None, web,
                compacted.get("stream", False), session_id)
    # Include all tool definitions even when the message prefix is long.
    prefix = json.dumps({"messages": compacted["messages"]}, sort_keys=True, ensure_ascii=False)[:4096]
    prefix += json.dumps(tools, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return Prepared(work, ProviderRequest.from_payload(compacted), baseline, compact, prefix, model)


def usable_completion(result: ProviderCompletion) -> bool:
    """Text or well-formed function calls count as output; reasoning alone does not."""
    if not isinstance(result, ProviderCompletion) or not isinstance(result.message, dict):
        return False
    calls = result.message.get("tool_calls")
    if calls is not None:
        if not isinstance(calls, list) or not calls:
            return False
        for call in calls:
            if not isinstance(call, dict) or call.get("type") != "function":
                return False
            function = call.get("function")
            if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
                return False
            try:
                arguments = json.loads(function.get("arguments", ""))
                if not isinstance(arguments, dict):
                    return False
                json.dumps(arguments, allow_nan=False)
            except (ValueError, TypeError, RecursionError):
                return False
        return True
    return isinstance(result.content, str) and bool(result.content.strip())


class Runtime:
    """The server must audit aliases and RetryTimes before constructing this runtime."""

    def __init__(self, ledger: QuotaLedger, state: RouterState, base_url: str,
                 api_key: str, *, free_only: bool = True, timeout: float = 30,
                 deadline: float = 90, max_attempts: int = 8,
                 weights: Weights = Weights(), explore: bool = True) -> None:
        self.ledger, self.state = ledger, state
        self.base_url, self.api_key = validate_endpoint(base_url), validate_key(api_key)
        timeout, deadline = nonnegative(timeout, "timeout"), nonnegative(deadline, "deadline")
        if not 0 < timeout <= deadline <= 180 or not 1 <= token_count(max_attempts, "max_attempts") <= 32:
            raise ValueError("invalid timeout or attempt budget")
        if not isinstance(free_only, bool) or not isinstance(explore, bool):
            raise ValueError("runtime flags must be booleans")
        self.free_only, self.explore, self.weights = free_only, explore, weights
        self.timeout, self.deadline = timeout, deadline
        self.max_attempts = token_count(max_attempts, "max_attempts")

    def adapter(self, remaining: float) -> OpenAICompatibleAdapter:
        adapter = OpenAICompatibleAdapter("newapi", base_url=self.base_url,
                                          timeout=min(self.timeout, remaining),
                                          response_limit=8 * 1024 * 1024, allow_redirects=False)
        adapter.api_key = self.api_key
        return adapter

    def ranked(self, prepared: Prepared, *, routes: list[Route] | None = None,
               explore: bool | None = None) -> list[Scored]:
        coding = prepared.work.profile == "auto-coding" and bool(prepared.work.session_id)
        preferred = self.state.preferred(prepared.work.session_id, "auto-coding") if coding else None
        hints = self.state.cache_hints(prepared.work.session_id, prepared.prefix) if coding else {}
        cached = {key: hint.cached_tokens for key, hint in hints.items() if hint.is_fresh(time.time())}
        ranked = rank_routes(self.ledger.routes() if routes is None else routes, prepared.work,
                             self.ledger, free_only=self.free_only, weights=self.weights,
                             preferred_id=preferred, cache_tokens=cached,
                             explore=self.explore if explore is None else explore)
        seen: set[tuple[str, str, str]] = set()
        unique = []
        for item in ranked:
            identity = (item.route.provider_id, item.route.model_id, item.route.key_id)
            if identity not in seen:
                unique.append(item)
                seen.add(identity)
        return unique

    @staticmethod
    def deployment(route: Route) -> Deployment:
        return Deployment(route.route_id, Capability.FAST_GENERAL,
                          route.context_window or 1, provider=route.provider_id, model=route.alias)

    def reserve(self, item: Scored, prepared: Prepared, benchmark_id: str | None) -> str | None:
        work = prepared.work
        return self.ledger.reserve(item.route, work.input_tokens, work.output_tokens,
                                   profile=work.profile, step=work.step, session_id=work.session_id,
                                   score=item.score, baseline_tokens=prepared.baseline_tokens,
                                   compressed_tokens=prepared.compressed_tokens,
                                   streaming=work.streaming, benchmark_id=benchmark_id)

    def failed(self, identity: str, route: Route, exc: ProviderError, started: float, *,
               ttft: float | None = None, reasoning_ttft: float | None = None) -> None:
        # A failed/truncated stream's partial usage is not a final bill.
        self.ledger.finish(identity, route, success=False,
                           latency_ms=(time.monotonic() - started) * 1000,
                           ttft_ms=ttft, reasoning_ttft_ms=reasoning_ttft,
                           error_class=exc.error_class)
        hints = exc.limit_hints if isinstance(exc.limit_hints, dict) else {}
        self.ledger.learn_limits(route, {key: value for key, value in hints.items()
                                       if key in {"rpm", "rpd", "tpm", "tpd"}
                                       and type(value) in {int, float} and 0 < value < 1e12})
        self.ledger.cooldown(route, exc.error_class, retry_after=exc.retry_after_seconds,
                             reset_at=exc.quota_reset_at, status_code=exc.status_code)
        LOG.warning("timestamp=%s provider=%s model=%s error=%s", time.time(),
                    route.provider_id, route.model_id, exc.error_class)

    def remembered(self, prepared: Prepared, route: Route, usage: dict | None) -> None:
        if prepared.work.profile != "auto-coding" or not prepared.work.session_id:
            return
        self.state.set_preferred(prepared.work.session_id, "auto-coding", route.route_id)
        if not isinstance(usage, dict):
            return
        details = usage.get("prompt_tokens_details")
        raw = details.get("cached_tokens", 0) if isinstance(details, dict) else usage.get("cache_read_input_tokens", 0)
        try:
            cached = token_count(raw, "cached_tokens")
        except ValueError:
            return
        self.state.record_cache(prepared.work.session_id, route.route_id, prepared.prefix,
                                cached_tokens=cached, ttl_seconds=route.cache_ttl_seconds)

    def attempts(self, prepared: Prepared, routes: list[Route] | None,
                 benchmark_id: str | None, end: float, identities: list[str]):
        """Cancel setup failures only; started dispatches must be settled by the caller."""
        for item in self.ranked(prepared, routes=routes):
            if time.monotonic() >= end or len(identities) >= self.max_attempts:
                return
            identity = self.reserve(item, prepared, benchmark_id)
            if identity is None:
                continue
            identities.append(identity)
            try:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    self.ledger.cancel(identity)
                    return
                adapter = self.adapter(remaining)
            except Exception:
                self.ledger.cancel(identity)
                continue
            except BaseException:
                self.ledger.cancel(identity)
                raise
            self.ledger.start(identity)
            yield item.route, identity, adapter, time.monotonic()

    def complete(self, prepared: Prepared, *, routes: list[Route] | None = None,
                 grader: Callable[[ProviderCompletion], bool] | None = None,
                 benchmark_id: str | None = None) -> Result:
        if prepared.work.streaming:
            raise ValueError("use open_stream for a streaming request")
        end = time.monotonic() + self.deadline
        identities: list[str] = []
        last = ProviderError("no feasible New-API route", error_class="provider_error")
        for route, identity, adapter, started in self.attempts(prepared, routes, benchmark_id, end, identities):
            try:
                result = adapter.complete(self.deployment(route), prepared.request)
                if time.monotonic() >= end:
                    raise ProviderError("completion deadline exceeded", error_class="timeout")
                if not usable_completion(result):
                    raise ProviderError("provider returned no usable output", error_class="invalid_response")
            except Exception as exc:
                last = exc if isinstance(exc, ProviderError) else _normalize_adapter_exception(exc)
                self.failed(identity, route, last, started)
                continue
            except BaseException:
                self.failed(identity, route, ProviderError("request cancelled"), started)
                raise
            # Settle before calling a grader: a broken grader cannot strand a lease.
            self.ledger.finish(identity, route, success=True, usage=result.usage,
                               latency_ms=(time.monotonic() - started) * 1000)
            if grader is not None:
                passed = grader(result)
                if not isinstance(passed, bool):
                    raise ValueError("grader must return a boolean")
                self.ledger.grade(identity, passed)
                if not passed:
                    continue
            self.remembered(prepared, route, result.usage)
            return Result(route, identity, tuple(identities), result)
        raise ProviderError("no feasible New-API route completed", error_class=last.error_class,
                            retry_after_seconds=last.retry_after_seconds) from None

    def open_stream(self, prepared: Prepared, *, routes: list[Route] | None = None,
                    benchmark_id: str | None = None) -> AccountedStream:
        if not prepared.work.streaming:
            raise ValueError("stream must be enabled")
        end = time.monotonic() + self.deadline
        identities: list[str] = []
        last = ProviderError("no feasible New-API stream", error_class="provider_error")
        for route, identity, adapter, started in self.attempts(prepared, routes, benchmark_id, end, identities):
            stream = None
            try:
                stream = AccountedStream(self, route, identity, tuple(identities), prepared, started, end)
                stream.source = iter(adapter.stream(self.deployment(route), prepared.request))
                stream.prime()
                return stream
            except Exception as exc:
                last = exc if isinstance(exc, ProviderError) else _normalize_adapter_exception(exc)
                if stream is None:
                    self.failed(identity, route, last, started)
                else:
                    stream.fail(last)
            except BaseException:
                if stream is not None:
                    stream.close()
                else:
                    self.failed(identity, route, ProviderError("request cancelled"), started)
                raise
        raise ProviderError("no feasible New-API stream completed", error_class=last.error_class,
                            retry_after_seconds=last.retry_after_seconds) from None


class AccountedStream(Iterator[str]):
    """Prime before headers; never retry after returning any downstream output."""

    max_buffer_bytes = 1024 * 1024

    def __init__(self, runtime: Runtime, route: Route, identity: str, attempts: tuple[str, ...],
                 prepared: Prepared, started: float, end: float) -> None:
        self.runtime, self.route, self.identity, self.attempts = runtime, route, identity, attempts
        self.prepared, self.started, self.end = prepared, started, end
        self.source = iter(())
        self.buffer: list[str] = []
        self.buffer_index = 0
        self.usage: dict | None = None
        self.ttft: float | None = None
        self.reasoning_ttft: float | None = None
        self.usable = self.settled = False

    def prime(self) -> None:
        size = 0
        while not self.usable:
            value = self.read()
            size += len(value.encode("utf-8"))
            if size > self.max_buffer_bytes:
                raise ProviderError("stream prefix exceeds buffer limit", error_class="invalid_response")
            self.buffer.append(value)

    def __iter__(self) -> AccountedStream:
        return self

    def __next__(self) -> str:
        if self.buffer_index < len(self.buffer):
            value = self.buffer[self.buffer_index]
            self.buffer_index += 1
            return value
        self.buffer.clear()
        self.buffer_index = 0
        if self.settled:
            raise StopIteration
        return self.read()

    def read(self) -> str:
        try:
            if time.monotonic() >= self.end:
                raise ProviderError("stream deadline exceeded", error_class="timeout")
            value = next(self.source)
            if time.monotonic() >= self.end:
                raise ProviderError("stream deadline exceeded", error_class="timeout")
            event = validate_stream_payload(value)
            elapsed = (time.monotonic() - self.started) * 1000
            if event is None:
                if not self.usable:
                    raise ProviderError("stream has no text or tool output", error_class="empty_stream")
                self.runtime.ledger.finish(self.identity, self.route, success=True, usage=self.usage,
                                           latency_ms=elapsed, ttft_ms=self.ttft,
                                           reasoning_ttft_ms=self.reasoning_ttft)
                self.settled = True
                close_stream(self.source)
                self.runtime.remembered(self.prepared, self.route, self.usage)
                return value
            # Reject constants such as NaN even when the generic adapter tolerates them.
            json.dumps(event, allow_nan=False)
            if isinstance(event.get("usage"), dict):
                self.usage = event["usage"]
            for choice in event["choices"]:
                if not isinstance(choice, dict) or not isinstance(choice.get("delta", {}), dict):
                    raise ValueError("invalid stream choice")
                delta = choice.get("delta", {})
                content = delta.get("content")
                if content is not None and not isinstance(content, str):
                    raise ValueError("invalid text delta")
                if content:
                    if self.ttft is None:
                        self.ttft = elapsed
                    self.usable = self.usable or bool(content.strip())
                calls = delta.get("tool_calls")
                if calls is not None:
                    if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
                        raise ValueError("invalid tool delta")
                    self.usable = self.usable or bool(calls)
                reasoning = delta.get("reasoning_content", delta.get("reasoning"))
                if reasoning is not None and not isinstance(reasoning, str):
                    raise ValueError("invalid reasoning delta")
                if reasoning and self.reasoning_ttft is None:
                    self.reasoning_ttft = elapsed
            return value
        except Exception as exc:
            failure = exc if isinstance(exc, ProviderError) else (
                ProviderError("stream ended before DONE", error_class="invalid_response")
                if isinstance(exc, StopIteration) else _normalize_adapter_exception(exc)
            )
            self.fail(failure)
            raise failure from None
        except BaseException:
            self.close()
            raise

    def fail(self, failure: ProviderError) -> None:
        if not self.settled:
            self.runtime.failed(self.identity, self.route, failure, self.started,
                                ttft=self.ttft, reasoning_ttft=self.reasoning_ttft)
            self.settled = True
        close_stream(self.source)

    def close(self) -> None:
        self.buffer.clear()
        self.fail(ProviderError("stream consumer disconnected", error_class="network_error"))
