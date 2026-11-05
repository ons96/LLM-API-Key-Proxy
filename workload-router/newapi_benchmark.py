"""Bounded completion microbenchmark with deterministic, non-executing graders."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
import time
import uuid

from newapi_runtime import Runtime, prepare, strict_json
from provider_adapter import ProviderCompletion, ProviderError
from quota_ledger import Route, token_count


# Capabilities a case may declare. An undeclared or unknown requirement fails
# closed before dispatch; live workloads only use capabilities verified from the
# provider's current catalog metadata.
KNOWN_CAPABILITIES = frozenset({"tools"})


@dataclass(frozen=True)
class Case:
    name: str
    step: str
    prompt: str
    expected: dict
    tool: bool = False
    requires: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.requires, frozenset) or not self.requires <= KNOWN_CAPABILITIES:
            raise ValueError("case requires an unknown capability")

    def payload(self, stream: bool, *, output: int = 512, schema: bool = True) -> dict:
        """Keep tools intact; schema is optional for prompt-only JSON workloads."""
        payload = {"model": "auto-coding", "messages": [{"role": "user", "content": self.prompt}],
                   "max_tokens": output, "temperature": 0, "stream": stream}
        if self.tool:
            payload["tools"] = [{"type": "function", "function": {
                "name": "read_file", "description": "Read a bounded source-file range.",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string"}, "start": {"type": "integer"}, "count": {"type": "integer"}},
                    "required": ["path", "start", "count"], "additionalProperties": False}}}]
            payload["tool_choice"] = {"type": "function", "function": {"name": "read_file"}}
        elif schema:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def grade(self, completion: ProviderCompletion) -> bool:
        """Compare JSON values or exact tool arguments; never execute model output."""
        try:
            if self.tool:
                calls = completion.message.get("tool_calls")
                if not isinstance(calls, list) or len(calls) != 1:
                    return False
                call = calls[0]
                if call.get("type") != "function" or call.get("function", {}).get("name") != "read_file":
                    return False
                actual = strict_json(call["function"]["arguments"].encode())
            else:
                if not isinstance(completion.content, str):
                    return False
                actual = strict_json(completion.content.encode())
            # JSON booleans must not compare equal to integer counters.
            return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(self.expected, sort_keys=True)
        except (AttributeError, KeyError, TypeError, ValueError, RecursionError):
            return False


CASES = (
    Case("plan", "plan", 'Task plan: Return only a JSON object with key "steps". '
         'The safe change order is read, change, test. Preserve this exact order.',
         {"steps": ["read", "change", "test"]}),
    Case("edit", "implement", 'Task edit: Change timeout_seconds from 30 to 8 and keep retries at 0. '
         'Return only the resulting JSON object with keys timeout_seconds and retries.',
         {"timeout_seconds": 8, "retries": 0}),
    Case("debug", "debug", 'Task debug: The positive-number predicate incorrectly uses n >= 0. '
         'Return only a JSON object with keys operator and right. Exclude zero, keeping the right operand at 0.',
         {"operator": ">", "right": 0}),
    Case("tool", "explore", 'Task tool: Call read_file for path src/app.py, start line 12, count 4. '
         'Use those exact arguments, not a text description.',
         {"path": "src/app.py", "start": 12, "count": 4}, tool=True,
         requires=frozenset({"tools"})),
    Case("review", "final-review", 'Task review: discount(price, percent) returns price * (100 + percent) / 100 '
         'and never checks percent is between 0 and 100. Return only a JSON object with findings equal to '
         '["adds_instead_of_subtracts", "missing_percent_bounds"] in that order.',
         {"findings": ["adds_instead_of_subtracts", "missing_percent_bounds"]}),
)


def collect(stream) -> ProviderCompletion:
    """Reconstruct content and fragmented tool arguments solely for grading."""
    text = []
    calls: dict[int, dict] = {}
    try:
        for value in stream:
            if value == "[DONE]":
                break
            event = strict_json(value.encode())
            for choice in event["choices"]:
                delta = choice.get("delta", {})
                if delta.get("content"):
                    text.append(delta["content"])
                for part in delta.get("tool_calls", []):
                    index = token_count(part.get("index"), "tool index")
                    if index >= 128:
                        raise ValueError("tool index exceeds limit")
                    call = calls.setdefault(index, {"type": "function", "function": {"name": "", "arguments": ""}})
                    for field in ("id", "type"):
                        if field in part:
                            call[field] = part[field]
                    function = part.get("function", {})
                    for field in ("name", "arguments"):
                        if field in function:
                            call["function"][field] += function[field]
        message = {"role": "assistant", "content": "".join(text)}
        if calls:
            message["tool_calls"] = [calls[index] for index in sorted(calls)]
        return ProviderCompletion(message, usage=stream.usage)
    finally:
        stream.close()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def benchmark(runtime: Runtime, routes: list[Route], *, cases: tuple[Case, ...] = CASES,
              seed: int = 0, rounds: int = 1, stream: bool = True, max_requests: int = 50) -> dict:
    """Sample each feasible route, persist grades, then expose before/after rankings.

    This intentionally dispatches completions. It is not a filesystem/tool-loop
    benchmark and its timing must not be labelled as real-provider data when the
    runtime uses local fixtures. Paid/infeasible candidates are never dispatched.
    """
    count = token_count(rounds, "rounds")
    limit = token_count(max_requests, "max_requests")
    if not routes or not cases or not 1 <= count <= 10 or not 1 <= limit <= 300:
        raise ValueError("invalid benchmark workload or budget")
    if count * len(routes) * len(cases) > limit:
        raise ValueError("benchmark exceeds request budget")
    if not isinstance(seed, int) or isinstance(seed, bool) or not isinstance(stream, bool):
        raise ValueError("invalid benchmark seed or stream flag")
    identity = uuid.uuid4().hex
    task_hash = hashlib.sha256(json.dumps([case.payload(stream) for case in cases], sort_keys=True).encode()).hexdigest()
    with runtime.ledger.transaction() as db:
        db.execute("INSERT INTO benchmark_runs VALUES (?, ?, ?, ?, NULL)", (identity, time.time(), task_hash, seed))
    def order(case):
        work = prepare(case.payload(stream), step=case.step)
        return [row.route.route_id for row in runtime.ranked(work, routes=routes, explore=False)]
    before = {case.name: order(case) for case in cases}
    workload = [(case, route) for _round in range(count) for case in cases for route in routes]
    random.Random(seed).shuffle(workload)
    samples = []
    for case, route in workload:
        prepared = prepare(case.payload(stream), step=case.step)
        if not runtime.ranked(prepared, routes=[route], explore=False):
            samples.append({"case": case.name, "route_id": route.route_id, "skipped": "infeasible"})
            continue
        started = time.monotonic()
        decision_id = None
        failed = None
        passed = False
        try:
            if stream:
                source = runtime.open_stream(prepared, routes=[route], benchmark_id=identity)
                decision_id = source.identity
                completion = collect(source)
            else:
                result = runtime.complete(prepared, routes=[route], benchmark_id=identity)
                decision_id, completion = result.decision_id, result.completion
            passed = case.grade(completion)
            runtime.ledger.grade(decision_id, passed)
        except ProviderError as exc:
            failed = exc.error_class
        samples.append({"case": case.name, "route_id": route.route_id, "decision_id": decision_id,
                        "correct": passed, "error_class": failed,
                        "time_to_correct_ms": (time.monotonic() - started) * 1000 if passed else None})
    after = {case.name: order(case) for case in cases}
    with runtime.ledger.lock:
        rows = [dict(row) for row in runtime.ledger.db.execute(
            "SELECT * FROM routing_decisions WHERE benchmark_id=?", (identity,))]
    metrics = {}
    for route in routes:
        attempts = [row for row in rows if row["route_id"] == route.route_id]
        graded = [row for row in attempts if row["quality"] is not None]
        elapsed = [row["latency_ms"] for row in attempts if row["latency_ms"] is not None]
        correct_times = [row["time_to_correct_ms"] for row in samples
                         if row["route_id"] == route.route_id and row.get("time_to_correct_ms") is not None]
        metrics[route.route_id] = {"attempts": len(attempts), "graded": len(graded),
            "correct": sum(row["quality"] == 1 for row in graded),
            "transport_error_rate": sum(row["status"] == "failure" for row in attempts) / len(attempts) if attempts else None,
            "p50_latency_ms": percentile(elapsed, 0.5), "p95_latency_ms": percentile(elapsed, 0.95),
            "p50_time_to_correct_ms": percentile(correct_times, 0.5),
            "p50_ttft_ms": percentile([row["ttft_ms"] for row in attempts if row["ttft_ms"] is not None], 0.5),
            "estimated_usage_attempts": sum(row["usage_estimated"] == 1 for row in attempts)}
    summary = {"id": identity, "seed": seed, "task_hash": task_hash, "stream": stream,
               "before": before, "after": after, "metrics": metrics, "samples": samples,
               "ranking_changed": before != after}
    with runtime.ledger.transaction() as db:
        db.execute("UPDATE benchmark_runs SET summary=? WHERE id=?", (json.dumps(summary, allow_nan=False), identity))
    return summary
