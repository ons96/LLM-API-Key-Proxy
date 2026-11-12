"""Persistent quota reservations, catalog metadata and per-attempt telemetry.

Reservation and only-tighten limit-learning ideas follow FreeLLMAPI's MIT
quota engine. This implementation uses SQLite transactions, not its runtime.
No prompts, API keys, cookies or raw provider errors are stored.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Iterator
import uuid


AXES = {"rpm", "rpd", "tpm", "tpd", "credits", "money_day"}
QUOTA_TYPES = {"unlimited", "rate_limited", "credit_based", "paid", "unknown"}
# The canonical provider vocabulary also uses free_daily/free_one_time/no_key.
# These are the sidecar's funding classes; quota_type remains the feasibility
# mechanism and is deliberately not overloaded with price information.
FREE_CLASSES = {"free_unlimited", "free_limited", "free_credits", "paid", "unknown"}
STATUSES = {"reserved", "started", "success", "failure", "cancelled"}
# Closed client-verdict enum, identical to router_state.VERDICTS. Only `pass`
# may carry grade=true: a clean HTTP 200 without a verdict is not correctness.
VERDICTS = frozenset({
    "pass", "fail_build", "fail_lint", "fail_test",
    "fail_scope", "fail_tamper", "timeout", "unknown",
})
SCHEMA = """
CREATE TABLE IF NOT EXISTS providers (
    provider_id TEXT PRIMARY KEY, base_url TEXT, env_var TEXT, quota_type TEXT,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
    provider_id TEXT NOT NULL, model_id TEXT NOT NULL, metadata TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1, fetched_at REAL NOT NULL,
    PRIMARY KEY (provider_id, model_id)
);
CREATE TABLE IF NOT EXISTS routes (
    route_id TEXT PRIMARY KEY, metadata TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS quota_limits (
    provider_id TEXT NOT NULL, model_id TEXT NOT NULL, key_id TEXT NOT NULL,
    window_type TEXT NOT NULL, configured REAL, learned REAL, day_offset INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (provider_id, model_id, key_id, window_type)
);
CREATE TABLE IF NOT EXISTS routing_decisions (
    id TEXT PRIMARY KEY, route_id TEXT NOT NULL, provider_id TEXT NOT NULL,
    model_id TEXT NOT NULL, key_id TEXT NOT NULL, channel_id INTEGER,
    session_hash TEXT NOT NULL, profile TEXT NOT NULL, step TEXT NOT NULL,
    created_at REAL NOT NULL, completed_at REAL, status TEXT NOT NULL,
    score REAL, baseline_tokens INTEGER, compressed_tokens INTEGER,
    reserved_input INTEGER NOT NULL, reserved_output INTEGER NOT NULL,
    input_tokens INTEGER, output_tokens INTEGER, usage_estimated INTEGER,
    credit_used REAL, ttft_ms REAL, reasoning_ttft_ms REAL, latency_ms REAL,
    tokens_per_second REAL, error_class TEXT, quality INTEGER,
    streaming INTEGER NOT NULL DEFAULT 0, benchmark_id TEXT, verdict TEXT
);
CREATE INDEX IF NOT EXISTS decision_evidence ON routing_decisions(route_id, step, completed_at);
CREATE INDEX IF NOT EXISTS decision_pending ON routing_decisions(status, created_at);
CREATE TABLE IF NOT EXISTS quota_windows (
    decision_id TEXT NOT NULL REFERENCES routing_decisions(id),
    provider_id TEXT NOT NULL, model_id TEXT NOT NULL, key_id TEXT NOT NULL,
    window_type TEXT NOT NULL, occurred_at REAL NOT NULL, amount REAL NOT NULL,
    PRIMARY KEY (decision_id, provider_id, model_id, key_id, window_type)
);
CREATE INDEX IF NOT EXISTS quota_lookup ON quota_windows(
    provider_id, model_id, key_id, window_type, occurred_at
);
CREATE TABLE IF NOT EXISTS cooldowns (
    provider_id TEXT NOT NULL, model_id TEXT NOT NULL, key_id TEXT NOT NULL,
    until_at REAL NOT NULL, error_class TEXT NOT NULL,
    PRIMARY KEY (provider_id, model_id, key_id)
);
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id TEXT PRIMARY KEY, created_at REAL NOT NULL, task_hash TEXT NOT NULL,
    seed INTEGER NOT NULL, summary TEXT
);
"""


def nonnegative(value: object, name: str) -> float:
    """Reject invalid configuration instead of making an expensive route free."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return number


def token_count(value: object, name: str) -> int:
    number = nonnegative(value, name)
    if number != int(number):
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile, matching the gateway's TPS export."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


@dataclass(frozen=True)
class Route:
    """One private New-API alias with one accountable upstream credential."""

    route_id: str
    provider_id: str
    model_id: str
    key_id: str
    alias: str
    channel_id: int
    quota_type: str = "unknown"
    context_window: int | None = None
    output_limit: int | None = None
    tool_call: bool | None = None
    stream: bool | None = None
    structured_output: bool | None = None
    web_search: bool | None = None
    input_price: float | None = None
    output_price: float | None = None
    cached_input_price: float | None = None
    # Optional provider-credit tariff. This is used only for depletion and
    # ranking, never for the customer-facing cost estimate. It lets a free
    # one-time-credit route keep input/output_price=0.
    credit_input_price: float | None = None
    credit_output_price: float | None = None
    credit_balance: float | None = None
    daily_budget: float | None = None
    shadow_price: float = 1.0
    latency_ms: float = 2000.0
    prior_success: float = 2.0
    prior_failure: float = 2.0
    cache_ttl_seconds: float = 600.0
    prefill_ms_per_token: float = 0.015
    handoff_ms: float = 25.0
    profiles: tuple[str, ...] = ("auto-coding", "auto-chat")
    limits: dict[str, float] = field(default_factory=dict)
    provider_limits: dict[str, float] = field(default_factory=dict)
    day_offset: int = 0
    key_fingerprint: str | None = None
    channel_base_url: str | None = None

    def __post_init__(self) -> None:
        for name in ("route_id", "provider_id", "model_id", "key_id", "alias"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or len(value) > 512 or any(ord(c) < 32 for c in value):
                raise ValueError(f"invalid {name}")
        if self.quota_type not in QUOTA_TYPES:
            raise ValueError("unknown quota_type")
        if token_count(self.channel_id, "channel_id") < 1:
            raise ValueError("channel_id must be positive")
        for name in ("context_window", "output_limit"):
            value = getattr(self, name)
            if value is not None and token_count(value, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("tool_call", "stream", "structured_output", "web_search"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean or unknown")
        for name in ("input_price", "output_price", "cached_input_price",
                     "credit_input_price", "credit_output_price", "credit_balance", "daily_budget"):
            if getattr(self, name) is not None:
                nonnegative(getattr(self, name), name)
        for name in ("shadow_price", "latency_ms", "prior_success", "prior_failure", "cache_ttl_seconds", "prefill_ms_per_token", "handoff_ms"):
            nonnegative(getattr(self, name), name)
        if not self.prior_success or not self.prior_failure:
            raise ValueError("Beta priors must be positive")
        if isinstance(self.day_offset, bool) or not isinstance(self.day_offset, int) or not 0 <= self.day_offset < 86400:
            raise ValueError("day_offset must be UTC seconds within one day")
        if not self.profiles or not set(self.profiles) <= {"auto-coding", "auto-chat"}:
            raise ValueError("unknown routing profile")
        if self.model_id == "*" or self.key_id == "*" or any(c.isspace() or c == "," for c in self.alias):
            raise ValueError("route identities cannot use pool wildcards or ambiguous aliases")
        if self.key_fingerprint is not None and (
            not isinstance(self.key_fingerprint, str) or len(self.key_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in self.key_fingerprint)
        ):
            raise ValueError("invalid key fingerprint")
        if self.channel_base_url is not None and (
            not isinstance(self.channel_base_url, str) or not self.channel_base_url
            or len(self.channel_base_url) > 2048
        ):
            raise ValueError("invalid channel base URL")
        for limits in (self.limits, self.provider_limits):
            if not set(limits) <= AXES:
                raise ValueError("unknown window_type")
            for axis, value in limits.items():
                nonnegative(value, axis)

    @property
    def free_class(self) -> str:
        """Classify who pays and what kind of free allowance this route uses.

        `unlimited` means no finite credit allowance; it does not mean no rate
        limit. Fixed request/token limits still contribute quota pressure in
        scoring, without changing the funding class.
        """
        return {"unlimited": "free_unlimited", "rate_limited": "free_limited",
                "credit_based": "free_credits", "paid": "paid"}.get(self.quota_type, "unknown")

    def money(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        """Actual customer price in currency units per million tokens.

        A free route has zero prices here even when it has provider credits.
        Unknown pricing remains infeasible.
        """
        if self.input_price is None or self.output_price is None:
            return math.inf
        cached = min(input_tokens, max(0, cached_tokens))
        cache_price = self.input_price if self.cached_input_price is None else self.cached_input_price
        return ((input_tokens - cached) * self.input_price + cached * cache_price + output_tokens * self.output_price) / 1_000_000

    def credit_money(self, input_tokens: int, output_tokens: int) -> float:
        """Notional provider-credit value used for depletion and ranking only.

        If no credit tariff is configured, the actual price is used for
        compatibility with paid-style credit routes. Free credit routes must
        set the separate tariff so their zero customer price does not make the
        allowance look infinite.
        """
        input_price = self.credit_input_price if self.credit_input_price is not None else self.input_price
        output_price = self.credit_output_price if self.credit_output_price is not None else self.output_price
        if input_price is None or output_price is None:
            return math.inf
        return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


@dataclass(frozen=True)
class Capacity:
    feasible: bool
    reason: str = "available"
    pressure: float = 0.0
    retry_after: float = 0.0


class QuotaLedger:
    """SQLite is the cross-process lock; the RLock protects this connection."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            path = Path(path).expanduser().resolve()
            if not path.parent.is_dir():
                raise FileNotFoundError(path.parent)
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(descriptor)
            except FileExistsError:
                if path.stat().st_mode & 0o077:
                    raise PermissionError("ledger must be owner-only")
        self.db = sqlite3.connect(str(path), timeout=5, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript(SCHEMA)
        # Add the client-verdict column to ledgers created before S1 grading.
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(routing_decisions)")}
        if "verdict" not in columns:
            self.db.execute("ALTER TABLE routing_decisions ADD COLUMN verdict TEXT")
        self.lock = threading.RLock()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise

    def configure(self, route: Route) -> None:
        """Explicit limits may change; provider-learned ceilings still only tighten."""
        with self.transaction() as db:
            db.execute("INSERT OR REPLACE INTO routes VALUES (?, ?)", (route.route_id, json.dumps(asdict(route))))
            scopes = [(route.model_id, route.key_id, route.limits), ("*", "*", dict(route.provider_limits))]
            if route.credit_balance is not None:
                scopes[1][2]["credits"] = route.credit_balance
            if route.daily_budget is not None:
                scopes[1][2]["money_day"] = route.daily_budget
            for model, key, limits in scopes:
                for axis, value in limits.items():
                    db.execute(
                        """INSERT INTO quota_limits VALUES (?, ?, ?, ?, ?, NULL, ?)
                           ON CONFLICT(provider_id, model_id, key_id, window_type)
                           DO UPDATE SET configured=excluded.configured, day_offset=excluded.day_offset""",
                        (route.provider_id, model, key, axis, value, route.day_offset),
                    )

    def routes(self) -> list[Route]:
        with self.lock:
            return [Route(**json.loads(row[0])) for row in self.db.execute("SELECT metadata FROM routes ORDER BY route_id")]

    def learn_limits(self, route: Route, hints: dict[str, int | float]) -> None:
        with self.transaction() as db:
            for axis, value in hints.items():
                if axis not in {"rpm", "rpd", "tpm", "tpd"}:
                    continue
                number = nonnegative(value, axis)
                if not number:
                    continue
                db.execute(
                    """INSERT INTO quota_limits VALUES (?, ?, ?, ?, NULL, ?, ?)
                       ON CONFLICT(provider_id, model_id, key_id, window_type)
                       DO UPDATE SET learned=MIN(COALESCE(learned, excluded.learned), excluded.learned)""",
                    (route.provider_id, route.model_id, route.key_id, axis, number, route.day_offset),
                )

    def _limits(self, route: Route) -> list[sqlite3.Row]:
        return self.db.execute(
            """SELECT * FROM quota_limits WHERE provider_id=?
               AND model_id IN (?, '*') AND key_id IN (?, '*')""",
            (route.provider_id, route.model_id, route.key_id),
        ).fetchall()

    @staticmethod
    def _start(axis: str, now: float, offset: int) -> float:
        if axis in {"rpm", "tpm"}:
            return now - 60
        if axis in {"rpd", "tpd", "money_day"}:
            return math.floor((now - offset) / 86400) * 86400 + offset
        return 0.0

    def _usage(self, scope: tuple[str, str, str], axis: str, start: float) -> float:
        # Outstanding started attempts remain charged beyond a window boundary.
        comparison = ">" if axis in {"rpm", "tpm"} else ">="
        return self.db.execute(
            f"""SELECT COALESCE(SUM(q.amount), 0) FROM quota_windows q
               JOIN routing_decisions d ON d.id=q.decision_id
               WHERE q.provider_id=? AND q.model_id=? AND q.key_id=? AND q.window_type=?
               AND (q.occurred_at{comparison}? OR d.status IN ('reserved', 'started'))""",
            (*scope, axis, start),
        ).fetchone()[0]

    def _capacity(self, route: Route, input_tokens: int, output_tokens: int, now: float) -> Capacity:
        nonnegative(now, "now")
        if route.quota_type == "unknown":
            return Capacity(False, "unknown_quota_type")
        until = self.db.execute(
            """SELECT MAX(until_at) FROM cooldowns WHERE provider_id=?
               AND model_id IN (?, '*') AND key_id IN (?, '*')""",
            (route.provider_id, route.model_id, route.key_id),
        ).fetchone()[0]
        if until is not None and until > now:
            return Capacity(False, "cooldown", retry_after=until - now)
        amounts = self._amounts(route, input_tokens, output_tokens)
        if not math.isfinite(amounts["credits"]):
            return Capacity(False, "unknown_pricing")
        if route.quota_type == "credit_based" and route.credit_balance is None:
            return Capacity(False, "unknown_credit_balance")
        if (route.quota_type == "credit_based"
                and route.credit_input_price is None and route.credit_output_price is None
                and (route.input_price in (None, 0) or route.output_price in (None, 0))):
            return Capacity(False, "unknown_credit_price")
        pressure = 0.0
        limits = self._limits(route)
        if route.quota_type == "rate_limited" and not any(
            row["window_type"] in {"rpm", "rpd", "tpm", "tpd"}
            and (row["configured"] is not None or row["learned"] is not None)
            for row in limits
        ):
            return Capacity(False, "unknown_limits")
        for limit in limits:
            values = [value for value in (limit["configured"], limit["learned"]) if value is not None]
            if not values:
                continue
            ceiling = min(values)
            axis = limit["window_type"]
            scope = (limit["provider_id"], limit["model_id"], limit["key_id"])
            used = self._usage(scope, axis, self._start(axis, now, limit["day_offset"]))
            projected = used + amounts[axis]
            pressure = max(pressure, projected / ceiling if ceiling else 1.0)
            if projected > ceiling or ceiling == 0:
                delay = 60.0 if axis in {"rpm", "tpm"} else 86400.0
                return Capacity(False, f"exhausted_{axis}", min(1.0, pressure), delay)
        return Capacity(True, pressure=min(1.0, pressure))

    @staticmethod
    def _amounts(route: Route, input_tokens: int, output_tokens: int) -> dict[str, float]:
        cost = route.money(input_tokens, output_tokens)
        credit = route.credit_money(input_tokens, output_tokens)
        return {"rpm": 1.0, "rpd": 1.0, "tpm": input_tokens + output_tokens,
                "tpd": input_tokens + output_tokens, "credits": credit, "money_day": cost}

    def capacity(self, route: Route, input_tokens: int, output_tokens: int, now: float | None = None) -> Capacity:
        token_count(input_tokens, "input_tokens")
        token_count(output_tokens, "output_tokens")
        with self.lock:
            return self._capacity(route, input_tokens, output_tokens, time.time() if now is None else now)

    def reserve(self, route: Route, input_tokens: int, output_tokens: int, *,
                profile: str, step: str, session_id: str = "", score: float | None = None,
                baseline_tokens: int = 0, compressed_tokens: int = 0,
                streaming: bool = False, benchmark_id: str | None = None,
                now: float | None = None) -> str | None:
        """Check and reserve every per-key/shared-pool axis in one transaction."""
        token_count(input_tokens, "input_tokens")
        token_count(output_tokens, "output_tokens")
        current = time.time() if now is None else now
        with self.transaction() as db:
            if not self._capacity(route, input_tokens, output_tokens, current).feasible:
                return None
            identity = uuid.uuid4().hex
            session_hash = hashlib.sha256(session_id.encode()).hexdigest()
            db.execute(
                """INSERT INTO routing_decisions
                   (id, route_id, provider_id, model_id, key_id, channel_id, session_hash,
                    profile, step, created_at, status, score, baseline_tokens, compressed_tokens,
                    reserved_input, reserved_output, streaming, benchmark_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'reserved', ?, ?, ?, ?, ?, ?, ?)""",
                (identity, route.route_id, route.provider_id, route.model_id, route.key_id,
                 route.channel_id, session_hash, profile, step, current, score,
                 baseline_tokens, compressed_tokens, input_tokens, output_tokens, int(streaming), benchmark_id),
            )
            scopes = {(route.provider_id, route.model_id, route.key_id, axis) for axis in AXES}
            # Keep pool history even when the operator adds a shared ceiling later.
            scopes.update((route.provider_id, "*", "*", axis) for axis in AXES)
            scopes.update((row["provider_id"], row["model_id"], row["key_id"], row["window_type"]) for row in self._limits(route))
            amounts = self._amounts(route, input_tokens, output_tokens)
            db.executemany("INSERT INTO quota_windows VALUES (?, ?, ?, ?, ?, ?, ?)",
                           [(identity, *scope, current, amounts[scope[3]]) for scope in scopes])
            return identity

    def start(self, identity: str, now: float | None = None) -> None:
        current = time.time() if now is None else nonnegative(now, "now")
        with self.transaction() as db:
            if db.execute("UPDATE routing_decisions SET status='started' WHERE id=? AND status='reserved'", (identity,)).rowcount != 1:
                raise ValueError("reservation is not available")
            db.execute("UPDATE quota_windows SET occurred_at=? WHERE decision_id=?", (current, identity))

    def cancel(self, identity: str) -> None:
        """Release ONLY an attempt which has provably not reached the network."""
        with self.transaction() as db:
            if db.execute("UPDATE routing_decisions SET status='cancelled', completed_at=? WHERE id=? AND status='reserved'", (time.time(), identity)).rowcount != 1:
                raise ValueError("started attempts cannot be cancelled")
            db.execute("UPDATE quota_windows SET amount=0 WHERE decision_id=?", (identity,))

    def finish(self, identity: str, route: Route, *, success: bool, usage: dict | None = None,
               latency_ms: float, ttft_ms: float | None = None, reasoning_ttft_ms: float | None = None,
               quality: bool | None = None, error_class: str | None = None,
               now: float | None = None) -> None:
        """Settle reported usage or keep the entire conservative reservation."""
        current = time.time() if now is None else now
        nonnegative(latency_ms, "latency_ms")
        for name, value in (("ttft_ms", ttft_ms), ("reasoning_ttft_ms", reasoning_ttft_ms)):
            if value is not None:
                nonnegative(value, name)
        with self.transaction() as db:
            row = db.execute("SELECT * FROM routing_decisions WHERE id=?", (identity,)).fetchone()
            if row is None or row["status"] not in {"reserved", "started"} or row["route_id"] != route.route_id:
                raise ValueError("attempt is already settled or belongs to another route")
            estimated = True
            actual_input, actual_output = row["reserved_input"], row["reserved_output"]
            if isinstance(usage, dict) and "prompt_tokens" in usage and "completion_tokens" in usage:
                try:
                    actual_input = token_count(usage["prompt_tokens"], "prompt_tokens")
                    actual_output = token_count(usage["completion_tokens"], "completion_tokens")
                    estimated = False
                except ValueError:
                    actual_input, actual_output = row["reserved_input"], row["reserved_output"]
            amounts = self._amounts(route, actual_input, actual_output)
            for axis, amount in amounts.items():
                # Requests belong to their dispatch window, not their completion window.
                if axis in {"rpm", "rpd"}:
                    db.execute("UPDATE quota_windows SET amount=? WHERE decision_id=? AND window_type=?", (amount, identity, axis))
                else:
                    db.execute("UPDATE quota_windows SET amount=?, occurred_at=? WHERE decision_id=? AND window_type=?", (amount, current, identity, axis))
            first_token = reasoning_ttft_ms if reasoning_ttft_ms is not None else ttft_ms
            generation_ms = latency_ms - (first_token or 0.0)
            tps = None if estimated or not row["streaming"] or generation_ms <= 0 else actual_output * 1000 / generation_ms
            db.execute(
                """UPDATE routing_decisions SET completed_at=?, status=?, input_tokens=?, output_tokens=?,
                   usage_estimated=?, credit_used=?, ttft_ms=?, reasoning_ttft_ms=?, latency_ms=?,
                   tokens_per_second=?, error_class=?, quality=? WHERE id=?""",
                (current, "success" if success else "failure", actual_input, actual_output,
                 int(estimated), amounts["credits"], ttft_ms, reasoning_ttft_ms, latency_ms,
                 tps, error_class, None if quality is None else int(quality), identity),
            )

    def cooldown(self, route: Route, error_class: str, *, retry_after: float | None = None,
                 reset_at: float | None = None, status_code: int | None = None,
                 now: float | None = None) -> None:
        current = time.time() if now is None else now
        defaults = {"rate_limit": 30, "upstream_failure": 15, "timeout": 5,
                    "network_error": 10, "invalid_response": 60, "provider_rejection": 300}
        seconds = defaults.get(error_class, 30)
        if retry_after is not None:
            seconds = max(seconds, nonnegative(retry_after, "retry_after"))
        if reset_at is not None:
            seconds = max(seconds, nonnegative(reset_at, "reset_at") - current)
        # An auth error is key-wide, not a reason to retry another model on it.
        model = "*" if status_code in {401, 403} else route.model_id
        with self.transaction() as db:
            db.execute(
                """INSERT INTO cooldowns VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(provider_id, model_id, key_id) DO UPDATE SET
                   until_at=MAX(until_at, excluded.until_at), error_class=excluded.error_class""",
                (route.provider_id, model, route.key_id, current + seconds, error_class),
            )

    def evidence(self, route: Route, step: str) -> tuple[float, float, float]:
        """Only a client verdict grades correctness.

        A settled attempt with no verdict is neither a success nor a failure:
        the priors stay put until the client reports build/test evidence.
        Operational failures still count as failures on their own.
        """
        with self.lock:
            rows = self.db.execute(
                """SELECT quality, status, latency_ms FROM routing_decisions WHERE route_id=?
                   AND status IN ('success', 'failure') AND step=? ORDER BY completed_at DESC LIMIT 200""",
                (route.route_id, step),
            ).fetchall()
            if not rows:
                rows = self.db.execute(
                    """SELECT quality, status, latency_ms FROM routing_decisions WHERE route_id=?
                       AND status IN ('success', 'failure') ORDER BY completed_at DESC LIMIT 200""",
                    (route.route_id,),
                ).fetchall()
        successes = sum(row["quality"] == 1 and row["status"] == "success" for row in rows)
        failures = sum(row["quality"] == 0 or row["status"] == "failure" for row in rows)
        latencies = [row["latency_ms"] for row in rows if row["latency_ms"] is not None]
        latency = sum(latencies) / len(latencies) if latencies else route.latency_ms
        return route.prior_success + successes, route.prior_failure + failures, latency

    def grade(self, identity: str, passed: bool) -> None:
        if not isinstance(passed, bool):
            raise ValueError("grade must be a boolean")
        with self.transaction() as db:
            if db.execute("UPDATE routing_decisions SET quality=? WHERE id=? AND status IN ('success', 'failure')", (int(passed), identity)).rowcount != 1:
                raise ValueError("attempt is not complete")

    def grade_attempt(self, identity: str, grade: bool, verdict: str) -> None:
        """Store one client verdict for a completed attempt.

        Unknown is not success: only verdict `pass` may carry grade=true, so a
        harness without an oracle cannot promote `unknown` to a win. Verdicts
        are a closed enum and an attempt is graded at most once.
        """
        if not isinstance(grade, bool):
            raise ValueError("grade must be a boolean")
        if not isinstance(verdict, str) or verdict not in VERDICTS:
            raise ValueError(f"unsupported verdict: {verdict}")
        if grade and verdict != "pass":
            raise ValueError("grade must be false unless verdict is 'pass'")
        if not grade and verdict == "pass":
            raise ValueError("verdict 'pass' requires grade true")
        with self.transaction() as db:
            row = db.execute(
                "SELECT status, quality, verdict FROM routing_decisions WHERE id=?",
                (identity,),
            ).fetchone()
            if row is None:
                raise ValueError("unknown attempt_id")
            if row["verdict"] is not None or row["quality"] is not None:
                raise ValueError("attempt already graded")
            if row["status"] != "success":
                raise ValueError("attempt is not complete")
            # `unknown` closes the attempt without moving either prior: no
            # oracle is evidence of neither correctness nor failure. Storing
            # it as quality=0 would penalise a route for a lack of evidence.
            quality = None if verdict == "unknown" else int(grade)
            db.execute(
                "UPDATE routing_decisions SET quality=?, verdict=? WHERE id=?",
                (quality, verdict, identity),
            )

    def decision(self, identity: str) -> dict:
        with self.lock:
            row = self.db.execute("SELECT * FROM routing_decisions WHERE id=?", (identity,)).fetchone()
            if row is None:
                raise KeyError(identity)
            return dict(row)

    def diagnostics(self) -> dict:
        with self.lock:
            counts = dict(self.db.execute("SELECT status, COUNT(*) FROM routing_decisions GROUP BY status"))
            classes: dict[str, int] = {}
            for route in self.routes():
                classes[route.free_class] = classes.get(route.free_class, 0) + 1
            return {"attempts": counts, "catalog_models": self.db.execute("SELECT COUNT(*) FROM models WHERE active=1").fetchone()[0],
                    "active_cooldowns": self.db.execute("SELECT COUNT(*) FROM cooldowns WHERE until_at>?", (time.time(),)).fetchone()[0],
                    "free_classes": dict(sorted(classes.items()))}

    def tps_stats(self, days: float = 7, min_samples: int = 5) -> list[dict]:
        """Aggregate successful stream TPS by provider/model from local evidence.

        The query returns prompt-free timing samples only. Non-streaming and
        estimated-usage attempts are excluded because they have no measured TPS.
        """
        if (isinstance(days, bool) or not isinstance(days, (int, float))
                or not math.isfinite(days) or not 1 <= days <= 365):
            raise ValueError("days must be a finite number between 1 and 365")
        sample_floor = token_count(min_samples, "min_samples")
        if not 1 <= sample_floor <= 10000:
            raise ValueError("min_samples must be between 1 and 10000")
        cutoff = time.time() - days * 86400
        with self.lock:
            rows = self.db.execute(
                """SELECT provider_id, model_id, tokens_per_second, ttft_ms
                   FROM routing_decisions
                   WHERE status='success' AND streaming=1
                     AND tokens_per_second IS NOT NULL AND tokens_per_second > 0
                     AND completed_at >= ?
                   ORDER BY provider_id, model_id""",
                (cutoff,),
            ).fetchall()
        groups: dict[tuple[str, str], dict[str, list[float]]] = {}
        for row in rows:
            tps = row["tokens_per_second"]
            if not isinstance(tps, (int, float)) or not math.isfinite(tps) or tps <= 0:
                continue
            key = (row["provider_id"], row["model_id"])
            group = groups.setdefault(key, {"tps": [], "ttft": []})
            group["tps"].append(float(tps))
            ttft = row["ttft_ms"]
            if isinstance(ttft, (int, float)) and math.isfinite(ttft) and ttft >= 0:
                group["ttft"].append(float(ttft))
        updated = datetime.now(timezone.utc).isoformat()
        stats = []
        for (provider, model), group in sorted(groups.items()):
            if len(group["tps"]) < sample_floor:
                continue
            stats.append({
                "provider_name": provider,
                "model": model,
                "median_tps": round(_percentile(group["tps"], 0.5), 1),
                "avg_tps": round(sum(group["tps"]) / len(group["tps"]), 1),
                "p95_tps": round(_percentile(group["tps"], 0.95), 1),
                "avg_ttft_ms": (round(sum(group["ttft"]) / len(group["ttft"]), 1)
                                if group["ttft"] else None),
                "sample_count": len(group["tps"]),
                "last_updated": updated,
            })
        return stats

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int,
                      cache_read_tokens: int = 0,
                      cache_creation_tokens: int = 0) -> dict | None:
        """Return the cheapest known ledger price for a model or alias.

        Prices are stored per million tokens; the response uses per-token
        pricing to match the legacy cost-estimate contract. This never selects
        or dispatches a route, and unknown pricing stays unknown.
        """
        if not isinstance(model, str) or not model or len(model) > 512:
            raise ValueError("model must be a non-empty string")
        prompt = token_count(prompt_tokens, "prompt_tokens")
        completion = token_count(completion_tokens, "completion_tokens")
        cached = token_count(cache_read_tokens, "cache_read_tokens")
        cache_creation = token_count(cache_creation_tokens, "cache_creation_tokens")
        routes = self.routes()
        matching = [route for route in routes
                    if route.alias == model or route.model_id == model]
        if not matching:
            matching = [route for route in routes if model in route.profiles]
        priced = []
        for route in matching:
            if route.input_price is None or route.output_price is None:
                continue
            cache_price = (route.input_price if route.cached_input_price is None
                           else route.cached_input_price)
            used_cache = min(prompt, cached)
            cost = ((prompt - used_cache) * route.input_price
                    + used_cache * cache_price
                    + cache_creation * route.input_price
                    + completion * route.output_price) / 1_000_000
            priced.append((cost, route))
        if not priced:
            return None
        cost, route = min(priced, key=lambda item: item[0])
        pricing = {"input_cost_per_token": route.input_price / 1_000_000,
                   "output_cost_per_token": route.output_price / 1_000_000}
        if route.cached_input_price is not None:
            pricing["cache_read_input_token_cost"] = route.cached_input_price / 1_000_000
        free = route.free_class != "paid" and route.free_class != "unknown"
        return {"model": model, "cost": cost, "currency": "USD",
                "pricing": pricing, "source": "free_tier" if free and cost == 0 else "sidecar_ledger",
                "free_class": route.free_class, "free": free}

    def close(self) -> None:
        with self.lock:
            self.db.close()
