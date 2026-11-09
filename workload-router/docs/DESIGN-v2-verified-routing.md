# Design v2 — Routing for *verified-correct* results at zero marginal cost

Status: proposal, not implemented. Supersedes the scoring and quota sections of
`docs/specs/p0-router.md` and `docs/free-tier-classes.md` where they conflict.
Written 2026-11-09 against branch `feat/newapi-sidecar`.

---

## 0. How to use this document (for the reviewing harness)

This is a **delta document**, not a greenfield plan. The repo already implements
a working P0/P1 router. Read this in this order:

1. `STATE.md` and `../HANDOFF-CURRENT-newapi-sidecar.md` for what actually runs.
2. This document, sections 2–7 (the correctness argument), then 8–9 (placement
   and build-vs-buy), then 13 (roadmap).
3. Section 12 lists what to **stop** building.

Rules for whoever implements this:

- Section 14 lists decisions that are **not yours to make**; ask the owner.
- Section 12 says which sibling repos are frozen. Do not "improve" them.
- Do not add a dependency that needs more than ~64 MB RSS on the target host.
- Do not add an LLM call to the routing path. Ever. Section 5 explains why.
- Every change must keep `python3 -m pytest tests -o addopts= --quiet` green and
  provider-free.

If you disagree with a section, write the counter-argument into
`docs/DESIGN-v2-review-notes.md` rather than silently re-architecting.

---

## 1. What exists today, and the three lies it currently tells

Current live-ish path (`newapi_server.py`): accept OpenAI-shaped request → build
`Work` → `route_policy.rank_routes()` → `quota_ledger` capacity check → dispatch
through `provider_adapter` → best-effort failover → record outcome.

That is a good skeleton. The problem is that its three most important inputs are
currently optimistic:

### Lie 1: HTTP 200 means success

`router_server.py` calls `state.record_outcome(..., "success")` when the provider
stream ends cleanly (lines ~308, ~339, ~368). `quota_ledger.evidence()` then
treats those rows as successes with `quality != 0`.

A model that returns fluent, confidently wrong code gets a clean 200. So the
bandit learns "this model is reliable" when it has learned nothing at all. In
practice the router ends up optimising for *providers that don't 429*, which it
already knew, while the actual signal — did the patch compile and pass tests —
never enters the loop.

### Lie 2: `pressure` is a cost

`quota_ledger._capacity()` returns `pressure = projected/ceiling` for the current
window and `route_policy.score_route()` charges
`quota_lambda * capacity.pressure`. That is a *utilisation* term: it makes a
provider look expensive exactly when you are using it well. It is maximised at
the moment you are about to exhaust a tier you were going to exhaust anyway, and
it is zero in the one case that matters — quota you will never spend.

The correct quantity is a **shadow price**: the utility of the task you give up
because that token is gone. Section 6 derives it, and shows it is zero whenever
remaining demand exceeds remaining quota.

### Lie 3: latency is a mean

`evidence()` averages the last 200 `latency_ms` values, then `score_route()`
multiplies a single `coding_time_value` by that mean. Free providers are
load-dependent and bimodal: p50 400 ms, p95 30 s, p99 = never. A mean of 1.2 s
describes a provider that is unusable at 6 pm and excellent at 4 am, and the
router cannot tell those apart. Section 7 fixes the estimator.

Everything else in the existing design — hard capability gates, deterministic
classification, session stickiness, bounded failover, SQLite state, free-only
enforcement by construction — is sound and should be kept.

---

## 2. The objective, stated correctly

The owner's goal is not "lowest cost per token". It is:

> minimise wall-clock time to a *correct* result, and minimise consumed resource,
> subject to never spending real money.

Both terms must be measured **to completion**, including retries and
verification. Define, for request `i` and route `m`:

- `s(i,m)` — probability that one attempt produces output that passes
  verification (not "returns 200")
- `T(i,m)` — elapsed time of one attempt, including queueing, prefill, decode,
  and the verification run
- `C(i,m)` — shadow-priced resource cost of one attempt

Then the expected cost of *finishing* is `C/s` and the expected time is `T/s`
(geometric). Rank by

```
score(i,m) = ( w_t · T(i,m) + w_c · C(i,m) ) / s(i,m)
```

which is what `score_route()` already computes, **provided** `s`, `T`, `C` mean
these things. Today they do not. Fixing the three inputs above is most of the
work in this document; the formula itself is fine.

Two consequences worth stating because they invert intuition:

- **A free model that needs two retries is more expensive than a paid one that
  doesn't.** Since money is pinned to zero here, this shows up purely as time.
  Ranking by quality-per-attempt is not a nicety, it is the whole objective.
- **Verification time counts.** If your test suite takes 90 s, a model that is
  30 s faster per attempt wins only if it does not add attempts. Oracles must be
  fast or sampled (section 3.3).

---

## 3. What "correct" can actually mean

There is no general oracle for code. There are only observables that are sound
for a subset of errors. Rank them by soundness and price:

| Tier | Oracle | Detects | Cost | Gamerable? |
|---|---|---|---|---|
| V0 | parse / compile / `tsc --noEmit` / `go build` / `cargo check` | syntax, types, imports, arity | 0.2–5 s, free | barely |
| V1 | linters, formatters-in-check-mode, schema validation | style, undefined names, bad JSON | 0.5–5 s, free | barely |
| V2 | **existing** test suite, changed-files subset | the behaviour the repo already promises | 2 s–2 min, free | **yes** (see 3.2) |
| V3 | behavioural: run the built artifact, golden files, exit codes, CLI contract | end-to-end behaviour | seconds–minutes | harder |
| V4 | diff invariants: file allowlist, no new deps, no deleted/skipped tests, no secrets, size caps | scope violations, cheating | <50 ms, free | no |
| V5 | independent grader model from a *different family* | "looks correct", spec drift | tokens, slow, unreliable | n/a |

### 3.1 The oracle is the repo's own CI, not a new test suite

Do **not** have the LLM write the tests that decide whether it passed. That is
self-consistency, not correctness: the model encodes the same misunderstanding
in both the patch and the test, and you pay tokens for the privilege. It also
doubles the failure surface.

The oracle should be whatever the repository already runs: its `make check`,
`npm test`, `pytest`, `cargo test`, CI workflow. Those tests are (a) free,
(b) independent of the model, (c) written by the project owner, and (d) already
the definition of "correct" the owner cares about. V0 and V4 are deterministic
and catch a large fraction of bad patches before you spend seconds on V2.

LLM-written tests are still useful — as a *proposal* to the human, or as a weak
V5 sanity check — but they must never be the gate that grades a route.

### 3.2 The test-tampering problem (important, and currently unhandled)

The cheapest way for a model to make tests pass is to edit the tests. Any
cascade whose reward is "tests green" will discover this within a week.

Mitigations, in order of preference:

1. Run verification against a **pristine worktree**: `git worktree add` at the
   base commit, apply only the model's non-test diff. Test files, fixtures,
   `conftest.py`, CI config and lockfiles come from HEAD.
2. Reject patches that touch test paths or `.github/workflows/**` unless the
   request explicitly asked for test changes; log it as `semantic_failure` with
   `error_class=test_tamper`.
3. V4 diff invariants: no `skip`/`xfail` additions, no assertion deletions, no
   changes to snapshots.
4. If a task legitimately needs new tests, that is a *different* task class with
   its own (human-approved) acceptance step — do not fold it into the cascade.

### 3.3 Sampling the oracle

Verification has a cost in wall-clock that is charged to every attempt. Use it
adaptively:

- Always run V0 + V4 (sub-second, catches the majority).
- Run V2 on the changed-files subset by default; full suite only when the diff
  touches shared modules or when the previous attempt already failed V2.
- Cache by `hash(diff ∪ base_commit)` so a re-run of the same patch is free.
- Never run the oracle twice for the same attempt id.

### 3.4 Who runs it

**The client, not the router.** The router has no checkout, no toolchain, and
1 GB of RAM. The harness (OpenCode et al.) already has all three, and already
runs `bash` tool calls. So:

- The harness (or a plugin inside it) runs V0–V4 and knows the result.
- It reports the result back with the attempt id issued by the router.
- The router only stores the grade.

This is the single most important architectural consequence in this document:
**the oracle lives on the client, so the grade must flow client → router.** A
router deployed purely on the VPS with no client cooperation can only ever learn
HTTP success, which is Lie 1. If the grade channel is not built, the router is
a load balancer with extra steps.

Required protocol change (small):

```
response header:  X-Router-Attempt-Id: <uuid>
POST /v1/router/outcome
  { "attempt_id": "...", "grade": true|false,
    "verdict": "pass"|"fail_build"|"fail_lint"|"fail_test"|"fail_scope"|"timeout"|"unknown",
    "oracle": "V2", "duration_ms": 4120 }
```

`grade` maps onto the existing `quota_ledger.grade()` boolean; `verdict` is a
bounded enum that must be closed (no free text, no raw stderr — that is how
prompt content leaks into telemetry, which `STATE.md` already correctly
forbids).

---

## 4. Cascade, and why it beats classification

### 4.1 The mechanism

Order the feasible routes into a ladder by *ascending expected total cost*,
then:

```
attempt(k) → verify → if pass: done, grade(attempt, true)
                     → if fail: escalate to k+1 with the failure evidence
                                  (compiler message / failing test name only)
                     → if cap exceeded: stop and report honestly
```

Escalation carries the error output forward, which is the cheapest possible
context: 200–2000 tokens of structured failure, not a re-explanation of the
task. That is why escalation is nearly always cheaper than restarting at a
higher tier from scratch, and why a cascade on top of *any* router beats a
better router without one.

Expected values for a ladder of attempts `1..k` with per-tier `(s, T, C)`:

```
E[T] = Σ_j  T_j · Π_{i<j} (1 − s_i)
E[C] = Σ_j  C_j · Π_{i<j} (1 − s_i)
```

You do not need to solve this. With ≤4 tiers, enumerate the ladder in the
policy module (a 20-line DP) and pick the entry tier minimising
`w_t·E[T] + w_c·E[C]`.

### 4.2 When to start high instead of cascading

Cascading is not always right. Start at the top tier when:

- **Context is huge.** Prefill dominates: re-sending a 120 k-token transcript
  twice costs more than the quality delta. Rule of thumb: above ~40 k input
  tokens, prefill time exceeds the expected savings of a cheap first attempt.
  `score_route()` already has `prefill_ms_per_token` and `handoff_ms`; use them
  to gate the ladder rather than a hardcoded number.
- **The failure is expensive and irreversible** (migrations, deploys, deletions,
  anything with a side effect that a retry cannot undo). One shot, top tier.
- **The cheap tier is known-bad for the class** (`s` below the exploration
  floor for that `(model, step)` cell). Don't explore what you have already
  disproved ten times.

Conversely, start at the bottom when the request is small, bounded, and
verifiable: single-file edits, test fixes, rename/refactor, dependency bumps,
error-message-driven debugging. That is the majority of agentic coding traffic.

### 4.3 Mapping onto existing code

- `Work.step` already carries the phase vocabulary. Add `attempt` (1..N) and
  `prior_verdict` to `Work`.
- `route_policy.rank_routes()` gains a `ladder_entry(work, routes, ledger)`
  helper returning the ordered tier list; keep `rank_routes()` as the ranking
  primitive so existing tests keep meaning something.
- `newapi_runtime` already accepts a `grader` callback and calls
  `ledger.grade(identity, passed)`. The missing piece is not the plumbing, it is
  (a) issuing an attempt id to the client, (b) accepting the grade back, and
  (c) not pre-stamping `success` at HTTP 200.
- `failover` currently fires on transport errors. Keep that: an operational
  failure should retry the *same tier* on a different deployment, once, before
  escalating a tier. Semantic failure is different — it must escalate, not
  retry, or you will loop forever on a model that is confidently wrong.

### 4.4 What cascade costs you

- Latency variance: p95 time-to-green gets worse than "one good model" because
  failures are paid in full. Track p95 T2G, not just mean.
- Credit burn: on `free_credits` routes a failed attempt still spends. That is
  a real cost and must be in `C`.
- Harness interaction: if the harness also retries, you get 2×N attempts.
  Disable harness-level retry (or set it to 1) when the router owns failover.

---

## 5. Classification: fast, free, and almost never learned

### 5.1 The latency claim, quantified

"Instant" means the classification is noise next to everything around it. Real
numbers for the three candidate implementations:

| Implementation | Cost per request | Where it runs |
|---|---|---|
| Regex + char-count + header read (today's `estimate_context_tokens`) | 20–80 µs | anywhere |
| 40-feature logistic regression as a pure function | < 10 µs | anywhere |
| 100-tree GBDT exported to nested `if`/arrays, depth ≤ 6 | 50–200 µs | anywhere |
| Any network call, even to localhost | 0.3–2 ms | sidecar |
| Any HTTP call to the VPS | 20–120 ms RTT | bad |
| Any LLM-based classifier | 300–3000 ms **and** quota | never |

Meanwhile the harness spends 200–2000 ms assembling context (reading files,
splicing the system prompt, serialising tools) before the provider sees the
request. So a 100 µs classifier is ~0.01% of the request. That is what
"instant" means here: **the classification must be a pure function evaluated in
the same process, with no I/O, no allocation of note, and no model.** Not "fast
LLM", not "fast API call".

Practical rule: budget **1 ms p99** for the decision, and add a test that
asserts it (`test_router_overhead.py`), because it will otherwise regress
silently the first time someone adds a "quick" SQLite query or a config reload
to the hot path.

### 5.2 Features worth having (all cheap)

From the request itself, with no tokenizer:

- input token **bound** (byte length; `route_policy.input_token_bound` already
  does this correctly and honestly)
- `tools` present, `structured_output` requested, `stream` requested
- message count, tool-result count, presence of a system/developer prompt
- last-user-message length and shape: code fence, stack trace regex
  (`Traceback`, `error TS\d+`, `panicked at`, `at .*\(.*:\d+:\d+\)`), diff
  (`^--- a/`), question mark density, imperative verb hit
- language/framework hints from fence tags or file extensions in context
- `repo_size_class` if the client sends it (it knows)

From the harness, via headers (the contract already exists in
`integrations/opencode_headers.py`):

- `X-Router-Phase` (explore/plan/implement/verify/debug/deep_debug) — strongest
  single signal, and free
- `X-Router-Action` (edit / run-tests / read / search)
- `X-Router-Verification: failed` (already there)
- new: `X-Router-Attempt`, `X-Router-Budget-Ms`, `X-Session-Id` (exists)

### 5.3 When to learn, and what to learn

Do **not** ship a learned classifier first. Ship the deterministic rules, then
learn only when there is data:

- Gate: ≥ 500 graded outcomes per `(step, class)` cell, or ≥ 2000 overall.
- Model: logistic regression on the features above (interpretable, 40 floats,
  trivially portable) — *not* a tree ensemble first, because with ~10k rows a
  tree will memorise `session_id`-shaped noise.
- Training: offline, on the laptop, on a weekly snapshot of the `outcomes`
  table. Export to JSON: `{"features": [...], "weights": [...], "bias": ...}`.
- Inference: `sum(w_i * x_i) + b > 0`, in the hot path, with a fallback to rules
  if the JSON is missing or its version does not match.
- Serve it as a **tie-breaker**, never as the primary gate: rules decide
  feasibility and hard filters; the model only nudges the entry tier.

Why not a tree: no inference server exists that is worth the memory, and a
portable JSON tree is a maintenance liability for a marginal accuracy gain.
Logistic regression gets you most of the signal at zero operational cost.

### 5.4 The honest limit

You cannot classify "is this task hard" from the prompt alone with high accuracy,
and you should not try. Difficulty reveals itself in the *failure*, which is
precisely what the cascade uses. Classification's real job is narrower and
easier: **feasibility (does the context fit, are tools supported, is the
provider up) and entry point (cheap vs expensive start)**. Framed that way, it
can be fast, free and mostly rule-based, which is what the current code already
does. Keep it.

---

## 6. Quota economics: a shadow price that doesn't lie

### 6.1 Reframing

The owner's intuition is: "if my daily allowance expires at midnight, unused
tokens are wasted, so raise that provider's priority as midnight approaches."

That is half right, and the missing half matters. The correct statement:

> The value of a token of quota is the utility of the task you will *not* be
> able to run because you spent it. If you will run out of quota before the
> reset regardless, its value is high. If you will not run out, its value is
> **zero** — wasting it costs nothing, because there was nothing to displace.

Formally, with remaining quota `Q` (tokens) and forecast demand `D` over the
remaining part of the window:

```
at_risk   = max(0, Q − D)
λ(t)      = ∂U/∂Q  ≈ 0                    if Q ≥ D (with buffer)
                   ≈  utility density of the marginal displaced task  otherwise
```

Note what this does at 23:50: `D` shrinks because there is little time left, so
`Q ≥ D` becomes *more* likely, not less, and `λ` correctly falls to zero. The
"panic-routing at the deadline" behaviour that the intuition suggests is exactly
what this formula forbids — and it is right to forbid it. You cannot spend
800 k tokens in ten minutes, and routing them into a bad fit makes the result
worse for no gain.

The only case where `λ` rises near the reset is when you have *plenty of demand*
and not enough quota (busy day, big refactor, long agent runs). Then the marginal
task is real, and the price should be paid.

### 6.2 Implementing it: a periodic shadow auction

You do not need an LP solver. Once every 5 minutes, for each provider/model:

1. Forecast `D` for the remaining window from the last 4 weeks of
   `routing_decisions` at the same weekday/hour (a simple seasonal mean is
   enough; median is better than mean because free tiers are bursty).
2. Compute `Q` from the ledger.
3. If `Q ≥ 1.15·D`, set `λ = 0` and stop.
4. Otherwise, walk tasks in the forecast in descending utility density
   (`utility / tokens`) and find the token at which cumulative demand crosses
   `Q`. `λ` = the utility density at the crossing.
5. Write `λ` per route into a `shadow_prices` table with a TTL; `score_route()`
   reads it instead of `quota_lambda * pressure`.

Cost: a few dozen routes × a few hundred forecast buckets, every 5 minutes, on
1/8 of a CPU. Tens of milliseconds. Completely feasible.

Utility density needs a number. Use the proxy already implicit in the config:
`w_c / (time_value)` is the owner's exchange rate between seconds and money;
utility density in *seconds of expected work displaced* per 1000 tokens is
stable and interpretable. Express `λ` in "ms of expected completion time per
1k tokens" and the units cancel in the score.

### 6.3 One-time credits: not a cost, a depreciating option

The owner's dilemma — prioritise one-time credits less (they don't renew) or
more (the provider may vanish)? — dissolves once you model them correctly.

A one-time credit is **not** cheaper tokens. It is an option on a *quality
advantage* that decays. Define:

- `a(m, class)` = expected quality of `m` minus the best *available* free
  alternative for that task class, measured in verified-success rate (or in
  expected seconds saved). If `a ≤ 0`, the credit is worth **nothing**, no
  matter how large the balance.
- `μ` = weekly hazard that the balance is lost (provider dies, grant expires,
  account closed). Estimate from observed provider lifetimes; use a curated
  prior per provider. AgentRouter-style random re-grants *lower* `μ` — model
  them as `μ_eff = μ · (1 − ρ)` where `ρ` is the renewal probability.

Then the rule is:

```
use now if  a · s(now)  >  e^(−μ·h) · E[a] · s(future task)
reserve otherwise
```

with `h` the expected holding time until a task arrives where `a` is large.
Practical form, which is all you need to implement:

- `a ≤ 0` → priority floor; never route there for its own sake. Its only use is
  as an **exploration** source (you learn something) or as **latency tier** for
  a fast first attempt if it is genuinely faster.
- `a > 0`, `μ` high (fly-by-night, small grant) → spend within days, on the
  first task in the class where it wins. Do not hoard.
- `a > 0`, `μ` low (big established provider, large grant) → keep a reserve for
  `deep_debug` / `implement` on large contexts, where the quality delta is
  largest and the alternative is worst.
- **Tiny balances** (less than one useful task: ~50 k tokens) → drain
  immediately into whatever class it wins, because the option value is smaller
  than the overhead of tracking it.

Key reframing for the "wasted credit" worry: **the loss happened when you
acquired the credit, not when it expires.** Expiry is just the accounting
date. Spending a credit on a task where it is worse than free is a second loss,
not a recovery. So "use it or lose it" is only valid when `a > 0`.

### 6.4 Decision table

| Class | Quota mechanism | `C` in the score | Priority behaviour |
|---|---|---|---|
| `free_unlimited` | none (rate caps only) | 0 money, `λ_rmp` for RPM/TPM caps | baseline floor; never deprioritised below `unknown` |
| `free_limited` | daily/weekly window, no carryover | 0 money + `λ(t)` from 6.2 | rises only when `Q < D`; automatically falls near reset |
| `free_credits` | one-time or finite balance | notional credit tariff + `λ_option(t)` from 6.3 | gated by `a(m,class) > 0`; hazard-discounted |
| `paid` | real money | real price | excluded (free-only enforced) |
| `unknown` | unknown | ∞ | hard-infeasible; never silently free |

One extra axis that is currently missing and cheap: **rate-limit admission
control**. A route with 4 RPM remaining is not "available at pressure 0.8", it
is available *four times per minute*, and a cascade will burn those four in ten
seconds. `λ` must be per-axis (`rpm`, `tpm`, `rpd`, `tpd`) and the cascade should
reserve attempts against the axis it will need, or it will strand itself mid-ladder.

---

## 7. Free-provider latency: censored, seasonal, and bursty

Three corrections to `evidence()`:

1. **Censor nothing.** A request that times out at 30 s is a 30 s failure, not
   an absent sample. Today `latency` averages only rows that have a value, which
   quietly deletes the worst outcomes and makes slow providers look good. Store
   `timeout` with `latency_ms = timeout_budget` and `quality = 0`.
2. **Seasonality.** Bucket by hour-of-week (168 buckets) and keep an EWMA
   (`α ≈ 0.2`) per `(route, step, bucket)`. Score with `p90` for time-sensitive
   classes and `p50` for batch. Fall back to the global EWMA until a bucket has
   ≥5 samples, so a cold bucket cannot look artificially fast.
3. **Bimodality.** Keep both `p50` and `p90`. If `p90/p50 > 5`, the provider is
   bursty: usable as a background attempt, bad as the only path. This is exactly
   the MiniMax-vs-DeepSeek/Kimi situation the owner describes — the "better"
   models are not slower on average, they are *unreliable*, and a mean hides it.

### 7.1 Hedging, but only where it is free

For `free_unlimited` routes, speculative hedging is free in money: fire the
fast baseline, and if no first token arrives by `TTFT_budget` (adaptive, start
at 4 s), start the same turn on a second unlimited provider; first useful token
wins, cancel the loser. This converts a 30 s stall into a 4.5 s worst case.

Do **not** hedge on `free_limited` or `free_credits` routes: a cancelled request
still consumes quota at most providers, so hedging there doubles the burn. This
distinction is worth an explicit field, because it is the kind of thing that
gets copy-pasted into the wrong branch.

### 7.2 The always-available baseline

Keep one route pinned as the floor: a model that is not the smartest but is
almost always reachable (the owner's current answer is MiniMax 3.1 Flash on
NIM). It is what the cascade falls back to when everything better is stalling,
and it is what makes "route to the best available" meaningful instead of
"route to the best on paper and wait 45 s". Give it a dedicated role in config
(`baseline: true`) rather than expressing it as a weight.

---

## 8. Placement: where each piece runs

### 8.1 The three options, measured

| Placement | Decision latency | Works for weak clients | Shared quota state | Memory on client | Harness coupling |
|---|---|---|---|---|---|
| In-harness plugin (OpenCode plugin, in-process) | ~0 (µs) | no | via sync | ~5 MB | tight |
| Client sidecar (`127.0.0.1:8787`, OpenAI-compatible) | 0.2–1 ms | mostly | via sync | 25–60 MB | none (baseURL) |
| VPS proxy (current sidecar) | 20–120 ms RTT | **yes** | native | 0 | none |

The VPS option's latency is not the problem — 60 ms against a 2 s LLM call is
3%. The problems are that (a) it sees no verification result, so it can only
learn Lie 1, and (b) at 1/8 OCPU it must stream every byte of every response,
which is CPU-bound work on the critical path (`io.Copy`-style passthrough helps;
JSON re-serialisation does not).

### 8.2 Recommended split

- **Policy** — pure functions, no I/O: `route_policy.py`, `router_core.py`.
  Runs **anywhere**, including in-process in a plugin.
- **State** — SQLite: quota ledger, outcomes, shadow prices, health. Lives on
  **one** VPS, single writer.
- **Clients** — laptop/phone get a **read-only snapshot** (a few KB: λ, health,
  `p50/p90`, cooldowns, balances) refreshed every 30–60 s, apply the policy
  locally, then post `(attempt, tokens, grade)` asynchronously.

This is safe because quota is a *soft* budget: overshooting by the few seconds
of replication delay does not violate anything, and the hard limit is enforced
by the provider's own 429, which the client already handles. Eventual
consistency is the right consistency model here, and it is what makes the 1 GB
constraint disappear.

### 8.3 Budgets to hold

Measured on the target (E2.1.Micro, 1/8 OCPU, 1 GB, zram):

- RSS: policy + HTTP + SQLite ≤ 60 MB; zram covers the rest.
- Router-added p99 latency: ≤ 5 ms in-process, ≤ 3 ms sidecar, ≤ 5 ms VPS
  (decision only; network RTT excluded).
- No `asyncio` + threads + subprocess all in one request path; pick one.
- Streaming: byte passthrough. Do not parse SSE, re-serialise JSON, or buffer
  the whole response to compute token counts — that is the one thing that will
  actually hurt at 1/8 core.
- Do not run a tokenizer. `input_token_bound()` is deliberately a byte bound;
  keep it, and use provider-reported usage for actuals.

### 8.4 Weak client

Remote mode *is* the weak-client answer: point the harness's `baseURL` at the
VPS, ship a snapshot-free request, let the router decide and dispatch. The cost
is that verification information is lost unless the client posts back. So:
weaker client = degraded learning, not degraded routing. State that trade-off
in the config (`state_mode: remote`, `grade_channel: best_effort`).

---

## 9. Buy vs build: is there a free OpenRouter with your own keys?

Short answer: there are several free **gateways** with BYOK, and none of them
routers by *verified quality*. They solve transport, retries, fallbacks,
caching, observability and spend caps. They do not solve the objective in
section 2, and they cannot, because they never see your tests.

| Option | BYOK free | Routing intelligence | Runs in ~1 GB | Verdict |
|---|---|---|---|---|
| **new-api / one-api** (self-host) | yes | channels, weights, priority, retries, model mapping | yes (Go, small) | keep as the *transport* + key vault; do not expect quality routing |
| **LiteLLM proxy** (self-host) | yes | latency/cost/usage-based routing, fallbacks, cooldowns, budgets | ~200–400 MB — tight but works with 1 worker | good reference implementation; Python cost |
| **Bifrost** (self-host, Go) | yes | fallbacks, load balancing, semantic cache, MCP | yes, small | strongest "free plumbing" alternative to new-api |
| **Portkey Gateway** (OSS) | yes | fallbacks, retries, conditional routing, caching | yes | good, but the hosted free tier is metered |
| **Cloudflare AI Gateway** (free tier) | yes | retries, fallbacks, caching, logging | n/a | best zero-maintenance plumbing; no quality routing |
| **Helicone** | yes | caching, observability, rate limits | n/a | observability, not routing |
| **Envoy AI Gateway / Higress / Kong AI** | yes | L7 traffic shaping, token-aware limits | heavy | overkill for one user |
| **OpenRouter itself** | BYOK, with a platform fee | `auto` = quality prior + price/latency sort; `:free` variants are rate-limited | n/a | useful as *one channel*, not as your router; it cannot use your own keys for its auto logic |

Conclusion: self-host **one** gateway for transport (new-api, already in
progress) and keep the policy in your own code. That is also what OpenRouter
does internally — their "auto" is a scoring function plus fallbacks plus
provider metadata, not magic. Reimplementing the scoring is ~300 lines. The
part you cannot buy is the verification loop, which is the part worth building.

---

## 10. Harness integration

Primary target: **OpenCode** (owner's daily driver). Requirements the router
needs from a harness, in priority order:

1. **Per-request model override** — so the router's decision is respected
   without restarting or reconfiguring the session. OpenCode plugins can rewrite
   `chat.params` per call.
2. **A baseURL override** to an OpenAI-compatible endpoint (universal fallback
   for pi/omp/jcode and anything else).
3. **Tool-result visibility** — a hook on `bash`/`edit` results so the client
   can grade attempts (V0–V4) without the user doing anything.
4. **Session id** — already supported (`X-Session-Id`) and needed for stickiness.
5. **Retry control** — the harness's own retry must be set to 0 or 1, else the
   amplification in 4.4 applies.

If a harness offers (1) and (3), it is a first-class target; if only (2), it
still works but grading is manual; if neither, it cannot participate.

Keep the portable contract `integrations/opencode_headers.py` and treat
harness-specific code as thin adapters. Do not fork the router per harness.

---

## 11. Evaluation: how we will know it works

Instrument five numbers and publish them weekly:

1. **T2G p50/p90** — request → first verified-correct result. The headline.
2. **Shadow-cost per completed task** — `Σ λ·tokens / verified tasks`.
3. **Waste ratio** — expired quota ÷ granted quota, per provider per week.
   Target < 15% for `free_limited`; for `free_credits`, waste is acceptable
   *only* when `a ≤ 0` (in which case routing there would have cost quality).
4. **Escalation rate + ladder depth distribution** — if the entry tier is right,
   most tasks finish at tier 1.
5. **Router overhead p99 + RSS** — regression guards for section 8.3.

Method:

- **Shadow mode first** (already exists: `workload_router_shadow.py`). Log
  decisions without acting; compare "what the router would have picked" against
  what actually ran. Free, zero risk.
- **Replay** existing traces (`test_trace_replay.py`, `examples/`) to check the
  scorer's monotonicity and the ladder DP against hand-computed cases.
- **A/B with epsilon exploration** on real traffic (`top_n=3` Thompson already
  gives this). Do not attempt offline counterfactual evaluation of unobserved
  routes — you have no labels for them, and it will produce confident nonsense.
  The honest options are exploration or nothing.
- **Failure taxonomy**, not a single bucket: `fail_build`, `fail_lint`,
  `fail_test`, `fail_scope`, `fail_tamper`, `timeout`, `wrong_but_green`
  (caught by human review). The last one is the one that matters and the one
  no automated metric will show you; sample 10 graded passes a week by hand.

---

## 12. Repo consolidation (do this, it is blocking)

There are five overlapping projects. That is why the last attempt stalled: every
session re-decided the architecture.

| Repo | State | Decision |
|---|---|---|
| `LLM-API-Key-Proxy` / `workload-router` | most complete: quota ledger, policy, sidecar, 402 tests, canary | **canonical** — all new work here |
| `llm-gateway` | new-api inventory + `SPEC.md` + chain ranking | keep as **config/inventory source only**; `SPEC.md` is superseded by this doc |
| `agentic_gateway` | older router, benchmark-based ranking, g4f | **freeze**; port any unique benchmark data, then archive |
| `Autonomous_LLM_Router` | LiteLLM + docker wrapper | **freeze**; it is the thing section 9 says not to build |
| `llm-routed-gateway` | prompt-length → provider | **archive**; superseded by classification + cascade |
| `llm-model-catalog` / `llm-provider-tracker` | metadata sync | keep, read-only inputs |

Put a one-line pointer in each frozen repo's README so future sessions stop
re-litigating. That is a 10-minute task that saves weeks.

---

## 13. Roadmap

Each stage has a gate; do not start the next until the gate passes.

**S1 — Truthful telemetry (1–2 days).**
Stop stamping `success` at HTTP 200. Add `X-Router-Attempt-Id`, accept graded
outcomes, add `verdict` enum, store `timeout` as a failure with elapsed time.
Gate: a scripted fake provider returns confidently-wrong output; the ledger
records it as failure, and `evidence()`'s alpha/beta move accordingly.

**S2 — Client oracle (2–4 days).**
OpenCode plugin: run V0 + V4 always, V2 on the changed subset, from a pristine
worktree; post the grade. Handle the Python/node/Go/rust toolchain detection
from the repo's own config. Gate: 20 real tasks on the owner's repo, grades
match a human read; no test-tamper patch is ever graded pass.

**S3 — Ladder (3–5 days).**
Entry-tier DP, semantic-vs-operational failure split, attempt reservation per
rate-limit axis, escalation with structured failure evidence only. Gate:
replay shows E[T] and E[C] as computed; live p90 T2G improves versus S1
baseline on the same task set.

**S4 — Shadow prices (2–3 days).**
Seasonal demand forecast, per-axis λ, `shadow_prices` table, replace
`quota_lambda * pressure`. Gate: simulation over 4 weeks of traces shows waste
ratio < 15% and no λ > 0 when `Q ≥ 1.15·D`; a synthetic "quota will expire"
scenario does **not** move traffic that cannot be completed in time.

**S5 — One-time credit policy (1–2 days).**
`a(m,class)` from graded outcomes, `μ` per provider, reserve logic, tiny-balance
drain. Gate: shadow mode shows no traffic moved to a credit route with `a ≤ 0`.

**S6 — Placement (3–5 days).**
Snapshot endpoint, local policy mode, async grade posting, `state_mode`
config. Gate: laptop plugin mode and VPS remote mode agree on ≥ 99% of decisions
given the same snapshot.

**S7 — Learned nudge (optional, only after 2000 graded outcomes).**
Logistic regression tie-breaker, offline train, JSON export. Gate: beats rules
by a measurable margin on held-out weeks, otherwise delete it.

---

## 14. Open questions for the owner

1. **Is the repo allowed to run the test suite as part of grading?** Some repos
   have slow/flaky suites; a 3-minute oracle changes the cascade economics
   completely (section 3.3). Which of your repos are "fast oracle" vs "slow"?
2. **Which task classes have no oracle at all?** Prose, design docs, research
   summaries — for those, is a different-family grader model acceptable (costs
   tokens, adds latency), or do you want those to stay single-shot?
3. **Credit value**: are there task classes where a credit-only model is
   genuinely the *only* option (not just better)? That changes `a` from a
   scalar to per-class and justifies hoarding.
4. **Tolerance for wrong-but-green.** Is a human sampling 10 passes/week
   acceptable, or do you want an automated second-opinion reviewer on a sample?
5. **Where does the canonical repo live** — confirm `LLM-API-Key-Proxy` (and
   specifically `workload-router/`) is the one to consolidate into, so the
   others can be frozen with pointers.
6. **Provider list**: I need the actual provider→model→limit table (or a pointer
   to `llm_providers.db`) before S4, because the forecast and λ need real
   windows (daily vs weekly vs rolling) and real per-axis caps.

---

## 15. Appendix: worked examples

### 15.1 Cascade vs single model

Task: fix a failing test in one file. Routes:

| Route | `s` | `T` | `C` (λ·tokens) |
|---|---|---|---|
| A: baseline, fast, dumber | 0.55 | 4 s | 10 |
| B: smart, flaky | 0.85 | 18 s (p90 45 s) | 40 |

Single-shot B: `T/s = 21 s`, cost 47.
Cascade A→B: `E[T] = 4 + 0.45·18 = 12.1 s`, `E[C] = 10 + 0.45·40 = 28`.

Cascade wins on both axes. Now make B reliable-fast (`T = 8 s`, p90 9 s):
single-shot B = 9.4 s / 47; cascade = 4 + 0.45·8 = 7.6 s / 28. Cascade still
wins. Now make A useless (`s = 0.1`): cascade = 4 + 0.9·8 = 11.2 s, single = 9.4
s — single wins, because you paid 4 s to learn nothing. That is exactly the
"cheap tier is known-bad" gate in 4.2.

### 15.2 Shadow price near a reset

Provider: 500 k tokens/day, resets 00:00. It is 22:00. Forecast demand for the
remaining 2 hours (from the same weekday/hour history): 60 k tokens. `Q = 180 k`.

`Q ≥ 1.15·D` → `λ = 0`. Correct: there is nothing to displace; burning 120 k
tokens in two hours is not possible, and routing to this provider because it
"expires" would only make the result slower.

Now: 22:00, `Q = 40 k`, `D = 60 k`. `Q < D` → `λ` = utility density of the
marginal task that gets dropped, say 400 ms of expected completion per 1 k
tokens. The provider is now genuinely scarce, and only the *difference* (20 k)
is at risk. Note that λ does not care that the clock is near midnight; it cares
that demand exceeds supply. That is the whole correction to the original
intuition.

### 15.3 One-time credit with a mediocre model

Credit: 2 M tokens on provider X, model is ~equal to the best free model
(`a ≈ 0`), provider looks fragile (`μ = 0.15/week`). Option value = `0.15 · 0`
per week = 0. Correct action: do not hoard, and also do not route there for its
own sake — it wins nothing. Use it only when it is *faster* (it becomes a latency
tier) or when a free tier is exhausted (it becomes overflow capacity). This is
the owner's "what if the credits aren't for frontier models" case, and the answer
is: the credit's price is not its balance, it is its advantage.

---

## 16. One-paragraph summary for the reviewing agent

The router already has the right skeleton (hard gates → scored ranking →
bounded failover → SQLite state). The missing thing is **ground truth**: it
currently learns from HTTP 200, prices quota by utilisation instead of
opportunity, and averages away the burstiness that defines free providers. Fix
those three and add a client-side verification loop, and the rest of the
architecture (cascade, shadow auction, hazard-discounted credits, client-side
policy with a shared state snapshot) follows mechanically. Do not add models to
the routing path, do not add dependencies that cost RAM, and do not build a
second router in a sibling repo.
