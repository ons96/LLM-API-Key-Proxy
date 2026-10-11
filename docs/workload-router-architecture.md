# AI Workload Router Architecture

## 1. Goals and constraints

The router minimizes expected wall-clock time to successful completion. Its
secondary objectives are conserving scarce API calls/tokens and reusing warm
prompt caches. It supports OpenCode coding agents and generic chat through one
OpenAI-compatible endpoint.

Hard constraints:

- Free-only operation; no mandatory paid API, hosted classifier, or service.
- Control plane fits a 1 GB Oracle VPS with weak CPU, zram, and no GPU.
- Local classification is deterministic, tiny, low-latency, and reliable.
- Provider credentials and volatile endpoints remain outside persistent data.
- Raw prompts are not persisted by default.

## 2. Recommended architecture

```text
OpenCode / generic client
        |
        v
OpenAI-compatible router (auto, auto-code, auto-chat)
        |
        +-- request normalization and feature extraction
        +-- session/phase/stall state
        +-- deterministic classifier
        +-- hard capability filtering
        +-- expected-time-to-success selector
        +-- provider adapter and bounded operational failover
        +-- asynchronous SQLite telemetry
        |
        v
Provider/model deployments
```

The router makes three separate decisions:

1. Infer mode, task, phase, risk, context size, and hard constraints.
2. Select an abstract capability group.
3. Select a concrete deployment: model + provider + configuration.

Suggested capability groups are `FAST_GENERAL`, `FAST_TOOL`, `CODE_EXPLORE`,
`CODE_IMPLEMENT`, `CODE_VERIFY`, `CODE_DEBUG`, `DEEP_REASONING`, and
`FRONTIER`. Groups are phase-aware rather than a single arbitrary model chain.

The P0 implementation should be a small Rust, Go, or Node service with SQLite,
in-memory hot state, and asynchronous telemetry. LiteLLM may be used later as
a transport/provider normalization layer, but its Python RSS and feature
coverage must be measured before deployment.

## 3. Routing flow

### Stage 0: explicit constraints and state

Check, in order:

- Explicit model or capability-group request.
- Context-window, modality, tools, tool choice, structured-output, reasoning
  effort, privacy, and request-format requirements.
- Existing session decision, minimum dwell time, and tool-loop stickiness.
- Provider/model health, cooldowns, quota class, and cache records.

### Stage 1: local deterministic classification

Extract only bounded metadata and lexical features: client mode, OpenCode phase,
tool presence, test/lint output, context estimate, requested format, risk words,
error class, repeated calls, and progress signals. Rules classify obvious
requests and detect stalls without embeddings or neural models.

Obvious requests route immediately. Ambiguous, high-risk, novel, or conflicting
requests use a safe capability default or optional Stage 2.

### Stage 2: optional typed classifier

JEV or another hosted typed-decision service is optional P1 infrastructure. It
may select a capability group or tier, never an exact deployment. Calls require
a strict timeout, circuit breaker, and local fallback. Never call it on every
tool-loop continuation. Its hosted/free availability is configuration-driven;
the router must work fully without it.

### Candidate selection

Filter hard requirements first, then score healthy deployments. Treat model,
provider, and configuration as one deployment identity. A conceptual objective
is:

```text
ET_success(c) = decision_overhead + TTFT(c) + decode_time(c)
  + operational_retry_time(c)
  + (1 - p_success(c)) * recovery_time(c)
  + cache_miss_penalty(c) + quota_penalty(c)
```

Choose the fastest eligible candidate whose estimated success confidence clears
a risk-dependent threshold. Thresholds are lower for disposable chat, higher
for ordinary coding, and highest for destructive operations and verification.
Use external rankings only as priors; personal telemetry updates them with
EWMA or Bayesian smoothing and monotonic tier constraints.

## 4. Deployment scoring and failure handling

Deployment profiles include context and output limits, capability tags, tool and
JSON support, supported effort, benchmark priors, quota class, cache behavior,
and configuration fingerprint. Live scoring includes robust latency percentiles,
TTFT, tokens/sec, success rate, error rate, cache value, quota scarcity/expiry,
provider affinity, and cooldown status.

Track provider health separately from model capability.

- Provider timeout, 429, 5xx, or connection failure: bounded retry and failover
  within the same capability group; then cooldown the deployment/provider.
- Model/tool protocol failure: try another compatible deployment, then escalate
  if the failure appears model-specific.
- Failed tests, semantic failure, or repeated incorrect work: escalate the
  capability group or effort level rather than retrying the same tier.
- Partial stream failures must not blindly replay requests that could duplicate
  work; retry policy is provider and idempotency aware.
- If the router or classifier is unavailable, use the deterministic safe default
  and continue whenever any eligible deployment exists.

Operational failover and semantic escalation are separate event types.

## 5. Session, phase, and stall policy

Track `EXPLORE`, `PLAN`, `IMPLEMENT`, `VERIFY`, `DEBUG`, and
`DEEP_DEBUG/FRONTIER` (or equivalent) in session state. Reclassify on a new
human turn, phase transition, verification result, or stall boundary. Keep the
current deployment through a raw continuation and tool loop.

Stall signals include identical tool calls and arguments, repeated failing
commands/tests, repeated edit patterns, consecutive failed verification, and a
no-progress window. Escalate once, enforce a ceiling and cooldown, and reset
after meaningful progress. Minimum dwell time, upgrade-on-stall guards,
hysteresis, and a no-oscillation rule prevent thrashing.

## 6. Cache-aware switching

Maintain cache records per session and deployment, not one global warm flag:

- model, provider, deployment/configuration fingerprint, and session key;
- reusable-prefix hash and length;
- cached-token/cache-write observations, estimated TTL, last hit, and confidence;
- provider cache behavior and prompt-cache key.

Do not switch in the middle of a tool loop by default. At a valid boundary,
switch only when the expected success-time improvement exceeds cache loss,
serialization, hysteresis, and recovery costs. Downgrades require stronger
evidence than upgrades. Keep a small map of prior session deployments so a
switch-back can use a warm cache when available.

Provider-specific usage parsing should capture fields such as OpenRouter
`cached_tokens` and `cache_write_tokens` when present. Cache TTLs and minimum
prefix sizes vary by provider, so estimates are explicitly provider-scoped.

## 7. OpenCode integration

Configure OpenCode to use one gateway alias. The plugin is a signal layer: it
sends phase, category, risk, session, and context metadata in headers. It must
not depend on rebinding the model in place, because released OpenCode plugin
APIs may mutate headers/system transforms without reliably changing the model.

Useful signals include user messages, session creation/compaction, todo updates,
tool completion, diagnostics, and test/lint output. The plugin does not make
expensive classifier calls. The gateway returns selected deployment, group, and
route reason in response metadata/headers where supported. Compaction preserves
router state and phase. Provider adapters normalize tool, reasoning,
structured-output, streaming, and cache-control fields.

## 8. SQLite data model

Static `deployments` fields: `deployment_id`, `model`, `provider`, endpoint
alias, configuration fingerprint, context/input/output limits, capability tags,
supported effort, benchmark prior/source/version, and quota class. Secrets are
never stored.

Live deployment fields: latency and TTFT rolling statistics, tokens/sec,
success/error counts by class, availability/cooldown, quota remaining/reset/
expiry, cache support/TTL/hit ratio, and last verification.

Request/decision records contain request and session hashes, mode/task/phase/
risk, context estimate, bounded features, classifier stage/confidence, selected
group/deployment, reason, candidate scores, cache estimate, fallback/escalation
events, token/cache usage, completion result, and verification result. Persist
aggregates and hashes by default; secure debug logging is opt-in.

Telemetry writes are asynchronous and cannot block completion. SQLite must
survive restart and concurrent requests; use transactions, short lock windows,
and bounded queues.

## 9. Roadmap

### P0

Implement one OpenAI-compatible endpoint; aliases and explicit modes; static
groups; SQLite profiles; deterministic classification; hard filtering; provider
health/cooldowns; bounded same-group failover; session/phase state; tool-loop
stickiness; cache metadata; latency/TTFT/tokens/sec telemetry; error classes;
repeated-tool/test stall detection; quota classes; OpenCode headers; route
diagnostics; and safe router-unavailable behavior.

### P1

Add JEV only for ambiguity, calibrated confidence, EWMA/Bayesian per
deployment/task/phase/context, adaptive thresholds, richer quota economics,
shadow evaluation, profile refresh, optional fastText-style classifier, context
escalation, and expected-value cache switching.

### P2

Consider contextual bandits, learned routing, speculative parallelism,
automatic model discovery, large benchmark generation, and multi-agent
orchestration. None may block P0.

## 10. Implementation sequence

1. Freeze request/response compatibility, configuration, and schema.
2. Implement pure feature extraction and deterministic classification.
3. Implement hard filtering and deployment scoring.
4. Add provider failover and cooldowns.
5. Add SQLite persistence and asynchronous telemetry.
6. Add session, phase, stall, hysteresis, and escalation state.
7. Parse cache and usage metadata.
8. Add the OpenAI-compatible server and diagnostics.
9. Add the thin OpenCode header plugin.
10. Replay representative traces in shadow mode.
11. Add optional JEV or fastText only after measuring the baseline.

Use the workspace `llm-provider-manager` SQLite database for static provider and
model metadata. Source URLs and credentials from approved configuration or
environment variables; do not hardcode secrets or endpoints.

## 11. Feasibility and alternatives

Rules, in-memory maps, SQLite, and a small Rust/Go/Node service fit the VPS.
FastText-style models can eventually be sub-megabyte but need workload labels
and calibration. Local embeddings and transformer semantic routers are not P0.
LiteLLM is useful for normalization and fallback if measured RSS is acceptable,
but Python deployments can consume hundreds of megabytes. LiteLLM Laya's
hundreds-of-millions-parameter checkpoints and Nimble's 9B classifier are not
appropriate for this VPS. LiteLLM Auto Router and JEV remain optional P1 tools.

## 12. Risks and verification

Risks include boundary misclassification, stale benchmark priors, sparse or
biased telemetry, provider-specific cache semantics, history incompatibility on
switches, uncalibrated JEV confidence, changing free quotas, OpenCode API drift,
LiteLLM runtime drift, prompt leakage, and duplicate work after stream retry.

Test cold/warm cache, switch-away/switch-back, expiry, compaction, long context,
tools, structured output, streaming, 429/5xx/timeouts, malformed classifier
responses, no eligible candidates, repeated failures, progress reset, quota
expiry, restart, SQLite contention, and provider normalization. Coding outcomes
use tests/compiler/lint/tool success and task completion; generic chat review is
a weaker sampled signal. Compare shadow replay against a fixed-model baseline.

## 13. First-release acceptance criteria

1. OpenAI-compatible chat/messages works for OpenCode and a generic client.
2. Deterministic routing overhead is under 10 ms p50; obvious requests make no
   remote classifier call.
3. Tools, structured output, context, modality, and effort are hard-filtered.
4. Outage/429/timeout failover is bounded and does not silently downgrade tier.
5. Model/provider/configuration telemetry is distinct and queryable.
6. Cooldowns prevent repeated attempts against dead deployments.
7. Tool loops remain stable without oscillation.
8. Phase changes and repeated failures trigger bounded escalation.
9. Operational failure is distinct from semantic escalation.
10. Cache usage is captured where available and switching has hysteresis.
11. SQLite survives restart/concurrency and telemetry cannot block requests.
12. Router/classifier failure still serves through a safe default.
13. Quota remaining/reset/expiry affect soft scoring without eliminating all
    candidates.
14. Persistent logs contain no secrets or raw prompts by default.
15. Trace replay improves expected-time-to-success or completion success versus
    a fixed baseline.
16. Resource tests stay within the VPS memory and CPU budget.
