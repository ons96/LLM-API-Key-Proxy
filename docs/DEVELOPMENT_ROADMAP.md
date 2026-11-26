# Development Roadmap

## 1. Baseline and strategy

Baseline: `main`, `5acacda126d6f9385f8ca17f1e4d032a93442f36`, audited 2026-11-23. Findings and verification qualifications are in [PROJECT_AUDIT.md](PROJECT_AUDIT.md). Audit checks passed 403 sidecar and 50 selected legacy tests; this is not deployment or financial-account proof.

**Sequence: close monetary/privacy exposure, establish one accountable execution lifecycle, add honest task feedback, then authorize a narrow cutover.** Keep the current layering: New-API transport, `workload-router/newapi_server.py` front door, LiteLLM rollback. Do not build a third router or rewrite the stdlib core. Freeze legacy feature development, but apply narrowly scoped safety fixes while it remains reachable.

This document proposes future work; none of it was implemented during the audit. Repository rules requiring changes to README/STATE/tests apply to those future implementation tasks, not to this documentation-only audit.

Priority rubric:

- **P0 - Critical:** monetary exposure, credential/privacy leaks or unsafe exposed operation.
- **P1 - High:** accountable routing, agentic reliability, quota correctness, release/cutover gates.
- **P2 - Medium:** bounded storage, maintainability, observability and post-core improvements.
- **P3 - Later:** optional specialization, experimentation and distribution enhancements.

Every implementation must produce a narrow diff plus a passing acceptance check. Default validation is provider-free: synthetic catalogs/accounts, stub adapters and loopback HTTP fixtures with disposable owner-only databases. Do not run live discovery/completions or mutate accounts/infrastructure without separate authorization. No paid providers, paid graders, automatic top-ups or overage are part of this roadmap.

## 2. Dependency-aware phases

| Phase | Work | Exit gate |
| --- | --- | --- |
| 0. Contain and prove zero-money/privacy | R01-R04 | All reachable dispatch surfaces fail closed on paid/unknown/ambiguous funding; required auth and log/privacy guarantees hold in local matrix. Unsafe aliases/auxiliaries are rejected until supported. |
| 1. One execution contract | R05-R06, R08-R10 | Physical attempts, stream terminal state, context/capabilities and crash recovery have local acceptance tests; root CI runs both relevant suites. |
| 2. Honest agentic evidence | R07; integrate into R05/R06 | Unknown quality is not positive correctness evidence; authenticated, correlated client outcomes and bounded stall handling work end-to-end with scripted tools. |
| 3. Authorized New-API opt-in | R11, after phases 0-2 | Pinned disposable transport proof, sanitized current account/channel proof, owner-approved narrow deployment, persistence/load/client/rollback gates. No default switch yet. |
| 4. Sustain and expand | R12-R14; R15 only if public/multi-user use is pursued | Storage/reporting/metadata limits are verifiable. Expansion uses financial/capability gates, not more guessed providers. |
| 5. Optional specialization | R16 | Graded workload evidence demonstrates the need; no safety or quality regression. |

Some offline work can overlap (auth/logging, CI and context tests), but do not promote a runtime whose financial or attempt-lifecycle gates are incomplete. If current deployed account/channel state is unknown, do not infer it from this checkout or older roadmap tables.

## 3. P0 - Smallest trustworthy financial and privacy boundary

### R01. One monetary eligibility check at startup and every dispatch

**Problem/evidence:** F01-F02. YAML disables legacy free-only; virtual/MoE paths differ; RotatingClient embeddings bypass it. Sidecar credit and embedding routes can carry positive customer prices. Runtime does not reject priced catalogs at startup.

**Change:** create one explicit eligibility contract inside the existing canonical sidecar, plus the smallest legacy containment guard while rollback remains available. All dispatches must require known zero customer input/output/cached tariffs, verified provider/account/model funding, enabled status and no paid/unknown fallback. Keep promotional-credit prices separate from money. In a zero-money process, a priced/ambiguous dispatch catalog is a startup error, not silently usable. Paid rows may be retained as disabled diagnostics, never dispatch candidates. Avoid multiple flags implying contradictory policy; expose the actual effective policy in diagnostics.

Use the guard for ordinary chat, virtual aliases, direct references, MoE experts **and aggregator**, Responses, embeddings, fallback, streaming and each retry. Recheck at dispatch after ranking/config changes. Temporarily reject unsupported auxiliaries/MoE rather than implement another complete legacy router.

- **Components:** `workload-router/quota_ledger.py`, `route_policy.py`, `newapi_runtime.py`, `newapi_server.py`; minimal legacy `router_core.py`, `main.py`/RotatingClient admission and deployment policy defaults.
- **Dependencies:** none for mechanical zero-price gates; final eligibility depends on R02's provenance/account constraints.
- **Acceptance:** table-driven tests across every entrypoint and retry branch show **zero adapter calls** for paid, unknown, missing tariffs, positive cached tariffs, contradictory funding and paid models on free-tier providers. Positive-money credit and embedding reproductions from the audit become denial tests. A priced enabled catalog fails before bind. Good free-limited/credit routes remain usable under explicit valid allowances. Actual reported policy matches startup state, not an unrelated env log.
- **Offline:** yes for code and synthetic catalogs. Real account qualification is a separate no-completion operator evidence gate in R02.
- **Tradeoff:** availability decreases where pricing cannot be proven. That is required fail-closed behavior, not a reason to default to free.

### R02. Funding provenance and real quota metadata, not marketing flags

**Problem/evidence:** F08/F12. Legacy expects a catalog mapping but receives a list, defaults failed credit reads to available, and checked-in dispatch fields lack effective rate limits. Free-tier provider flags do not establish model/account no-overage. Sidecar quota taxonomy is better, but does not model credit expiration or billing assurances.

**Change:** use the existing provider-manager DB/import/export tooling as metadata authority; add a validated route snapshot tying provider, account/credential fingerprint, physical model, funding class, zero customer tariffs, quota/reset axes, credit balance/expiry and verification provenance. Contradictory `paid`/free flags must not become free. Reject expired/unverified snapshots and exhausted/unknown finite allowances. Correct only the legacy schema/limit bridge needed for safe rollback; do not hardcode another provider whitelist.

Promotional credits are acceptable only if the account cannot create monetary debt or overage. Require owner-sanitized evidence of disabled top-ups/billing fallback and fixed model identity; if unavailable, leave the lane disabled. No quota probe/completion is necessary for policy compilation.

- **Components:** `provider_classification.py`, `newapi_catalog.py`, `newapi_operator.py`, Route metadata; legacy `_get_provider_credit_cfg/_compute_rate_limits`; existing provider-manager tooling outside this repo for approved later integration.
- **Dependencies:** R01's invariant; existing DB schema/import policy.
- **Acceptance:** list and mapping legacy fixtures compile to identical valid metadata or reject explicitly; malformed reads never authorize finite-credit dispatch. Missing/expired balance, conflicting flags, unknown reset and paid model fixtures fail closed. Configured and learned limits are visible and actually passed to reservations. Snapshot changes are previewed and reversible; removed models become disabled, not erased. No credentials are persisted. Distinct free-quota/credit/money fields have unit tests.
- **Offline:** code/import/schema checks yes; genuine account billing/overage/expiry proof requires authorized operator evidence, not invented labels.
- **Tradeoff:** retain last-good metadata for diagnostics, but do not use a stale snapshot to authorize unknown financial exposure. Explicit refresh workflow adds maintenance, less risk than silent discovery.

### R03. Secure the legacy surface while preserving local rollback

**Problem/evidence:** F04. Missing-key open access, unauthenticated mounted model/status/CLIProxyAPI routers, network-default binds and shared session mutation.

**Change:** apply application/router-level auth to non-liveness endpoints; fail startup on missing key outside an explicitly selected loopback development mode. Separate administrative mutation/exports from read-only discovery. Prefer loopback default; any network access needs a separately approved authenticated TLS front and verified firewall. Do not rely on interactive launcher setup to protect ASGI imports.

- **Components:** `main.py`, `model_api.py`, `status_api.py`, `cliproxyapi_api.py`, `outcomes_api.py`; deployment examples in a future implementation diff.
- **Dependencies:** none; coordinate with R04 on safe error messages.
- **Acceptance:** enumerate mounted routes in tests. With a key set, every data/write/export route rejects missing/wrong/duplicate/malformed auth; root/health expose only liveness. Missing key prevents non-loopback startup. Ordinary token cannot perform administrative changes. Legacy model switches cannot modify another authenticated identity's session. Stub dependencies ensure no probes/startup discovery run in tests.
- **Offline:** yes. Current network/firewall exposure must be checked separately before any network-facing change.
- **Tradeoff:** compatibility changes for scripts that previously read status anonymously; document explicit exceptions rather than preserve insecure access.

### R04. Preserve protected-alias semantics and eliminate sensitive persistence

**Problem/evidence:** F03/F05. `safe-coding` silently becomes ordinary coding; reachable legacy failure logs store headers/upstream bodies; embeddings/status echo raw errors; old prototype/session telemetry is not hash-only.

**Change:** reject protected aliases unless an explicit protected route is verified. Never bypass a scrubbing service on failure. Use allowlisted structured error/log fields; remove credentials, request headers/bodies, tool outputs and arbitrary upstream errors from default persistence. Replace auxiliary raw errors with safe categories/request IDs. Hash or keyed-pseudonymize session/task IDs within the appropriate identity domain; exclude free-form task notes by default. Redact/gate snapshot error text. Keep opt-in diagnostics bounded, owner-only and still credential-redacted.

Do not wire the currently unused DetailedLogger just to fix a historical assertion; address actual failure_logger/task/prototype persistence first.

- **Components:** `newapi_runtime.py`, alias listing, `provider_status.py`, `router_gate.py`; legacy `failure_logger.py`, `task_telemetry.py`, `telemetry/logger.py`, auxiliary handlers and `detailed_logger.py` if retained as a callable utility.
- **Dependencies:** R01 for dispatch policy; R03 for diagnostic access.
- **Acceptance:** `safe-coding` returns bounded rejection with zero upstream calls unless a mocked verified scrubber is installed; scrubber failure never dispatches raw context. Canary fake secrets/prompts/Authorization headers in exceptions, tool output and request metadata appear nowhere in persisted rows, stdout, SSE or HTTP errors. Test all logging branches, including normal failure logging (not just the CLI flag). Only safe error categories are exported.
- **Offline:** yes for policy/redaction tests. A real protected lane later requires explicit local scrubbing verification; no paid API is needed.
- **Tradeoff:** less raw debugging data; retain safe correlation IDs and bounded metadata instead. Public availability of `safe-coding` may decrease until its promise is real.

**Minimum financial-safety slice:** R01 plus R02's fail-closed snapshot/account rule, with unsafe legacy MoE/auxiliaries disabled until their guard is proven. R03-R04 are the minimum accompanying exposure/privacy slice. Changing `free_only_mode` alone is not completion.

## 4. P1 - Routing, quota, streaming and agentic reliability

### R05. Count each physical attempt and bound retries

**Problem/evidence:** F08. One legacy reservation encloses multiple provider tries; nested chain/alias retries amplify attempts; stream usage is missing; waiting precedes shared cooldown publication.

**Change:** define reserved -> started -> output_started -> terminal -> settled transitions. Each physical network try needs its own request/token reservation and timing/error outcome. Publish operational cooldown before a wait so other requests see it. Bound the entire request by an absolute deadline and physical-attempt budget, including aliases/MoE. Disable hidden SDK/transport retries. Never retry a billable or financially ineligible lane, and never repeat uncertain non-idempotent operations automatically. Cancellation releases only provably unstarted reservations.

- **Components:** canonical `newapi_runtime.py`, `quota_ledger.py`, adapters; minimal legacy physical retry loops, rate limiter and stream usage capture.
- **Dependencies:** R01-R02; coordinate terminal state with R06 and process death with R09.
- **Acceptance:** scripted two failures plus success produce exactly three physical calls and three ledger attempts; provider counters match. Zero retry configuration truly yields one try. Alias recursion cannot exceed the global budget. Retry-After/cooldown is visible to concurrent requests before sleep; daily exhaustion is not a transient wait. Failure/cancellation never returns consumed quota as unstarted. Final stream usage and estimated/actual flags persist and survive restart.
- **Offline:** yes, including local socket timeout, client disconnect and concurrency fixtures. External implicit retry proof is R11.
- **Tradeoff:** conservative uncertainty can reserve more free quota than actually consumed; losing availability is safer than undercounting or spending.

### R06. Make terminal streams and structured/tool outputs truthful

**Problem/evidence:** F07. Legacy error content ends with normal stop/DONE; New runtime accepts malformed tool fragments plus DONE; truncation can become completed Responses output.

**Change:** preserve no-post-output-failover guards. Assemble indexed tool-call fragments and validate final names/JSON arguments against the accepted output contract before operational completion. Record terminal finish/truncation/error separately from HTTP and semantic quality. After output, emit a supported terminal failure protocol or unambiguously incomplete connection; never synthesize success or append another model's answer. Keep truncated content/usage available, not silently replaced by a new response.

Document/reject unsupported Responses tool continuations; do not advertise full parity until `function_call_output` and streaming tools are actually supported and tested.

- **Components:** `AccountedStream`, `provider_adapter.py`, `newapi_server.py` Responses shim; legacy `_sse_wrap`, Responses boundary and stream telemetry.
- **Dependencies:** R05 lifecycle, R04 safe errors.
- **Acceptance:** pre-output failure falls back once; post-output failure never starts another provider. `tool_calls:[{}]`, split malformed arguments, missing terminal marker, abrupt EOF, reasoning-only, length/content-filter, late errors and client disconnect have explicit expected outcomes. SDK-like fixture parsers do not see late failure as a successful task/response. Usage is retained once without treating partial usage as final. Indentation/text bytes are preserved.
- **Offline:** yes. Installed-client/scripted transport compatibility can be checked without live models; real-provider late-fault semantics remain a later authorized gate.
- **Tradeoff:** buffering/assembly needs explicit bounded memory; avoid buffering whole text streams merely to grade them.

### R07. Separate operational success from semantic success; wire client evidence

**Problem/evidence:** F06. Null-quality success increases semantic priors; normal handler has no grader/outcome linkage; stall detection only exists in the old server.

**Change:** preserve unknown quality as unknown. Use operational reliability separately from graded correctness, rather than forcing a grade onto every response. Add authenticated, correlated feedback for decision/task/session/turn and grader provenance, using actual harness test/tool/verification outcomes. Do not run another LLM as an oracle or execute model output on the gateway. Integrate bounded normalized action/failure escalation into the **canonical New-API path**, not by switching back to the old server.

- **Components:** `quota_ledger.evidence/grade`, `newapi_server.py` feedback boundary, `stall_detector.py`/existing integration headers, benchmark/client tooling.
- **Dependencies:** R03-R06 provide identity, privacy, attempt IDs and terminal outcomes; R09 if durable escalation is required.
- **Acceptance:** quality `{null,0,1}` has distinct counts; null does not increase correctness alpha. Duplicate/out-of-order/cross-session reports cannot corrupt another decision or double-update priors. A normal scripted tool loop reports a failing verification, changes only appropriate task evidence, escalates within a ceiling, then resets on proven progress. HTTP 200, non-empty content and valid tool JSON never alone label a task correct. Grader failure leaves leases settled and quality unknown. Streaming/nonstream evidence has the same meaning.
- **Offline:** yes for fixture clients/unit tests. Later installed-harness integration does not require provider calls if scripted outputs are used.
- **Tradeoff:** sparse grades are honest; do not fix them with guessed success. Avoid automatic semantic failover after tools/actions already ran. Persist stall state only if the workflow requires it; otherwise explicitly test/document restart reset.

### R08. Align context/capability admission on the promoted path

**Problem/evidence:** F09. New chat gates are stronger than legacy/prototype/embeddings; generic adapters guess support; step-specific quality readiness is not demonstrated.

**Change:** retain conservative text budgeting including tools/schema/system/history/output. Give embeddings an equally safe bound and model/batch validation. Unknown required capabilities fail closed; keep the old message-only estimator explicitly diagnostic/non-production. Validate model-level context/output/tools/schema/stream metadata from approved sources; admission must happen before dispatch and after payload transformations. Use profile/step evidence to establish coding suitability without inventing benchmark floors.

- **Components:** `prepare`, `input_token_bound`, `score_route`, `Runtime.embed`, catalog route validation; minimal legacy safety gates.
- **Dependencies:** R01-R02; R07 supplies task-quality evidence.
- **Acceptance:** large system/tool schema, assistant arguments, response schema, Unicode text and output reserve are all counted; image/audio requests reject with zero dispatch until explicitly supported. Embedding vectors match batch indexes/counts and finite limits. Unknown tool/schema/stream capability rejects; no cache/Thompson preference overrides hard gates. Direct/alias/auxiliary requests cannot evade context admission. Representative agent fixtures are tested intact, not stripped to fit.
- **Offline:** yes. Metadata truth needs later provider documentation/account evidence; exact tokenizer integration is optional after measuring conservative rejection.
- **Tradeoff:** byte bounds over-reject; a lightweight provider-specific tokenizer may improve availability later, but must not consume VPS memory merely to increase theoretical context efficiency.

### R09. Recover process-death leases without refunding uncertain usage

**Problem/evidence:** F08/F11. Outstanding reservations remain charged outside all windows; no startup reconciliation. State persistence alone is not recovery.

**Change:** introduce owner/attempt-generation/deadline metadata and a tested reconciliation rule. Provably unstarted old leases may cancel; started leases from a dead process must finalize as uncertain failure with conservative charges, preserving dispatch/window semantics. Make settlement idempotent. Separate durable route preference/cache/quota from optional ephemeral model/stall state. Do not delete live leases or lifetime credit charges as a cleanup shortcut.

- **Components:** `quota_ledger.py`, startup/shutdown in `newapi_server.py`, optional bounded orchestration state.
- **Dependencies:** R05 lifecycle; R07 for graded-versus-uncertain outcome distinction.
- **Acceptance:** process-kill/restart fixture leaves no permanently pending leases; no double settlement across concurrent recovery; started credit usage is not refunded. RPD/RPM charges cease blocking after the correct settled window, while lifetime credits remain consumed. Startup/shutdown and simultaneous workers preserve exact counters. Schema changes are additive with backup/recovery tests, not destructive resets.
- **Offline:** yes with disposable processes/SQLite; no real database migration during development verification.
- **Tradeoff:** crash uncertainty can overcharge free quota. Explicit reconciliation is preferable to unlimited stale blocking or unsafe refunds.

### R10. Fix startup validation and install a root release gate

**Problem/evidence:** F10/F13. Provider-status env strings fail numeric validation; vendored workflow is not root CI; broad suites can load dotenv/state at import.

**Change:** parse environment types at the boundary; test actual `serve()` configuration branches with mocked AdminClient/proof/state. Add a root workflow/command that runs sidecar and a hermetic legacy safety selection, with offline/mocked network, disposable storage and no credential loading. Respect `requirements.lock` where applicable. Add lightweight lint/type checks appropriate to each subsystem, not a new heavy toolchain. Compile in-memory or outside the checkout in read-only audit contexts.

- **Components:** `newapi_server.py`, root `.github/workflows`, pytest isolation/config, existing requirements/verification tools and future README/STATE updates.
- **Dependencies:** R01-R09 tests can be added incrementally; no live dependency for CI.
- **Acceptance:** default and explicit provider-status intervals start successfully against a fake authorized transport; malformed settings fail safely. CI triggered from repository root actually runs all 403-baseline sidecar tests plus the growing safety matrix. No fake "no tests collected" success. Tests cannot read `.env`, contact external endpoints or touch persistent default DBs. Suite/lint/type failures block release. Pin transport integration proof to a known revision and report fixture versus live results separately.
- **Offline:** yes with existing/local tools; dependencies must be free and locked. Do not reinstall packages just to perform this audit.
- **Tradeoff:** legacy import isolation may expose additional failures; fix harness boundaries instead of masking business failures or asserting a whole repository is green from a subset.

## 5. P1 - New-API authority and cutover

### R11. Promote exactly one brain through a measured opt-in

**Problem/evidence:** F13 and architecture decision. Private-alias Runtime exists, but current account/channel/alias/retry/canary state is unknown; compatibility includes unsafe protected alias, missing task feedback and partial auxiliaries. Historical branch/issue tables are not current proof.

**Change and ordering:**

1. Reconcile local source with current issue bodies #370-#389 and branch integration, without treating comments as remote closure. Review owner-controlled `s0-cost-vector` integration separately; no automatic merge/deploy.
2. Pass R01-R10 and the pinned **disposable** real-New-API verifier using scripted providers, including actual startup/proof path, exact physical-attempt logging, SSE and persistence.
3. Obtain owner-sanitized read-only transport/account evidence: `RetryTimes=0`, no hidden SDK/cross-group retries, one strict alias/channel/credential, exact physical model mapping, no overrides or endpoint drift, authorized proof mechanism, constrained relay token and financial assurances. Resolve any alias collision; do not assume the old `router/tools` table is current.
4. Approve compatibility by consumer: actual alias semantics, discovery, Responses limits, protected aliases, embeddings/status and current task-feedback routes. Reject unsupported contracts explicitly. A client alias must not silently select a different privacy or financial class.
5. After explicit deployment authorization, install only the canonical New-API front door on loopback with separate bounded state/unit. Keep LiteLLM intact as rollback, not chained beneath each sidecar attempt. Optional edge rewriting must not select paid/unknown destinations or duplicate policy.
6. Run a bounded owner-approved free-account tool loop, both-direction failure cases, resource/concurrency measurements, graceful and crash restart, durable recovery and rollback drill. Budget requests/output in advance. Then opt in a small client allowlist; default cutover is a separate owner decision after observed stability.

- **Components:** `newapi_server/runtime/catalog`, operator/verifier tooling, deployment/compatibility docs; external New-API and optional edge inventory under separate scope.
- **Dependencies:** phases 0-2; owner approval for any external mutation or live quota use.
- **Acceptance:** one physical upstream request per started ledger attempt, no paid/unknown traffic in local policy matrix or approved account inventory; actual client tool loop completes with verifiable tool/test outcomes; late failure is not success; restart recovers leases; measured RSS/concurrency stay within chosen sidecar/transport and whole-VPS limits. Rollback requires one documented client/config switch and preserves data/services. No consumer depends on an unreviewed partial endpoint.
- **Offline:** most code/protocol/startup gates yes. Current VPS config, real account policy and live reliability/rollback gates are **not** established by offline tests and require separate authorization. No live test is authorized by this roadmap alone.
- **Tradeoff:** retain a small dual deployment during validation, not two policy engines per request. Do not hold all useful cutover work until optional specialization or multi-user features exist.

**Minimum authoritative-brain slice:** the existing New-API front door plus shared financial guard, single physical-attempt transport contract, truthful streams, complete bounded context/tool admission, recoverable ledger, feedback/correlation and root release gate. Reject unsupported aliases/routes. No broad rewrite or extra classifier is required.

**Cutover blockers:** zero-money/account proof; safe-coding semantics; terminal streams/tool integrity; real per-attempt/limit/recovery evidence; grader/outcome linkage; actual transport alias/retry/key proof; consumer-specific parity; measured service/resource/restart/rollback checks; owner approval. Historical claims of a canary or a passing v1 do not remove these gates.

## 6. P2 - Sustain, observe and maintain

### R12. Bounded retention that preserves financial accounting

- **Problem/evidence:** F11: sidecar history/cache/benchmarks unbounded; legacy retention covers only some stores.
- **Change/components:** configurable retention and bounded batch purge for `quota_ledger`, `router_state`, task/prefix-cache telemetry; summarized evidence; periodic SQLite maintenance and storage diagnostics.
- **Dependencies:** R09 recovery; R02 lifetime-credit/reset semantics; R07 evidence retention.
- **Acceptance:** aged finalized request/window rows can be reduced without changing current capacity, lifetime credits or graded aggregates; pending/live leases never disappear. Cache TTL and storage TTL are tested separately. Seeded large-history fixture stays within configured bounds and queries remain bounded; maintenance can pause/resume. Backup/restore preserves summaries.
- **Offline:** yes. Retention durations/storage limits need an operator policy, not arbitrary audit estimates.
- **Tradeoff:** keep aggregate credit/account history even when request-level history expires; avoid destructive VACUUM/purge against live data without backups and authorization.

### R13. Honest money, quota, quality and reliability reports

- **Problem/evidence:** F12/F06: cheapest quote is not actual attempt cost; null grades/usage ambiguity blur comparisons.
- **Change/components:** ledger diagnostics/reporting expose separate zero actual-money policy/verified debit status, renewable quota, expiring promotional-credit consumption, estimated/actual usage, attempts/terminal outcomes and graded/unknown correctness. Do not label notional tariffs as charges or a non-empty response as success. Correlate request totals across all attempts.
- **Dependencies:** R02, R05-R07, R09.
- **Acceptance:** synthetic retry/credit/free/unknown/truncated tasks yield independently correct money/quota/quality totals; cost estimate explicitly says quote, not bill. Unknown values remain null/unknown. Reports contain no prompts, credentials or user IDs. Comparisons show sample sizes, metadata age and estimation flags.
- **Offline:** yes. Actual account debit reconciliation needs authorized sanitized account evidence; no paid API integration is required.
- **Tradeoff:** more explicit unknowns produce less impressive but more truthful dashboards.

### R14. Consolidate metadata, replay and documentation

- **Problem/evidence:** F09/F13: several policy paths, obsolete counts/chains/flags, replay runs the old selector, model switching is only compatibility state.
- **Change/components:** clarify prototype versus canonical runtime; remove or deprecate unused alternate helpers after caller inventory. Generate safe capability/funding/alias inventories from validated snapshots using existing tooling. Make replay call canonical preparation/ranking with seeded exploration and recorded safe quota/cache snapshots. Update README/SPEC/STATE/compatibility claims in later implementation work; choose whether session switch really affects dispatch or is explicitly diagnostic.
- **Dependencies:** R01-R11; do not delete rollback before separate retirement approval.
- **Acceptance:** one canonical policy path for new requests, replay agrees with live-boundary fixtures for identical safe state/seed; aliases resolve to documented contracts and unavailable model detail does not imply dispatchability. Documentation names exactly which flags are implemented and which component owns them. Provider additions cannot bypass R01/R02/R08 checks.
- **Offline:** yes; verifying external edge/sibling behavior is a separate scoped review.
- **Tradeoff:** small adapters/documentation cleanup over wholesale module relocation. Retain compatibility exceptions where the implementation cost exceeds demonstrated client need.

### R15. Multi-user/public-use prerequisites, only if that becomes a goal

- **Problem/evidence:** F04-F05/F13: one client token, shared session domains/credential pools, insufficient abuse/privacy boundaries and uncertain external license/security posture.
- **Change/components:** authenticated per-user identities and roles; tenant-scoped sessions/grades/quotas; account/credential financial isolation; per-user rate/concurrency/body/output limits; TLS/CSRF/admin controls as applicable; owner-only secrets/storage, retention/export/delete policy and resource/abuse defenses. Review New-API admin/relay surface and distribution/provider terms. Existing infrastructure only; no paid identity or hosting dependency.
- **Dependencies:** R01-R14; treat public exposure as blocked until implemented and reviewed, not a default next step.
- **Acceptance:** cross-user sessions/outcomes/exports cannot be read or changed; one user cannot drain another's allowance or access another's credentials. Anonymous clients cannot trigger provider calls. Quotas/roles survive restart, abuse cannot exhaust all workers, exports are privacy-safe, and no tenant can enable monetary fallback. Legal/security review is explicitly recorded, not inferred from an OAuth session or a local test suite.
- **Offline:** isolation/abuse tests yes; external deployment/terms review and adversarial integration need separate scope/approval.
- **Tradeoff:** materially larger system than a local proxy. Do not implement tenancy until requested use justifies it; meanwhile keep the service private and bounded.

## 7. P3 - Deferred capabilities

### R16. Evidence-led specialization, not speculative expansion

After R01-R14 and graded workload evidence, consider:

- **Agentic coding:** per-step measured suitability/priors and better normalized stall actions. Prefer harness feedback and tests over another model-based evaluator.
- **Research/data analysis:** verified long-context/schema/tool capabilities and task-specific evaluation fixtures; no promise of semantic parity from a profile label.
- **Computer use:** explicit multimodal budgeting, action/tool authorization, idempotency and progress verification. It is not presently supported merely because generic tools exist.
- **Provider expansion:** add only validated free/no-overage accounts/models through existing provider DB/tooling. New providers must pass the same contract matrix; no bulk import as a substitute for correctness.
- **Optional routing experiments:** alternative priors, embedding classification or cache/latency optimizations only when graded data identifies a bottleneck and resource usage remains suitable for the constrained VPS.

Dependencies: safe authoritative runtime and representative graded data. Acceptance: seeded offline benchmarks show a specific benefit without new financial/capability/privacy/latency regression; future live work stays explicit and bounded. Tradeoff: every feature increases metadata and policy maintenance. No timeline is assigned without evidence.

Do **not** prioritize now: a Go rewrite, another router, New-API fork, multi-model MoE/council generation, paid fallback, paid graders, broad automatic provider imports, arbitrary system/code compaction, fuzzy semantic oracles, fancy dashboards, public launch or new cloud infrastructure. Whitespace-only tool JSON compaction is already useful; changing prompts/tool output semantics requires a demonstrated need and fidelity tests.

## 8. Validation and completion checklist

1. Each change references an audit finding, adds an acceptance test and updates the applicable future docs/state. Preserve established style and existing libraries/tooling.
2. Unit matrix: funding/price/account/model combinations, aliases, all auxiliaries, MoE expert/aggregator rejection, cache tariffs, unknown metadata and quota expirations.
3. Execution matrix: reserve/setup/start, per-physical retries, deadline, cooldown visibility, early/late SSE faults, split tools, final usage, cancellation, process death, recovery and idempotent settlement.
4. Quality/privacy matrix: operational versus semantic versus delivery state; null/failed/passed grades; authenticated feedback correlation; fake secrets in all logs/errors/exports; protected-route fail-closed behavior.
5. Boundary matrix: route-wide auth, malformed input/duplicate headers/body limits, unsupported Responses/multimodal requests, context/tool/output limits and batch identity.
6. Offline integration: canonical server with scripted loopback providers, pinned disposable New-API and installed scripted client. Explicitly label fixture proof; no current account or live provider proof is inferred.
7. Authorized operational gate: current sanitized account/transport state, bounded free live requests only if approved, resource/restart/rollback checks, then allowlist and separately approved default promotion.
8. Keep regression suites provider-free. A passing subset, planned issue or configured cron is not task completion. Never refund uncertain started usage, claim actual money saved from notional tariffs, or mark ungraded tasks correct to make a dashboard look better.
