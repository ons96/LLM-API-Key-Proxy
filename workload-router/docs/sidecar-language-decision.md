# Sidecar language and resource decision

Date: 2026-11-03
Status: keep Python for the v2 sidecar. No rewrite is planned and no deployment
was performed.

## Decision

The sidecar stays on Python 3 and the standard library. Its measured idle RSS is
small enough for VPS-40, and the existing implementation already contains the
quota ledger, SSE handling, alias mapping, audit checks and 324 provider-free
tests. A Go or Node rewrite would add a second implementation and a second
security/accounting review without solving a demonstrated resource problem.

Start the sidecar under a 128 MiB `MemoryMax` when it is eventually installed.
That is a guardrail, not a substitute for a live load test.

## Measured baseline

Measurement on the development host, Python 3.13:

- Started `newapi_server.py` with an in-memory ledger and an in-memory session
  store.
- Sent 50 authenticated `GET /v1/models` requests.
- Read `VmRSS` from `/proc/<pid>/status`.
- Result: **26,984 KiB** (about 27 MiB) RSS.
- No provider request, credential, or VPS service was involved.

The older router-only resource smoke reports approximately 26,352 KiB, which is
consistent with the sidecar measurement. These are local idle/discovery
measurements, not a VPS concurrency or streaming benchmark.

VPS-40 has approximately **954 MiB** total RAM and 5 GiB swap. The existing
New-API process used approximately **58-64 MiB** in the recorded inventory. The
sidecar's 27 MiB baseline is therefore a small fraction of the host, even before
the old gateway is retired. The combined stack still needs a canary measurement
under real streams.

## Why not Go

Go would produce a small static binary and could use less memory at idle, but
the current sidecar is not memory-bound. A rewrite would have to reproduce:

- SQLite reservation transactions and only-tighten quota learning;
- SSE priming, failover and stream cleanup;
- alias intent and phase mapping;
- the strict New-API audit and channel-key proof boundary;
- request limits, error classification and all existing regression tests.

That is a high-risk change for an unmeasured memory gain. Go remains a valid
option if a later profile shows Python's CPU or memory is the limiting factor.

## Why not Node/TypeScript

Node has a good streaming model, but it adds a package/runtime surface and its
idle RSS is not expected to beat the measured Python baseline. The project has
no Node service to reuse, so it would also be a full rewrite. No measured reason
to choose it exists.

## Constraints that keep Python appropriate

- The workload is I/O-bound: waiting on New-API and upstream streams, not local
  model inference.
- The server has a bounded worker semaphore (currently eight) and a 1 MiB request
  cap.
- SQLite access is local, transactional and short-lived.
- The sidecar does no local ML inference and loads no model weights.
- The current suite is provider-free and fast enough to run before every release.

## Revisit triggers

Reconsider the language only if a canary or representative load test shows one
of these:

- sustained RSS above the 128 MiB cap with normal request sizes;
- CPU saturation or request latency caused by the Python process;
- a need for a CPU-heavy local classifier or tokenizer;
- concurrency requirements that the bounded thread model cannot meet.

Until then, the lower-risk path is to keep Python, measure the real canary, and
fix the compatibility and operational gaps rather than rewrite the runtime.

## Related documents

- `docs/gateway-compatibility.md` — current HTTP surface and remaining gaps.
- `docs/newapi-v1-v2-roadmap.md` — staged replacement gates.
- `docs/DEPLOYMENT.md` — deployment and rollback constraints.
- `../HANDOFF-newapi-sidecar.md` — current continuation point.
