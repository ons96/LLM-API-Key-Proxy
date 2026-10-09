# P0 Router

Implement a free-only, resource-light OpenAI-compatible workload router that
uses deterministic classification, hard capability filtering, deployment-aware
selection, bounded provider failover, SQLite telemetry, session stickiness,
basic phase/stall escalation, and safe fallback behavior, while preserving the
repository's existing portable artifacts and keeping VPS deployment changes
out of scope unless explicitly requested.

## Acceptance criteria

1. Existing validation passes, and any new P0 behavior has focused tests for
   deterministic routing, hard filtering, bounded failover, and safe fallback.
2. A session remains stable through tool-loop continuations, while meaningful
   phase changes or repeated failures trigger bounded escalation without route
   oscillation.
3. `STATE.md` records each verified milestone, and the implementation stays
   free-only, secret-free in persistent logs, and within the 1 GB VPS resource
   budget.
