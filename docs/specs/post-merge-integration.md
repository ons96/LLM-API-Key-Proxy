# Post-Merge Gateway Integration

Integrate the vendored workload-router with the existing asynchronous gateway as an observable, default-off decision layer: preserve the current provider adapters, virtual-model chains, streaming/tool history, free-only policy, cooldowns, and client behavior; emit comparable router decisions and bounded diagnostics at the existing routing boundary; and make any later cutover reversible through explicit configuration, without deploying or changing live traffic in this milestone.

## Acceptance criteria

1. With the integration flag disabled, existing gateway routing, fallback, streaming, tool calls, virtual-model aliases, and response contracts produce unchanged results in regression tests.
2. With observe-only mode enabled, representative requests produce redacted, structured router decisions containing group, selected candidate, chain reason, phase/session identifiers, and fallback metadata without changing the selected upstream request.
3. A separate controlled test proves that enabling the integration flag can be reverted without code changes and that provider failure, quota exhaustion, malformed streams, and router errors preserve bounded existing-gateway fallback behavior.
