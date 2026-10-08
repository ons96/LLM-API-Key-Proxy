# Deferred decisions and external actions

This file records work intentionally left for a later session because it needs
credentials, host access, or an explicit repository decision.

## Provider and VPS

- Code-capable VPS profiles are now mapped to the existing gateway: `coding-smart`
  handles tool/implementation/debug and `coding-fast` handles verification.
  Credentials remain outside git in the mode-0600 VPS environment file.
- The canary currently routes ordinary chat through the existing LLM gateway;
  decide later whether production traffic should remain there or use direct
  upstream adapters.
- Run provider-backed timeout, 429, 5xx, quota, and streaming tests on VPS 40.
- The provider-free side-by-side canary is installed and tested on port 8123;
  the existing gateway remains active on port 8000. Defer reverse-proxy
  cutover until provider configuration and compatibility testing are complete.
- Provider-backed tool-call, structured-output, code-phase, debug-phase, native
  stream, and cooldown checks passed on the side-by-side canary. Remaining
  external checks are provider-specific timeout/429/5xx/quota behavior and
  OpenCode client compatibility before any cutover.
- The VPS has no `opencode` executable. The router's OpenCode signal-header path
  was exercised directly, but a real client session still requires the client
  to be installed or tested from another host.

## Repository integration

- Decide whether this project remains a separate private repository or is
  integrated into `LLM-API-Key-Proxy` through a reviewed branch/PR.
- The existing `LLM-API-Key-Proxy` checkout has unrelated uncommitted changes;
  do not merge or overwrite them without an explicit clean integration point.
- Configure a remote only after choosing the destination repository and access
  policy.

## OpenCode

- Confirm the target OpenCode plugin API/version and preferred installation
  location before adding a version-specific plugin package.
- Verify which phase, action, and verification events are available in the
  deployed OpenCode client.

## Calibration

- Supply redacted outcome JSONL or an approved telemetry export for personal
  success-rate calibration.
- Decide whether semantic outcomes are supplied by tests/tools, the client, or
  an operator review workflow.
