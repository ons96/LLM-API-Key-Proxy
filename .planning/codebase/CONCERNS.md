# Concerns

- **Provider stability (free tiers)**: G4F and other free providers are unreliable/slow; rate-limit and availability issues may break fallback chains. Virtual models rely on these chains (`config/virtual_models.yaml`).
- **Rate limits & errors**: Adapters should map provider errors to HTTP status; rate-limit handling is critical. Telemetry/health tracking exists but not all adapters update it uniformly.
- **API keys & secrets**: Gateway auth via `verify_api_key`; provider keys pulled from env/credential tools—must avoid logging secrets. `.env` is required but not committed; missing envs will break providers.
- **Supacoder streaming**: Stall timeout is enforced; ensure `SUPACODER_STREAM_TIMEOUT_SECONDS` is set appropriately and telemetry import path remains valid.
- **Typing/lint debt**: BasedPyright reports many warnings (deprecated typing aliases, Any usage) across adapters; not fatal but noise for CI/type-check.
- **Config drift**: Multiple virtual model configs (primary/backup/generated); risk of divergence. Router/ranker behavior depends on YAML correctness and benchmark data.
- **Testing realism**: Test suite uses mocks/fixtures; live provider behavior (esp. free tiers) may differ. Some tests marked expected-fail per tests/README.md.
- **Security**: Ensure API key enforcement stays enabled; avoid storing user data; restrict network exposure when running uvicorn.
- **Operational**: Need monitoring around `/v1/chat/completions` and `/v1/responses` for latency/timeouts; streaming paths and tool-call finish_reason handling should be observed.
