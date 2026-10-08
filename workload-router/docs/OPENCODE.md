# OpenCode integration

Configure OpenCode to use the router's `auto`, `auto-code`, or `auto-chat`
alias. A plugin or client wrapper should send these headers on requests:

- `X-Session-Id`: stable session identifier;
- `X-Router-Phase`: `explore`, `plan`, `implement`, `verify`, or `debug`;
- `X-Router-Action`: bounded tool/action identifier;
- `X-Router-Verification`: `failed` after failed verification.

The router keeps a deployment through tool loops and re-evaluates at human-turn,
phase, verification, and stall boundaries. The integration sends signals only;
it does not make provider calls or attempt in-place model rebinding.
