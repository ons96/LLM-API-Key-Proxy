# New-API channel-key proof helper

Production startup requires `NEWAPI_CHANNEL_KEY_PROOF_COMMAND` whenever a
New-API channel audit needs a hidden channel key. The sidecar invokes the
operator-controlled helper once per channel.

## Contract

- The command is configured as an executable plus fixed, non-secret arguments.
- The sidecar writes only the numeric channel ID followed by `\n` to stdin.
- The helper must write exactly one non-empty proof token to stdout.
- The helper has a 30-second timeout and a non-zero exit status fails startup.
- The proof token is validated in memory and is never written to the ledger,
  logs, argv, or repository.
- The helper owns the interactive or pre-approved authentication flow. Do not
  put passwords, backup codes, API keys, or enrollment secrets in this repo.

Example configuration shape (use an operator-owned path, not this example):

```text
NEWAPI_CHANNEL_KEY_PROOF_COMMAND=/usr/local/sbin/newapi-channel-proof --mode=operator
```

The helper must read the channel ID from stdin rather than accepting it as an
argv value. A production helper should reject unexpected input, use a
least-privilege account, and return a non-zero status when authorization cannot
be completed.

## Reference implementation

`tools/newapi_channel_proof.py` implements this contract with two credential
sources, so selecting one is a configuration change rather than a code change:

| Mode | Flag | Credential source | Use |
| --- | --- | --- | --- |
| `interactive` | default | 2FA or backup code typed at the controlling terminal | attended canary |
| `totp` | `--mode=totp` | base32 seed in a `0600` operator-owned file | unattended service |

Configuration comes from an operator-owned env file, never argv:

```text
NEWAPI_PROOF_ADMIN_TOKEN=<admin token>
NEWAPI_PROOF_ADMIN_USER_ID=<numeric user id>
NEWAPI_PROOF_BASE_URL=http://127.0.0.1:8001
NEWAPI_PROOF_TOTP_SECRET_FILE=/etc/newapi-channel-proof/totp.seed   # totp mode
```

Hardening in the implementation: loopback-only base URL (the admin token cannot
be sent off-host), redirects refused, 10-second HTTP timeout, one line of stdin
only, `0600` seed required, and failure messages that never echo a credential,
code, or proof. It returns the proof without redeeming it; the sidecar redeems
it against `POST /api/channel/<id>/key`, so consuming it in the helper would
break that call.

### Operational caveat: proofs are per-channel and per-startup

`serve()` audits every configured route at boot, so a three-channel install
needs three successful proofs on every restart. 2FA backup codes are single
use, which makes them unsuitable beyond a few restarts. Before choosing a mode,
confirm against a disposable instance whether New-API accepts the same TOTP
window code for several channel proofs in one audit; if it does not, the
startup audit needs an approved reusable proof source instead of single-use
codes. This is not yet tested.

## Validation checklist

1. Run the provider-free test suite.
2. Test a valid proof through the helper with a disposable New-API instance.
3. Test missing, non-zero, empty, multi-line, and invalid-token output; each
   must fail closed before the sidecar serves traffic.
4. Confirm helper stdout/stderr and sidecar logs contain no credential values.
5. Configure the helper only on the VPS canary after an operator has approved
   the credential flow. Keep the existing LiteLLM gateway and rollback path.

The disposable verifier's generated 2FA enrollment is test-only and must not be
reused as a production proof mechanism.
