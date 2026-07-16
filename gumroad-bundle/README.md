# Free LLM Gateway Starter Kit

Setup materials for a self-hosted, OpenAI-compatible gateway. This bundle does
not include gateway source, hosting, provider accounts, API keys, uptime, or
support.

## Included files

| File | Purpose |
|---|---|
| `SETUP_GUIDE.md` | Secure installation, local testing, SSH tunnel, and HTTPS proxy guidance. |
| `.env.starter` | Sanitized environment template. |
| `quickstart.sh` | Idempotent Ubuntu/Debian systemd installer bound to loopback only. |
| `GUMROAD_LISTING.md` | Unpublished product-page draft. |
| `CHANGELOG.md` | Bundle version history. |
| `package-release.sh` | Packages and validates the exact buyer archive. |

## Product terms

One one-time product: USD 19 for the first 10 completed purchases or the first
14 calendar days after launch, whichever comes first, then USD 29. Set the
actual UTC launch deadline immediately before publishing.

Buyers receive published v1.x bundle updates. The purchase does not include
hosting, provider accounts, a service-level commitment, personal support, or a
future major version.

Third-party provider, hosting, and marketplace terms, availability, quotas, and
charges can change. Buyers must use accounts they control and review the terms
that apply to them.

## Package release

From this directory, run:

```bash
bash package-release.sh
```

It writes a ZIP and SHA-256 file under `/tmp/free-llm-gateway-starter-kit/` by
default, includes only the listed bundle files, and verifies the archive. Set
`OUT_DIR` to use another local output directory.

Review the generated archive, complete the platform's tax, payout, and identity
steps, then upload it manually. Do not package the whole repository.

## Source licensing

Gateway source is separately licensed: the proxy application is MIT and the
rotator library is LGPL-3.0. This bundle contains setup materials only; source
use and redistribution are governed by the repository license notices.
