# Product listing draft -- Free LLM Gateway Starter Kit

This is draft copy. Complete marketplace identity, payout, tax, refund, and
launch-date fields manually before publishing.

## Title

Free LLM Gateway Starter Kit -- Secure Self-Hosted Setup

## Tagline

Setup materials for running a loopback-only, OpenAI-compatible gateway from a
source checkout you control.

## Price

One one-time product. Charge USD 19 for the first 10 completed purchases or
the first 14 calendar days after launch, whichever comes first, then USD 29.
Immediately before publishing, add the exact UTC launch deadline to the product
page. Keep the same price across checkout platforms; taxes and currency
conversion can vary at checkout.

## What buyers receive

- A security-first installation guide for Ubuntu/Debian.
- A sanitized `.env` template with no real credentials.
- An idempotent systemd installer that binds the gateway to `127.0.0.1` only.
- SSH-tunnel and optional HTTPS reverse-proxy guidance.
- Local health and authenticated model-list smoke tests.
- An explicit release script that packages only these buyer-facing files.

The source repository is public at
`https://github.com/ons96/LLM-API-Key-Proxy`. This product sells curated setup
materials, not exclusive source access. Gateway source licensing remains
separate: the proxy application is MIT and the rotator library is LGPL-3.0.

## Requirements

- A source checkout of the gateway and an Ubuntu/Debian host where you have
  normal-user sudo access.
- Provider credentials and hosting accounts controlled by the buyer.
- Basic command-line comfort: SSH, editing a file, and running a script.
- A personal SSH tunnel or, for public use, a domain and separately configured
  HTTPS reverse proxy.

Provider, hosting, marketplace, quota, eligibility, and pricing terms can
change. Review current third-party terms before creating an account or adding a
credential. This bundle does not promise a particular provider, model, rate
limit, cost, uptime, or result.

## Security model

The installer keeps the gateway on `127.0.0.1:8000`. It does not open a
firewall port or print a public HTTP endpoint. For remote personal use, tunnel
the port over SSH. For internet-facing use, configure HTTPS on ports 80/443 and
reverse-proxy only to the loopback gateway. Keep `PROXY_API_KEY` enabled in all
cases. Never upload or send `.env` files, provider keys, or proxy keys.

## Product scope

Buyers receive published v1.x updates to these bundle materials. Purchase does
not include hosting, provider accounts, API keys, a service-level commitment,
personal support, or a future major version. Configure any refund policy in the
checkout platform before launch; do not condition refunds on private support.

## FAQ

**Does this include hosted API access?** No. It is a self-hosted setup bundle.

**Can I expose port 8000 directly?** No. Keep it loopback-only; use SSH or a
properly configured HTTPS reverse proxy.

**Will every configured provider work?** No guarantee. Credentials, models,
quotas, and upstream availability are external and can change.

**What do I test first?** Run the local `/health` and authenticated `/v1/models`
checks, then point one client through an SSH tunnel before considering HTTPS.

## Honest expectations

This bundle removes setup ambiguity; it does not remove third-party account
requirements or operational responsibility. Start with one host and a small
number of credentials, verify one client call, and expand only after that works.
