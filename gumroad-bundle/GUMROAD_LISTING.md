# GUMROAD LISTING - Free LLM Gateway Starter Kit

> This is the copy for the Gumroad product page. Paste the sections below into
> the Gumroad product editor. Keep the tone honest and practical.

---

## Product title

Free LLM Gateway Starter Kit -- 9 Providers, 30-Minute Setup

## Tagline

Your own always-on, OpenAI-compatible gateway that aggregates free LLM
providers with automatic fallback. Zero monthly cost.

## Price

**$29** -- Standard tier (setup guide + templates + one-click script + 30-day email support)

**$49** -- Bonus tier (everything above + the auto-optimizing reorder service
that re-ranks your fallback chains from live speed data every 30 minutes)

### Why two tiers
The standard tier gets you a working free gateway in 30 minutes. The bonus tier
adds a self-optimizing layer: a systemd timer that reads your gateway's
telemetry and rewrites the fallback chains to favor whichever free providers are
actually fastest for you right now. Over a few days, your `coding-elite` chain
quietly tunes itself. If you are comfortable on the command line and want the
gateway to get better on its own, get the bonus tier. If you just want it
working and forget about it, the standard tier is enough.

## What you get

- **The 30-minute setup guide** (`SETUP_GUIDE.md`) -- step-by-step, from Oracle
  Cloud signup to a working gateway to pointing your coding tools at it. No
  assumed knowledge beyond basic terminal use.
- **Sanitized config template** (`.env.starter`) -- every env var you need,
  pre-filled with safe defaults and clear `# TODO` markers for your free keys.
  No internal paths, no leaked secrets, no guesswork.
- **One-click setup script** (`quickstart.sh`) -- run it on your VPS; it installs
  deps, writes the systemd service (auto-restart on crash, memory-capped for 1GB
  VPS), starts the gateway, and runs a smoke test. Idempotent -- safe to re-run.
- **Virtual model config** -- the four virtual models (`coding-elite`,
  `coding-fast`, `chat-smart`, `chat-fast`) backed by deep fallback chains, ready
  to use the moment the gateway boots.
- **Tool integration examples** -- working config snippets for opencode, Cursor,
  continue.dev, and LangChain so you can point your tools at the gateway in
  under a minute.
- **30 days of email support** -- stuck on a step? Email with your redacted
  `.env` and log output; response within 2 business days.
- **Bonus tier only: the reorder service** -- systemd unit + timer that
  re-optimizes your fallback chains from live telemetry every 30 minutes.

## Who it's for

- Developers who want free LLMs for coding and chat without stitching nine
  provider accounts into their tools by hand.
- opencode, Cursor, and continue.dev users who want a single custom OpenAI
  endpoint instead of juggling provider configs.
- Hobbyists and indie hackers who want an always-on LLM endpoint for side
  projects without a monthly bill.
- Anyone on a tight budget who has heard "just use free LLM providers" but
  bounced off the setup complexity.

## What it saves you

The underlying gateway is open source and free. What you are paying for is the
hours this bundle removes:

- Reading a large repo to figure out which env vars matter and which are
  internal noise.
- Signing up for nine providers and not knowing which two are enough to start.
- Debugging the first boot (port not open, wrong key format, OOM on 1GB,
  virtual model name typo) -- the troubleshooting section covers each.
- Writing a systemd unit that survives reboots and does not OOM your VPS.
- Figuring out how to point opencode/Cursor/continue.dev at a custom endpoint.

If your time is worth more than about $6/hour to you, the standard tier pays for
itself in the first afternoon.

## Requirements

- An Oracle Cloud account (Always Free tier, credit card for identity verify
  only -- never billed for free-tier resources).
- 2 to 5 free LLM provider API keys (Groq + Gemini minimum; all free, links in
  the guide). About 10 minutes of signup.
- Basic terminal comfort (SSH in, edit a file, run a script). No Python or
  networking expertise needed.

## FAQ

**Is the gateway code free?**
Yes. It is open source at github.com/ons96/LLM-API-Key-Proxy and always will be.

**Then what am I paying for?**
The curated 30-minute setup path, the sanitized config template, the one-click
script, the tool-integration examples, and 30 days of email support. You are
buying convenience and curation, not the code -- like a book that explains free
software.

**Will it cost me anything ongoing?**
No. All providers have free tiers. The gateway is free-first by default
(`FREE_ONLY_MODE=true`), so even if you add a paid key later it will not use it
unless you explicitly turn that off.

**What if a provider rate-limits me?**
The gateway automatically falls back to the next provider in the chain. You
usually never see the failure. Adding more free keys spreads the load.

**Can I add paid providers later?**
Yes. Drop an OpenAI or Anthropic key in `.env`, set `FREE_ONLY_MODE=false`, and
they join the fallback chains. The free-first ordering keeps your bill at zero
unless you change it.

**Does it work on a 1GB VPS?**
Yes. The systemd unit is memory-capped at 400MB and the gateway is designed for
low-RAM hardware. If you pick the 6GB ARM free shape, you can raise the cap.

**Is my gateway exposed to the internet?**
Only if you choose to open the port that wide. The guide shows you how to lock
access to your IP or use Tailscale (free, private VPN) so only your devices can
reach it.

**What if I get stuck?**
Email the address in your Gumroad receipt with what you expected, what happened,
the error, and your `.env` with key values replaced by `XXXX`. Response within 2
business days, 30 days from purchase.

**Is there a refund?**
If you cannot get the gateway running after contacting support, yes -- full
refund within 30 days. We want this to actually work for you.

## Sample (first page of the setup guide)

```
What you're building
--------------------
You are about to set up your own free OpenAI-compatible LLM gateway on a
tiny cloud server. Instead of juggling nine different provider accounts and
hardcoding each one into your coding tools, you point your tools at one URL
and the gateway does the rest:

- It speaks the OpenAI API (/v1/chat/completions, /v1/models), so any tool
  that accepts a custom OpenAI base URL works immediately.
- It exposes a handful of virtual models -- coding-elite, coding-fast,
  chat-smart, chat-fast -- each backed by a fallback chain of free
  providers. If Groq is rate-limited, it silently tries Cerebras, then
  Gemini, and so on. You never see the failure.
- It runs on a 1 GB Oracle Cloud free-tier VM, costs $0/month, and stays
  up 24/7 (no sleep, unlike Render's free tier).

The gateway code itself is open source and always will be. What you paid for
is this curated 30-minute path, the pre-filled config template, the
one-click setup script, and support.
```

## Honest expectations (include on the listing page)

- Free tiers have rate limits. The gateway spreads load across providers so you
  rarely hit a wall, but it is not "unlimited free LLMs forever." Heavy agentic
  sessions may occasionally wait for a cooldown.
- Quality varies by provider. `coding-elite` prefers the best free coding models
  available; some (GLM-5.2, Gemini 3 Pro, Qwen3) are genuinely strong, but it is
  not a paid Claude/GPT-tier experience.
- The VPS is yours to maintain. Oracle occasionally restarts instances; systemd
  brings the gateway back automatically. Apply OS updates now and then.

No false scarcity. No "only 50 left." No countdown timer. This is a digital
product that will be here when you are ready.
