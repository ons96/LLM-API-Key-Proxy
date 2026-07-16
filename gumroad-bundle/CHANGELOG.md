# Changelog -- Free LLM Gateway Starter Kit

All notable changes to this Gumroad bundle are documented here. The underlying
gateway repo (github.com/ons96/LLM-API-Key-Proxy) has its own changelog; this
file tracks the bundle itself (the setup guide, templates, and scripts).

Format: [Keep a Changelog](https://keepachangelog.com/), semantic versioning.

## [1.0.0] - 2026-07-16

### Added
- `SETUP_GUIDE.md` -- the main deliverable. A 30-minute, step-by-step guide
  from Oracle Cloud signup to a working free LLM gateway to pointing coding
  tools (opencode, Cursor, continue.dev, LangChain) at it. Includes a
  troubleshooting section covering the common first-boot failures (port not
  open, 401, rate-limit fallback, OOM on 1GB, model name typos).
- `.env.starter` -- sanitized config template. Every env var a buyer needs,
  pre-filled with safe defaults and `# TODO` markers for free provider keys.
  No internal paths, no leaked secrets, no VPS-specific values.
- `quickstart.sh` -- one-click setup script. Installs uv + Python 3.12, creates
  a venv, installs deps, writes a systemd service (auto-restart, MemoryMax=400M
  for 1GB VPS safety), starts the gateway, runs a smoke test. Idempotent.
- `GUMROAD_LISTING.md` -- the Gumroad product page copy (title, tagline, two-tier
  pricing, FAQ, honest expectations).
- `README.md` -- this bundle's README (what the directory is, how to zip for
  Gumroad, file list).

### Pricing
- Standard tier: $29 (guide + templates + script + 30-day email support).
- Bonus tier: $49 (adds the auto-optimizing reorder service that re-ranks
  fallback chains from live telemetry every 30 minutes).

### Notes
- The underlying gateway repo is public and open source. The bundle's value is
  the curated setup path, not the code.
- All providers referenced (Groq, Gemini, Cerebras, OpenRouter, NVIDIA NIM,
  Mistral, iFlow, SambaNova) have free tiers as of 2026-07. Provider limits and
  availability can change; the guide's troubleshooting covers rate-limit
  fallback behavior.
