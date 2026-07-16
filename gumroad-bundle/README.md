# Free LLM Gateway Starter Kit (Gumroad Bundle)

This directory is the **product** sold on Gumroad. It is a self-contained
curation layer on top of the public, open-source gateway at
[github.com/ons96/LLM-API-Key-Proxy](https://github.com/ons96/LLM-API-Key-Proxy).

The gateway code is free and always will be. What this bundle sells is the
convenience of a 30-minute setup path, a sanitized config template, a one-click
deploy script, tool-integration examples, and email support.

## Files

| File | What it is |
|------|------------|
| `SETUP_GUIDE.md` | The main deliverable. Step-by-step guide from zero to a working free gateway. |
| `.env.starter` | Sanitized config template with `# TODO` markers for free provider keys. |
| `quickstart.sh` | One-click setup script for a fresh Ubuntu/Debian VPS. |
| `GUMROAD_LISTING.md` | Copy for the Gumroad product page (title, pricing, FAQ). |
| `CHANGELOG.md` | Version history of the bundle. |
| `README.md` | This file. |

## How to package for Gumroad

The bundle is designed to ship as a single zip. From the repo root:

```bash
zip -r gateway-starter-kit.zip gumroad-bundle/
```

Upload `gateway-starter-kit.zip` as the product file on Gumroad. Buyers unzip
it and follow `SETUP_GUIDE.md`. The guide also tells buyers they can clone the
public repo (which contains this `gumroad-bundle/` directory) if they prefer
git over a zip.

## Two tiers

- **Standard ($29):** guide + templates + script + 30-day email support.
- **Bonus ($49):** adds the auto-optimizing reorder service (systemd timer that
  re-ranks fallback chains from live telemetry every 30 minutes).

To ship the bonus tier, include the reorder service setup section of
`SETUP_GUIDE.md` (the "Enable the reorder service" subsection under
"Customizing") and the associated scripts. The standard tier omits that section.

## What this bundle does NOT include

- The gateway source code (it is in the public repo; buyers clone it).
- Any real API keys, internal VPS IPs, or private config (the starter template
  is fully sanitized).
- A hosting account (buyers create their own free Oracle Cloud account).

## Maintenance

When the underlying gateway repo changes in a way that affects setup (new
required env var, changed virtual model names, new systemd path), bump the
bundle version in `CHANGELOG.md` and update `SETUP_GUIDE.md` + `.env.starter`.
Re-zip and re-upload to Gumroad. Existing buyers get the update if you enable
Gumroad's "buyers get free updates" option.
