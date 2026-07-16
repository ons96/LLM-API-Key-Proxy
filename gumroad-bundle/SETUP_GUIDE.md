# Free LLM Gateway Starter Kit -- Setup Guide

> Version 1.0.0 | Updated 2026-07-16
> Time to complete: about 30 minutes. Everything you need is free.

---

## What you're building

You are about to set up your own **free OpenAI-compatible LLM gateway** on a
tiny cloud server. Instead of juggling nine different provider accounts and
hardcoding each one into your coding tools, you point your tools at **one
URL** and the gateway does the rest:

- It speaks the OpenAI API (`/v1/chat/completions`, `/v1/models`), so any
  tool that accepts a custom OpenAI base URL works immediately.
- It exposes a handful of **virtual models** -- `coding-elite`, `coding-fast`,
  `chat-smart`, `chat-fast` -- each backed by a **fallback chain** of free
  providers. If Groq is rate-limited, it silently tries Cerebras, then Gemini,
  and so on. You never see the failure.
- It runs on a 1 GB Oracle Cloud free-tier VM, costs $0/month, and stays up
  24/7 (no sleep, unlike Render's free tier).

The gateway code itself is [open source](https://github.com/ons96/LLM-API-Key-Proxy)
and always will be. What you paid for is this curated 30-minute path, the
pre-filled config template, the one-click setup script, and support. You are
buying the convenience, not the code.

---

## What you need (all free)

| Item | Where to get it | Cost | Time |
|------|-----------------|------|------|
| Oracle Cloud account | [cloud.oracle.com](https://cloud.oracle.com) | Free tier, credit card verify only | 5 min |
| Groq API key | [console.groq.com](https://console.groq.com/keys) | Free, generous limits | 2 min |
| Google Gemini key | [aistudio.google.com](https://aistudio.google.com/apikey) | Free tier | 2 min |
| Cerebras key (optional) | [cloud.cerebras.ai](https://cloud.cerebras.ai) | Free, 1M tokens/day | 2 min |
| NVIDIA NIM key (optional) | [build.nvidia.com](https://build.nvidia.com) | Free | 2 min |
| Your laptop's SSH key | `ssh-keygen` if you don't have one | Free | 1 min |

You can get a working gateway with **just Groq + Gemini** (two keys, ~4 min).
The others add more fallback headroom so a rate-limit on one provider never
reaches you. Cerebras is worth the two minutes -- it is extremely fast.

You do **not** need a credit card that gets charged. Oracle requires a card
to verify identity; the Always Free tier is never billed as long as you stay
within the free limits (which this gateway does easily on 1 GB RAM).

---

## Step 1: Get a free VPS (Oracle Cloud Always Free) -- ~10 min

We use Oracle Cloud because their "Always Free" tier includes a real
always-on VM with more RAM than other free tiers, and it does not sleep.

### 1.1 Sign up

1. Go to [cloud.oracle.com](https://cloud.oracle.com) and click **Sign up**.
2. Choose the **Home Region** closest to you (this cannot be changed later
   without opening a support ticket).
3. Enter a credit card for verification. **You will not be charged.** The
   Always Free resources are explicitly excluded from billing.
4. Wait for the account to provision (can take up to 15 minutes; you'll get
   an email).

### 1.2 Generate an SSH key pair

If you already have `~/.ssh/id_ed25519.pub` or `~/.ssh/id_rsa.pub`, skip to
1.3. Otherwise, on your laptop:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# Press Enter to accept the default path, set a passphrase or leave empty
```

You need the **public** key (the contents of `id_ed25519.pub`) in the next step.

### 1.3 Create the Always Free instance

1. In the Oracle Cloud console, open the hamburger menu (top left) and go to
   **Compute -> Instances**.
2. Click **Create instance**.
3. **Name**: `llm-gateway` (or anything you like).
4. **Image and shape**: Click *Edit* next to Image and Shape.
   - **Image**: Canonical **Ubuntu 22.04** (or 24.04). Click *Change image*,
     pick "Canonical Ubuntu 22.04" under "Free tier eligible", save.
   - **Shape**: Click *Change shape*. Under "Ampere" (ARM) select
     **VM.Standard.A1.Flex**. Set **1 OCPU** and **6 GB RAM** -- these stay
     free (the A1 flex shape is the generous one). If ARM shapes are
     unavailable in your region, use **VM.Standard.E2.1.Micro** (AMD, 1 GB)
     -- it also works, just tighter. Save.
5. **Add SSH keys**: choose **"Save private key" and "Save public key"** OR
   paste your existing public key from step 1.2 into the box. This is how you
   will log in. **Do not skip this** -- you cannot SSH in without it.
6. Click **Create**. The instance provisions in 1-3 minutes.

### 1.4 Find your public IP

On the instance's detail page, copy the **Public IP Address** (looks like
`129.150.x.x`). Save it -- you'll use it constantly.

### 1.5 Open port 8000 (security list)

The gateway listens on port 8000. Oracle blocks it by default. You need to
allow your own traffic.

1. On the instance detail page, scroll to **Resources -> Subnet** and click
   the subnet link.
2. Click the **Security List** (usually `Default Security List for ...`).
3. Click **Add Ingress Rules**:
   - **Source CIDR**: your home/office IP followed by `/32`, e.g.
     `203.0.113.42/32`. To find your current public IP, visit
     [ifconfig.me](https://ifconfig.me) from the same network.
     - For maximum security, lock it to your IP. If your IP changes often,
       use a VPN/Tailscale (see "Customizing" later). **Do not use
       `0.0.0.0/0`** unless you are sure -- it exposes your gateway to the
       whole internet.
   - **IP Protocol**: TCP
   - **Destination Port Range**: `8000`
4. Click **Add Ingress Rules**.

### 1.6 SSH in and update

```bash
ssh ubuntu@YOUR_PUBLIC_IP
# Accept the host key fingerprint on first connect
sudo apt update && sudo apt upgrade -y
```

If the `ubuntu` user doesn't work, try `opc`. Oracle Ubuntu images use
`ubuntu` by default.

You now have a server. Move on to getting your LLM keys.

---

## Step 2: Get free LLM provider keys -- ~5 min

Get at least **Groq** and **Gemini**. The others are bonus fallback layers.

| Provider | Signup URL | What to copy | Free limit (approx) | Notes |
|----------|-----------|--------------|---------------------|-------|
| Groq | [console.groq.com/keys](https://console.groq.com/keys) | `gsk_...` key | ~30 req/min, generous daily | Fastest free coding models. Start here. |
| Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | `AIza...` key | 15 req/min free tier | Google models, good quality |
| Cerebras | [cloud.cerebras.ai](https://cloud.cerebras.ai) | `csk-...` key | 1M tokens/day free | Insanely fast inference |
| NVIDIA NIM | [build.nvidia.com](https://build.nvidia.com) | `nvapi-...` key | 40 req/min, 1000 credits | Many open models |
| Mistral | [console.mistral.ai](https://console.mistral.ai) | key from API Keys page | Free tier with La Plateforme | Codestral good for code |
| OpenRouter | [openrouter.ai/keys](https://openrouter.ai/keys) | `sk-or-...` key | Free models, paid ones too | Aggregator; pick `:free` models |
| HuggingFace | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | `hf_...` token | Rate-limited free inference | Optional, queue-based |

**Tips:**
- On each site, sign up, find the "API Keys" or "Tokens" page, and click
  "Create new key". Copy the value immediately -- some sites hide it after.
- Store the keys in a text file on your laptop for now. We'll paste them into
  the config in Step 3.
- All of these have **free tiers that do not require a credit card** (except
  OpenRouter, which is free for the `:free` models and lets you add credits
  later if you want paid models -- entirely optional).

You do not need all of them. Two is enough to have automatic fallback. Five
gives you a robust setup that rarely shows you an error.

---

## Step 3: Deploy the gateway -- ~10 min

Run this on your VPS (you are SSH'd in from Step 1.6).

### 3.1 Clone the repo

```bash
cd ~
git clone https://github.com/ons96/LLM-API-Key-Proxy.git
cd LLM-API-Key-Proxy
```

> If you purchased this bundle, the `gumroad-bundle/` directory is part of
> the public repo. If you cloned without it (older checkout), you can
> instead clone the bundle separately or copy the files from your download.

### 3.2 Copy the starter config

```bash
cp gumroad-bundle/.env.starter .env
chmod 600 .env   # only your user can read it -- good hygiene
```

### 3.3 Fill in your free keys

Open `.env` in an editor (`nano .env` is easiest on a fresh VPS). Find every
line that says `# TODO: fill in your free key from <URL>` and paste your key.
At minimum, fill in:

```
GROQ_API_KEY_1="gsk_your_real_groq_key"
GEMINI_API_KEY_1="AIza_your_real_gemini_key"
```

Also set a **proxy password** -- this is the key your coding tools will use to
talk to your gateway. Make up any strong string:

```
PROXY_API_KEY="make-up-a-long-random-string-here"
```

Save and exit. Leave the optional providers (`CEREBRAS_API_KEY_1`, etc.) blank
for now; you can add them later.

### 3.4 Run the one-click setup script

```bash
bash gumroad-bundle/quickstart.sh
```

The script does the rest:

1. Installs **uv** (a fast Python package manager) and Python 3.12 if needed.
2. Creates a virtualenv and installs the gateway dependencies.
3. Writes a **systemd service** so the gateway starts on boot and restarts on
   crash, capped at **400 MB RAM** so your 1 GB VPS never runs out.
4. Starts the service.
5. Runs a smoke test (`curl /v1/models`).
6. Prints your gateway URL and next steps.

Watch the output. When it finishes you'll see something like:

```
============================================================
  Gateway is live: http://129.150.x.x:8000/v1
  Proxy password: the PROXY_API_KEY you set in .env
  Next: point your coding tools here (see SETUP_GUIDE.md Step 5)
============================================================
```

If the smoke test failed, jump to **Troubleshooting** below -- the common
causes are listed and each has a one-line fix.

---

## Step 4: Test it -- ~2 min

Still on the VPS (or from your laptop if port 8000 is open to your IP), run:

### 4.1 List the virtual models

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer YOUR_PROXY_API_KEY"
```

You should get JSON with the virtual models -- `coding-elite`, `coding-fast`,
`chat-smart`, `chat-fast`, and more -- alongside the raw provider models.

### 4.2 Send a chat completion through a virtual model

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_PROXY_API_KEY" \
  -d '{
    "model": "coding-elite",
    "messages": [{"role": "user", "content": "Write a Python function that reverses a string."}],
    "max_tokens": 200
  }'
```

You should get a normal OpenAI-style JSON response with the assistant's
answer. If you do, the fallback chain worked -- you didn't have to know or
care which provider served it.

### 4.3 Check health

```bash
curl http://localhost:8000/health
```

Returns a small JSON status. If the gateway is running, this always responds.

### 4.4 Watch it fall back (optional, but satisfying)

To see fallback in action, temporarily set a garbage Groq key in `.env`
(`GROQ_API_KEY_1="invalid"`), restart the service
(`sudo systemctl restart llm-gateway`), and re-run the chat completion. It
will still succeed -- just served by the next provider in the chain. Restore
your real key afterward.

---

## Step 5: Point your tools at it -- ~3 min

Your gateway is an OpenAI-compatible endpoint. Any tool that lets you set a
custom OpenAI base URL works. Use these values:

| Setting | Value |
|---------|-------|
| Base URL | `http://YOUR_VPS_IP:8000/v1` |
| API key | the `PROXY_API_KEY` you set in `.env` |
| Model | `coding-elite` (or `coding-fast`, `chat-smart`, `chat-fast`) |

### OpenCode

In your OpenCode config (`~/.config/opencode/opencode.json` or project
`.opencode/opencode.json`), add a provider:

```json
{
  "provider": {
    "my-gateway": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "My Free Gateway",
      "options": {
        "baseURL": "http://YOUR_VPS_IP:8000/v1",
        "apiKey": "YOUR_PROXY_API_KEY"
      },
      "models": {
        "coding-elite": { "name": "coding-elite" },
        "coding-fast": { "name": "coding-fast" },
        "chat-smart": { "name": "chat-smart" },
        "chat-fast": { "name": "chat-fast" }
      }
    }
  }
}
```

Then pick `coding-elite` from the model dropdown.

### Cursor

1. Open **Settings -> Models**.
2. Add a custom OpenAI-compatible provider:
   - **Base URL**: `http://YOUR_VPS_IP:8000/v1`
   - **API Key**: `YOUR_PROXY_API_KEY`
   - **Model**: `coding-elite`
3. Verify the connection.

### continue.dev (VS Code)

Edit `~/.continue/config.json`:

```json
{
  "models": [{
    "title": "My Free Gateway",
    "provider": "openai",
    "model": "coding-elite",
    "apiBase": "http://YOUR_VPS_IP:8000/v1",
    "apiKey": "YOUR_PROXY_API_KEY"
  }]
}
```

### LangChain (Python)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://YOUR_VPS_IP:8000/v1",
    api_key="YOUR_PROXY_API_KEY",
    model="coding-elite",
)
print(llm.invoke("Write a Python function that reverses a string.").content)
```

### Anything else that speaks OpenAI

Set `OPENAI_BASE_URL=http://YOUR_VPS_IP:8000/v1` and
`OPENAI_API_KEY=YOUR_PROXY_API_KEY` as environment variables. Most tools pick
those up automatically.

> **About HTTP vs HTTPS**: This guide uses plain `http://`. For local-network
> or personal use that's fine. If you expose the gateway to the internet,
> put it behind a reverse proxy (Caddy/Nginx) with TLS. See "Customizing".

---

## Troubleshooting

### "Connection refused" / can't reach port 8000

- Did you open port 8000 in the Oracle **security list** (Step 1.5)? That's
  the #1 cause. The VM firewall and the Oracle security list are separate --
  you need the security list rule.
- Is the service running? `sudo systemctl status llm-gateway`. If it says
  `failed`, check logs: `sudo journalctl -u llm-gateway -n 50 --no-pager`.
- Are you testing from an IP you allowed? If you locked the security list to
  your home IP and you're on a different network, it will time out.

### "401 Unauthorized" from the gateway

- The `PROXY_API_KEY` in your `.env` must match the `Authorization: Bearer`
  value your client sends. Re-check for trailing quotes or whitespace.
- If you set `PROXY_API_KEY` to empty in `.env.starter`, the gateway runs
  **unauthenticated**. That's fine for testing but don't leave it open to the
  internet.

### Provider returns "invalid api key" / 401 from upstream

- That specific provider's key is wrong or expired. The gateway **falls back
  to the next provider automatically**, so your request still succeeds -- but
  you'll see warnings in the logs. Fix the bad key in `.env` and restart:
  `sudo systemctl restart llm-gateway`.

### Provider rate-limited (429)

- This is exactly what the fallback chain is for. You usually won't notice.
- If you hit it often, add more providers (Cerebras, NVIDIA) so the load
  spreads. Each free key adds headroom.

### Gateway crashes / OOM on 1 GB RAM

- The systemd unit is capped at `MemoryMax=400M`. If the VPS only has 1 GB,
  that's the right setting -- do not raise it.
- If you picked the ARM shape with 6 GB, you can raise it to `800M` or remove
  the cap.
- Symptom: `sudo journalctl -u llm-gateway` shows `Killed` or
  `Out of memory`. Fix: `sudo systemctl daemon-reload && sudo systemctl
  restart llm-gateway`. Ensure no other memory-hungry service is running.

### Models list is empty or only shows a few

- You have no working provider keys. Confirm Groq/Gemini keys are valid. Run
  `curl http://localhost:8000/v1/models` again after fixing.
- A provider may be marked disabled in `config/router_config.yaml`. The
  starter `.env` enables the free ones; check the file if you customized it.

### "Model not found" when calling a virtual model

- Use the exact virtual model name: `coding-elite`, not `coding_elite` or
  `CodingElite`. Check `/v1/models` for the exact strings.
- The chain may have zero available providers if all your keys are invalid.
  See the 401 note above.

### Changes to `.env` don't take effect

- The systemd service reads `.env` at start. After editing: `sudo systemctl
  restart llm-gateway`.

---

## What you get

### Virtual models (the main interface)

These are the names you put in your tools. Each is backed by a fallback chain
that tries free providers in priority order.

| Virtual model | Best for | Behavior |
|---------------|----------|----------|
| `coding-elite` | Agentic coding, complex generation | Highest-quality free coding models first, then broader fallbacks |
| `coding-fast` | Quick edits, completions | Speed-prioritized chain (low latency) |
| `chat-smart` | Reasoning, analysis, research | Intelligence-weighted chain |
| `chat-fast` | Quick Q&A, low-latency chat | TPS-ranked chain |
| `auto` | Let the gateway decide | Semantic intent routing to the right chain |

The full list with current chains lives in `config/virtual_models.yaml`.
Chains are long (dozens of candidates deep) so a single rate-limit never
reaches you.

### Automatic fallback

When a request to priority-1 provider fails (rate limit, timeout, error), the
router immediately tries priority-2, then 3, and so on. If a provider is on
cooldown from a recent 429, it is skipped entirely until its cooldown
expires. You see one successful response; the retries are invisible.

### Telemetry

The gateway logs every request to an SQLite database
(`/dev/shm/telemetry.db` -- in-memory, wiped on reboot by default). Tracked:
time-to-first-token, tokens-per-second, success/failure per provider, token
counts, cost estimates. Use it to see which providers actually serve you:

```bash
sqlite3 /dev/shm/telemetry.db \
  "SELECT provider, COUNT(*) as calls, ROUND(AVG(tps),1) as avg_tps
   FROM llm_events WHERE success=1 GROUP BY provider ORDER BY calls DESC LIMIT 10;"
```

### The reorder service (bonus tier)

If you bought the bundle with the reorder-chains bonus, you also get a
systemd timer that runs every 30 minutes and **re-ranks your fallback chains
based on live speed and quality data** from telemetry. Over time, your
`coding-elite` chain self-optimizes toward whichever free providers are
actually fastest for you right now. Setup is in `SETUP_GUIDE.md` ->
"Customizing -> Enable the reorder service". If you have the standard tier,
you can enable it manually using the scripts in `scripts/` -- see that
section.

---

## Customizing

### Add or remove a provider

1. Get the provider's free key (Step 2 table).
2. Add it to `.env`, e.g. `CEREBRAS_API_KEY_1="csk_..."`.
3. Confirm the provider is `enabled: true` in `config/router_config.yaml`.
   The free providers are enabled by default in the starter config.
4. `sudo systemctl restart llm-gateway`.

To remove a provider, set `enabled: false` in `router_config.yaml` (or just
blank its key in `.env`).

### Tune a fallback chain

Open `config/virtual_models.yaml`. Each virtual model has a `fallback_chain`
list ordered by `priority` (1 = tried first). Reorder, add, or remove
entries. Restart the service. This is a normal YAML edit -- no code changes.

> Note: if you enable the reorder service (below), it rewrites this file
> every 30 min from telemetry data. Hand-edits will be overwritten. To keep
> manual control, either don't enable the timer, or edit and then stop the
> timer (`sudo systemctl stop reorder-chains.timer`).

### Change the virtual model definitions

Same file: `config/virtual_models.yaml`. You can add a new virtual model by
copying an existing block and changing the name and chain. It will appear in
`/v1/models` automatically after restart.

### Enable the reorder service (auto-optimize chains)

The reorder service reads your telemetry and rewrites the fallback chains to
favor your fastest, most-reliable providers. To enable:

```bash
cd ~/LLM-API-Key-Proxy
sudo cp scripts/reorder-chains.service /etc/systemd/system/
sudo cp scripts/reorder-chains.timer /etc/systemd/system/
# Edit the .service file: replace /home/ubuntu paths with your install path
# if different, and set LLM_BENCHMARK_DB / LLM_PROVIDERS_DB if you have them
# (optional -- without them it falls back to telemetry-only ranking).
sudo systemctl daemon-reload
sudo systemctl enable --now reorder-chains.timer
```

It runs every 30 minutes. Check status with
`systemctl status reorder-chains.service`.

### Secure it with HTTPS

For internet-facing use, put the gateway behind Caddy (automatic Let's
Encrypt):

```bash
sudo apt install -y caddy
# Edit /etc/caddy/caddyfile:
#   your-domain.com {
#     reverse_proxy 127.0.0.1:8000
#   }
sudo systemctl restart caddy
```

Point your domain's A record at the VPS IP first. Now your base URL becomes
`https://your-domain.com/v1`.

### Lock access down with Tailscale (recommended)

Instead of opening port 8000 to a specific IP (which breaks when your IP
changes), install [Tailscale](https://tailscale.com) on the VPS and your
devices. The gateway then only needs to listen on the Tailscale interface,
and you connect via a stable private IP. Free for personal use.

### Back up your config

Your `.env` and any custom `config/*.yaml` edits are the only things unique
to you. Back them up somewhere off the VPS:

```bash
# On your laptop
scp ubuntu@YOUR_VPS_IP:~/LLM-API-Key-Proxy/.env ./gateway-env-backup
```

---

## Updating the gateway

The repo gets improvements over time. To update:

```bash
cd ~/LLM-API-Key-Proxy
git pull
source venv/bin/activate
uv pip install -r requirements.txt   # or: pip install -r requirements.txt
sudo systemctl restart llm-gateway
```

Your `.env` and config edits are preserved (they're not tracked by git). If a
config schema changed, check `CHANGELOG.md` in this bundle for migration
notes.

---

## Support

Purchased the bundle with support? Email the address in your Gumroad receipt
with:
- What you expected, what happened, and the exact error.
- Output of `sudo journalctl -u llm-gateway -n 50 --no-pager`.
- Your `.env` with the key values redacted (replace each with `XXXX`).

We respond within 2 business days, 30 days from purchase.

---

## Honest expectations

- **Free tiers have rate limits.** Groq ~30 req/min, Gemini ~15 req/min,
  Cerebras 1M tokens/day. The gateway's job is to spread load across them so
  you rarely hit a wall -- but it is not "unlimited free LLMs forever."
  Heavy agentic-coding sessions may occasionally wait for a cooldown.
- **Quality varies by provider.** `coding-elite` prefers the best free coding
  models available; it is not a paid Claude/GPT-tier experience, though some
  free models (GLM-5.2, Gemini 3 Pro, Qwen3) are genuinely strong.
- **The VPS is yours to maintain.** Oracle occasionally restarts instances;
  systemd brings the gateway back automatically. Apply OS security updates
  now and then (`sudo apt update && sudo apt upgrade -y`).
- **You can add paid providers later.** Drop an OpenAI or Anthropic key in
  `.env` and they join the fallback chains. The free-first ordering keeps
  your bill at zero unless you explicitly change it.

You now have a personal, free, always-on LLM gateway. Enjoy.
