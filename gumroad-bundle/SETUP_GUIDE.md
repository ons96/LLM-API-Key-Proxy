# Free LLM Gateway Starter Kit -- Setup Guide

Version 1.0.0 | Updated 2026-07-16

This guide installs a self-hosted, OpenAI-compatible gateway from the public
source checkout. It does not provide hosting, provider accounts, credentials,
uptime, or a guaranteed zero-cost service. Third-party availability, eligibility,
quotas, and charges can change; review each provider and host before use.

## Security first

The installer binds the gateway to `127.0.0.1:8000` only. Do not expose TCP
port 8000 directly. Use an SSH tunnel for personal remote access. If you need
internet-facing access, put a separately configured HTTPS reverse proxy such as
Caddy on ports 80/443 in front of the loopback service. Keep `PROXY_API_KEY`
set even behind HTTPS.

Never share `.env`, provider credentials, or `PROXY_API_KEY`. Do not paste them
into support chats, issue trackers, screenshots, or public repositories.

## 1. Prepare a host

Use an Ubuntu or Debian host where you have a normal user account with `sudo`.
Choose a provider and plan yourself, review its current terms, and keep your SSH
key safe. From your computer, verify you can sign in:

```bash
ssh user@YOUR_SERVER
```

Apply the host's normal security updates before installing software.

## 2. Get the source and bundle

The gateway source is public. Clone it on the host:

```bash
git clone https://github.com/ons96/LLM-API-Key-Proxy.git
cd LLM-API-Key-Proxy
```

If the purchased bundle is newer than the copy in the checkout, replace only
the `gumroad-bundle/` directory with the downloaded one. Do not overwrite your
existing `.env`.

Gateway source licensing is separate from this bundle: the proxy application is
MIT and the rotator library is LGPL-3.0. Review the root license notices before
redistributing source.

## 3. Configure credentials

Create a private environment file:

```bash
cp gumroad-bundle/.env.starter .env
chmod 600 .env
openssl rand -hex 32
```

Paste the generated value into `PROXY_API_KEY` in `.env`. It protects your
gateway from unauthenticated requests. The installer refuses an empty,
placeholder, or short proxy key.

Add only credentials for provider accounts you created and control. Check
`config/router_config.yaml` for the corresponding credential names and current
routing policy. A setting in `.env` does not override router policy by itself.

Before continuing, confirm that `.env` is not tracked:

```bash
git status --short .env
```

It should produce no output. If it appears, stop and fix `.gitignore` before
continuing.

## 4. Install the loopback service

Run as your normal user; the script asks for `sudo` only when installing the
systemd unit:

```bash
bash gumroad-bundle/quickstart.sh
```

The script installs `uv` and Python if needed, creates a virtual environment,
installs requirements, creates `llm-gateway.service`, starts it, checks
`/health`, verifies the listener is loopback-only, and makes an authenticated
`/v1/models` request.

Inspect its state and logs:

```bash
sudo systemctl status llm-gateway
sudo journalctl -u llm-gateway -n 50 --no-pager
```

## 5. Test locally

On the server, use the proxy key from `.env`:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer YOUR_PROXY_API_KEY"
```

An empty model result can mean no usable provider credentials are configured.
Check the service logs and the router configuration before adding more keys.

## 6. Use an SSH tunnel first

From your own computer, leave this command running:

```bash
ssh -N -L 8000:127.0.0.1:8000 user@YOUR_SERVER
```

Your local client can now use:

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:8000/v1` |
| API key | Your `PROXY_API_KEY` |
| Model | A model returned by `/v1/models` |

For an OpenAI-compatible client, use the local base URL above and the proxy key.
This avoids exposing the gateway while you validate one client end to end.

## 7. Optional public HTTPS access

Do this only after local and SSH-tunnel testing work. You need a domain whose
DNS points to the server and firewall rules for HTTPS only. Do not open TCP/8000.

Install Caddy using its official installation instructions, then use a minimal
configuration like:

```caddy
gateway.example.com {
    reverse_proxy 127.0.0.1:8000
}
```

Open only ports 80 and 443 as required for the HTTPS proxy. Test the public
endpoint with `https://gateway.example.com/v1/models` and the proxy key. The
gateway process must still show `127.0.0.1:8000` as its listener.

## Troubleshooting

### The service fails to start

Read the recent journal, then validate the environment-file path, Python
dependencies, and port availability:

```bash
sudo journalctl -u llm-gateway -n 50 --no-pager
sudo ss -ltnp 'sport = :8000'
```

### I receive 401

Use exactly `Authorization: Bearer YOUR_PROXY_API_KEY`. Restart the service
after changing `.env`:

```bash
sudo systemctl restart llm-gateway
```

### I cannot connect from another device

That is expected before you create an SSH tunnel or HTTPS reverse proxy. The
installer intentionally keeps the gateway on loopback.

### A provider request fails or is rate limited

Check the journal and confirm the credential and router configuration for that
provider. Upstream failures, limits, and models vary; do not assume any fallback
will be available.

## Updating

Read release notes before updating a running gateway. Back up `.env` privately,
then update the source and dependencies:

```bash
cd ~/LLM-API-Key-Proxy
git pull
source venv/bin/activate
uv pip install -r requirements.txt
sudo systemctl restart llm-gateway
```

Re-run the local health and authenticated model-list checks after every update.

## Bundle scope

Buyers receive published v1.x updates to these setup materials. This purchase
does not include personal support, hosting, provider accounts, credentials,
service-level commitments, or future-major-version access. Use the checkout
platform's published refund process and policy.
