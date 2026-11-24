# Permanent HTTPS Setup Status

**Last Updated:** 2026-02-07

## 🟢 Current Status: Working (Temporary)

Your gateway is currently accessible via **ngrok**. This URL is valid but will change if the VPS reboots.

- **Active URL:** `https://mahalia-pneumatophorous-uriel.ngrok-free.dev/v1`
- **Kobold Lite:** Use this URL directly.

---

## 🟡 Pending Actions (For Permanent URL)

To get a **permanent** URL that never changes (even after reboot), you must complete **ONE** of the following free options. No credit card required for either.

### Option A: Tailscale Funnel (Recommended)
1.  **Click this link:** [Authorize Tailscale](https://login.tailscale.com/a/1357f7d8012d57)
2.  **Log in** with Google/GitHub/Microsoft.
3.  **Done.** The system will automatically detect the login and switch to the permanent URL.

### Option B: zrok (Reserved Share)
1.  **Sign up:** [zrok.io](https://zrok.io) (Email only)
2.  **Check email:** Click the invite link to set a password.
3.  **Get Token:** Copy the `zrok enable <token>` command from your dashboard.
4.  **Paste it** in the OpenCode chat.

---

## Troubleshooting

If the temporary URL stops working:
1.  Open OpenCode.
2.  Run: `./scripts/check_https_status.sh`
3.  It will show you the new active URL.
