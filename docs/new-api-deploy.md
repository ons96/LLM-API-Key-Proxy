# new-api parallel deploy (task-board #862, EPIC #876)

Live FastAPI gateway (:8000) untouched. new-api runs alongside on :8001.

## Layout (VPS-40)

- Binary: `/opt/new-api/bin/new-api` (0755, root-owned) — pinned `v1.0.0-rc.36` amd64,
  sha256 `5d9b254721c205f4f442848a4af5b58617a8c0ee090366732cfff90455e957f3`
- Data: `/opt/new-api/data/one-api.db` (SQLite, owned `newapi`)
- Logs: `/opt/new-api/logs/` (owned `newapi`)
- Env: `/opt/new-api/new-api.env` (0600 root:root, `SESSION_SECRET` only)
- Unit: `/etc/systemd/system/new-api.service` (source: `deploy/new-api.service`)

## Verify

```
systemctl is-active new-api
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8001/
ps -o pid,rss -C new-api   # RSS must stay < 200MB idle
```

## First login

Default admin `root` / `123456` — change immediately in the console.
Covers hardening task #873; do not paste credentials anywhere.

## Rollback

```
sudo systemctl stop new-api && sudo systemctl disable new-api
```

Old gateway unaffected. No cutover until #874 suite green (EPIC #876).

## Deferred

Reboot-persistence test skipped (live gateway shares the host; `is-enabled`
verified instead). Re-run `systemctl is-active new-api` after next maintenance reboot.
