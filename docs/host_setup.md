## Host Setup (Non-Docker)

### Start the gateway
- From the repo root: `./scripts/run_host.sh`
- Script loads `.env`, sets `PYTHONPATH=src`, and runs `python src/proxy_app/main.py --host 0.0.0.0 --port 8000`.

### Zeroclaw baseURL
- Same machine apps: `http://127.0.0.1:8000/v1`
- External clients: `http://<VPS_PUBLIC_IP>:8000/v1`
- Auth header: `Authorization: Bearer $PROXY_API_KEY`

### Quick health checks
- `curl http://127.0.0.1:8000/health`
- `curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer $PROXY_API_KEY"`
