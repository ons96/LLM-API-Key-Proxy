# Quick Start: Auto-Discovery System

Get the auto-discovery system running in 5 minutes.

## Prerequisites

- GitHub repository (for Actions)
- VPS with gateway running
- Basic familiarity with cron

## Step 1: Configure GitHub Secrets

Go to your repository → Settings → Secrets → Actions → New repository secret

Add:
- `VPS_TELEMETRY_URL`: (optional) URL where telemetry JSON is hosted
- `VPS_SSH_KEY`: (optional) SSH private key for VPS
- `VPS_HOST`: (optional) Your VPS IP/domain

## Step 2: Install VPS Sync Daemon

```bash
# On your local machine
scp scripts/vps_sync_daemon.sh ubuntu@YOUR_VPS_IP:~/

# SSH into VPS
ssh ubuntu@YOUR_VPS_IP

# Make executable
chmod +x ~/vps_sync_daemon.sh

# Add to crontab
crontab -e
# Add this line:
*/5 * * * * /home/ubuntu/vps_sync_daemon.sh >> /home/ubuntu/sync_daemon.log 2>&1
```

## Step 3: Set Up Telemetry Export (on VPS)

```bash
# SSH into VPS
ssh ubuntu@YOUR_VPS_IP

# Add telemetry export cron
crontab -e
# Add this line (runs daily at 01:50 UTC, before GitHub Actions):
50 1 * * * cd ~/LLM-API-Key-Proxy && python scripts/export_telemetry_summary.py >> ~/telemetry_export.log 2>&1
```

## Step 4: Update OpenCode Config (local)

```bash
# On your local machine
cd ~/LLM-API-Key-Proxy
python scripts/update_opencode_config.py
```

## Step 5: Verify

### Check GitHub Actions

1. Go to Actions tab in GitHub
2. Run workflow manually (workflow_dispatch)
3. Check logs for success

### Check VPS

```bash
# SSH into VPS
ssh ubuntu@YOUR_VPS_IP

# Check if daemon is running
tail -f ~/sync_daemon.log

# Check telemetry export
cat ~/LLM-API-Key-Proxy/config/telemetry_summary.json | head -20
```

### Check Gateway

```bash
# Test API
curl http://localhost:8000/v1/models | jq '.data | length'
```

## Daily Operation

The system runs automatically:
1. **01:50 UTC**: VPS exports telemetry to JSON
2. **02:00 UTC**: GitHub Actions runs discovery pipeline
3. **02:05 UTC**: Configs committed to repo
4. **02:10 UTC**: VPS daemon detects changes, reloads gateway

## Manual Trigger

Force an update anytime:

```bash
# Via GitHub UI
Actions → auto-discovery.yml → Run workflow

# Or via VPS
ssh ubuntu@YOUR_VPS_IP
cd ~/LLM-API-Key-Proxy
git pull
pkill -f 'main.py' -HUP
```

## Next Steps

- Read full docs: `docs/AUTO_DISCOVERY_SYSTEM.md`
- Customize agent profiles: Edit `scripts/run_discovery_pipeline.py`
- Monitor: Check GitHub Actions logs weekly

## Troubleshooting

**GitHub Actions fails:** Check secrets are correct

**VPS doesn't pull changes:** Verify crontab entry

**Gateway doesn't reload:** Run `pkill -f 'main.py' -HUP` manually

Need help? Check logs at:
- GitHub: Actions → workflow → logs
- VPS: `~/sync_daemon.log` and `~/llm_proxy.log`
