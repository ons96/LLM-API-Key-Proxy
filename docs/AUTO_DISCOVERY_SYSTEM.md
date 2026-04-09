# Auto-Discovery & Optimization System

Automatically discovers free LLM models, optimizes fallback chains based on telemetry, and generates agent-specific virtual models.

## Architecture

```
GitHub Actions (Daily 02:00 UTC)
    ↓
Fetch models.dev + Provider APIs
    ↓
Pull VPS Telemetry
    ↓
Compute Composite Scores
    ↓
Generate Agent Virtual Models
    ↓
Commit to Config Files
    ↓
VPS Detects Changes → Graceful Reload
```

## Components

### 1. GitHub Actions Workflow (`.github/workflows/auto-discovery.yml`)

Runs daily to:
- Fetch models.dev free models
- Pull VPS telemetry
- Run discovery pipeline
- Update configs and commit

**Setup:**
1. Add GitHub Secrets:
   - `VPS_TELEMETRY_URL`: URL to fetch telemetry JSON
   - `VPS_SSH_KEY`: (optional) SSH key for direct VPS access
   - `VPS_HOST`: (optional) VPS hostname

2. Enable GitHub Actions on your repository

### 2. Discovery Pipeline (`scripts/run_discovery_pipeline.py`)

Orchestrator that:
- Fetches models.dev data
- Analyzes telemetry
- Computes composite scores
- Generates agent-specific virtual models
- Updates providers_database.yaml

**Usage:**
```bash
python scripts/run_discovery_pipeline.py \
  --telemetry telemetry_summary.json \
  --output-dir config/
```

### 3. Telemetry Analyzer (`src/optimization/telemetry_analyzer.py`)

Analyzes VPS telemetry to compute:
- Success rates per provider/model
- Average latency
- TPS metrics
- Composite performance scores

**Scoring Formula:**
```
Score = Benchmark(40%) + SuccessRate(40%) + Latency(20%)
```

### 4. VPS Sync Daemon (`scripts/vps_sync_daemon.sh`)

Lightweight daemon that:
- Runs every 5 minutes via cron
- Detects config changes
- Gracefully reloads gateway (no restart)

**Install on VPS:**
```bash
# Copy script to VPS
scp scripts/vps_sync_daemon.sh ubuntu@40.233.101.233:~/

# Add to crontab
crontab -e
# Add this line:
*/5 * * * * /home/ubuntu/vps_sync_daemon.sh
```

### 5. OpenCode Config Updater (`scripts/update_opencode_config.py`)

Updates OpenCode config with agent-specific models:
- `oracle` → `agent-oracle`
- `explore` → `agent-explore`
- `librarian` → `agent-librarian`
- `build` → `agent-build`
- `metis` → `agent-metis`
- `momus` → `agent-momus`

**Run locally:**
```bash
python scripts/update_opencode_config.py
```

### 6. Telemetry Exporter (`scripts/export_telemetry_summary.py`)

Exports VPS telemetry to JSON for GitHub Actions.

**Run on VPS:**
```bash
python scripts/export_telemetry_summary.py
# Creates: config/telemetry_summary.json
```

## Agent Profiles

Each agent gets optimized fallback chains based on task requirements:

| Agent | Needs | Weights |
|-------|-------|---------|
| oracle | reasoning, tools, long_context | benchmark(60%), success(30%), latency(10%) |
| explore | speed, tools | benchmark(20%), success(30%), latency(50%) |
| librarian | tools, search | benchmark(30%), success(50%), latency(20%) |
| build | coding, tools | benchmark(40%), success(40%), latency(20%) |
| metis | reasoning, analysis | benchmark(70%), success(20%), latency(10%) |
| momus | reasoning, analysis | benchmark(65%), success(25%), latency(10%) |

## Deployment

### Option A: GitHub Actions + VPS Cron (Recommended)

**Pros:**
- Free GitHub Actions minutes
- No VPS resource usage for computation
- Works around 60-day GitHub runner limit

**Setup:**
1. Configure GitHub Secrets
2. Enable GitHub Actions
3. Install VPS sync daemon (5-min cron)
4. Set up telemetry exporter on VPS

### Option B: VPS-Only (Alternative)

If GitHub Actions unavailable, run everything on VPS:

```bash
# Add to VPS crontab
0 2 * * * cd ~/LLM-API-Key-Proxy && python scripts/run_discovery_pipeline.py --telemetry /tmp/llm_proxy_telemetry.db --output-dir config/
```

## Configuration

### Environment Variables

- `VPS_TELEMETRY_URL`: URL to fetch telemetry JSON
- `VPS_SSH_KEY`: SSH private key for VPS access
- `VPS_HOST`: VPS hostname or IP

### Customization

Edit `scripts/run_discovery_pipeline.py` to adjust:

1. **Agent Profiles** (lines 45-75):
```python
AGENT_PROFILES = {
    "oracle": {
        "needs": ["reasoning", "tools"],
        "weights": {"benchmark": 0.6, "success_rate": 0.3, "latency": 0.1}
    }
}
```

2. **Scoring Weights** (lines 135-140):
```python
bench_score = bench["swe_bench"] * 0.5 + bench["coding"] * 0.3
composite = bench_score * 0.4 + success_score * 0.4 + latency_score * 0.2
```

3. **Fallback Chain Length** (line 160):
```python
for i, c in enumerate(candidates[:20])  # Top 20 models
```

## Testing

### Manual Test

```bash
# Create test telemetry
echo '{"providers": {}}' > telemetry_test.json

# Run pipeline
python scripts/run_discovery_pipeline.py \
  --telemetry telemetry_test.json \
  --output-dir config/

# Check results
ls -la config/discovery_report.json
ls -la config/optimization_report.json
```

### VPS Test

```bash
# SSH into VPS
ssh ubuntu@40.233.101.233

# Run telemetry export
cd ~/LLM-API-Key-Proxy
python scripts/export_telemetry_summary.py

# Check output
cat config/telemetry_summary.json | jq '.total_models'
```

## Monitoring

### GitHub Actions Logs

Check workflow runs at:
`https://github.com/YOUR_USERNAME/LLM-API-Key-Proxy/actions`

### VPS Logs

```bash
# Sync daemon log
tail -f /home/ubuntu/sync_daemon.log

# Gateway log
tail -f /home/ubuntu/llm_proxy.log
```

### Telemetry Dashboard

```bash
# On VPS
python scripts/export_telemetry_summary.py
cat config/telemetry_summary.json | jq '.providers | to_entries | .[] | select(.value.success_rate > 0.9)'
```

## Troubleshooting

### GitHub Actions Fails

1. Check secrets are set
2. Verify models.dev API is accessible
3. Check workflow logs for errors

### VPS Sync Doesn't Trigger

1. Verify crontab is installed: `crontab -l`
2. Check script is executable: `chmod +x ~/vps_sync_daemon.sh`
3. Test manually: `./vps_sync_daemon.sh`

### Gateway Doesn't Reload

1. Check gateway is running: `pgrep -f 'main.py'`
2. Test graceful reload: `pkill -f 'main.py' -HUP`
3. If needed, restart: `pkill -f 'main.py'` then start again

### No Telemetry Data

1. Verify gateway is running and handling requests
2. Check database exists: `ls -la /tmp/llm_proxy_telemetry.db`
3. Verify telemetry recording is enabled in router

## Future Enhancements

- [ ] Web UI for monitoring
- [ ] Slack/Discord notifications on updates
- [ ] A/B testing for fallback chains
- [ ] Machine learning for score prediction
- [ ] Provider health scoring beyond success rate
