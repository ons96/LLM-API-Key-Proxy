# Security Deployment Checklist

Quick reference checklist for secure deployment of Mirro-Proxy.

## Pre-Deployment

### Environment Setup
- [ ] Created strong `PROXY_API_KEY` (32+ random characters)
- [ ] Stored `PROXY_API_KEY` in `.env` file (not in code)
- [ ] Set `.env` file permissions to `600` (readable only by owner)
- [ ] Added `.env` to `.gitignore`
- [ ] Verified no secrets in git history (`git log --all --full-history -- .env`)

### Credential Management
- [ ] Generated provider API keys with minimal permissions
- [ ] Enabled provider spending limits/alerts
- [ ] Stored provider credentials in separate `.env` files (not `.env` in repo root if public)
- [ ] Tested credential encryption: `python -m proxy_app.main --add-credential`
- [ ] Documented credential rotation schedule (90 days recommended)

### Network Configuration
- [ ] Configured firewall to block port 8000 from public internet (if possible)
- [ ] Set up reverse proxy (nginx/Caddy/traefik) with TLS 1.2+
- [ ] Obtained valid SSL certificate (not self-signed for production)
- [ ] Configured DNS with proper A/AAAA records
- [ ] Restricted CORS to specific domains (no `*`)

## Deployment

### Application Security
- [ ] Running with non-root user
- [ ] Bound to localhost only (`--host 127.0.0.1`) when behind reverse proxy
- [ ] Disabled request logging (`--enable-request-logging` NOT used in prod)
- [ ] Verified API key masking in startup logs (`✓ abcd...efgh`)
- [ ] Confirmed `/health` endpoint doesn't leak sensitive info

### Rate Limiting & Monitoring
- [ ] Configured reverse proxy rate limiting
- [ ] Set up provider dashboard alerts for:
  - [ ] Unusual spend patterns
  - [ ] High error rates
  - [ ] New IP addresses accessing keys
- [ ] Configured log rotation (prevent disk full DoS)
- [ ] Set up monitoring for `PROXY_API_KEY` authentication failures

### Provider Configuration
- [ ] Enabled automatic key rotation in config
- [ ] Set circuit breaker thresholds
- [ ] Configured allowed model whitelist (block expensive models if not needed)
- [ ] Verified backup provider configured (failover capability)

## Post-Deployment

### Verification
- [ ] Tested authentication: Request without API key returns 401/403
- [ ] Tested HTTPS: No warnings in browser/API client
- [ ] Verified CORS: Cross-origin requests rejected from unauthorized domains
- [ ] Checked logs: No API keys or credentials in plaintext logs
- [ ] Load tested: Application stable under expected traffic

### Documentation
- [ ] Documented incident response procedures
- [ ] Shared `PROXY_API_KEY` securely with team (password manager, not Slack/email)
- [ ] Documented which models are enabled and why
- [ ] Created runbook for credential rotation

### Maintenance
- [ ] Scheduled monthly security review
- [ ] Set calendar reminder for credential rotation (90 days)
- [ ] Configured automated security scanning on repository
- [ ] Enabled Dependabot or similar for dependency updates

## Emergency Procedures

### If Compromised:
1. [ ] Immediately revoke all provider API keys via provider dashboards
2. [ ] Rotate `PROXY_API_KEY`
3. [ ] Check access logs for unauthorized IPs
4. [ ] Review provider usage for unexpected costs
5. [ ] Force re-authentication for all clients

### Kill Switch:
Create file to immediately stop processing:
```bash
touch /tmp/mirro-proxy-kill-switch
```

Remove to resume:
```bash
rm /tmp/mirro-proxy-kill-switch
```

---

**Sign-off:**
- [ ] Security review completed by: _________________
- [ ] Date: _________________
- [ ] Version deployed: _________________
