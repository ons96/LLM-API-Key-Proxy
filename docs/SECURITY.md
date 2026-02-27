# Security Best Practices

This document outlines security best practices for deploying and operating the Mirro-Proxy API Key Proxy Server.

## Table of Contents

1. [Authentication & Authorization](#authentication--authorization)
2. [Credential Management](#credential-management)
3. [Network Security](#network-security)
4. [Logging & Privacy](#logging--privacy)
5. [Rate Limiting & Abuse Prevention](#rate-limiting--abuse-prevention)
6. [Provider Security](#provider-security)
7. [Deployment Security](#deployment-security)

---

## Authentication & Authorization

### PROXY_API_KEY (Required)

The `PROXY_API_KEY` environment variable is your primary defense against unauthorized access.

**Critical Requirements:**
- **Always set PROXY_API_KEY in production.** Running without it allows anyone to access your proxy and consume your provider credits.
- Use a cryptographically secure random string (minimum 32 characters recommended).
- Generate using: `openssl rand -hex 32` or similar secure random generator.
- Rotate keys periodically (90-day recommended maximum age).
- Never commit the key to version control.

**Example secure generation:**
```bash
export PROXY_API_KEY=$(openssl rand -hex 32)
echo "PROXY_API_KEY=$PROXY_API_KEY" >> .env
```

**Key Masking:**
The proxy automatically masks API keys in startup logs (showing only first/last 4 characters). Verify this masking is working on startup:
```
✓ a1b2...c3d4
```

---

## Credential Management

### Environment Variable Security

All sensitive credentials (provider API keys, OAuth tokens) are loaded from `.env` files:

1. **File Permissions:**
   ```bash
   chmod 600 .env
   chmod 600 *.env
   ```

2. **File Locations:**
   - Store `.env` files outside web-accessible directories.
   - Never upload `.env` files to Docker registries or Git repositories.
   - Use `.gitignore` to exclude all `*.env` files.

3. **Multiple Environment Files:**
   The proxy loads all `*.env` files in the root directory. Use this for organizing credentials:
   - `.env` - Core proxy settings (PROXY_API_KEY)
   - `antigravity_all_combined.env` - Antigravity provider credentials
   - `gemini_cli_all_combined.env` - Gemini provider credentials

### OAuth Credential Encryption

When using `python -m proxy_app.main --add-credential`:
- Credentials are encrypted at rest using the system's keyring or secure storage.
- Encryption keys are derived from machine-specific identifiers.
- Credentials are never stored in plaintext in configuration files.

**Best Practices:**
- Run the credential tool on the production server to ensure encryption keys match the deployment environment.
- Backup encryption keys separately from configuration files.
- Use the credential rotation feature every 90 days.

---

## Network Security

### Host Binding

**Development:**
```bash
python -m proxy_app.main --host 127.0.0.1 --port 8000
```

**Production (with reverse proxy):**
```bash
python -m proxy_app.main --host 127.0.0.1 --port 8000
# Use nginx/traefik/caddy for TLS termination and public access
```

**⚠️ Dangerous:**
```bash
# Never expose directly to internet without authentication
python -m proxy_app.main --host 0.0.0.0 --port 8000  # INSECURE without reverse proxy
```

### CORS Configuration

The proxy includes CORS middleware. Restrict origins in production:

```python
# In your deployment configuration, override CORS:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Never use "*" in production
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

### HTTPS/TLS

**Required for Production:**
- Never transmit API keys or OAuth tokens over unencrypted HTTP.
- Use a reverse proxy (nginx, Caddy, Traefik) with valid TLS certificates.
- Consider using Cloudflare or similar for DDoS protection.

**Example nginx configuration:**
```nginx
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## Logging & Privacy

### Request Logging

Enable with `--enable-request-logging` only when debugging:

```bash
# Development only - contains sensitive data
python -m proxy_app.main --enable-request-logging

# Production - disable to prevent logging PII/API keys
python -m proxy_app.main
```

**Data Retention:**
- Logs may contain user prompts, API responses, and metadata.
- Implement log rotation to prevent disk exhaustion.
- Sanitize logs before sharing for debugging (remove `Authorization` headers).

### PII Handling

The proxy processes user messages that may contain PII. Consider:
- Implementing request/response sanitization middleware.
- Using provider-specific privacy features (zero-data retention where available).
- Documenting data processing in your privacy policy.

---

## Rate Limiting & Abuse Prevention

While the proxy includes basic rate limiting, implement additional layers:

1. **Reverse Proxy Rate Limiting:**
   ```nginx
   limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
   limit_req zone=api burst=20 nodelay;
   ```

2. **Provider Quotas:**
   - Configure spending limits at the provider level (OpenAI, Anthropic, etc.).
   - Monitor the `/health` endpoint for unusual traffic patterns.
   - Set up alerts for cost thresholds.

3. **IP Whitelisting:**
   - Restrict access to known IP ranges when possible.
   - Use VPNs or private networks for internal deployments.

---

## Provider Security

### API Key Rotation

The rotator library supports automatic key rotation. Enable in `config/router_config.yaml`:

```yaml
security:
  auto_rotate_keys: true
  rotation_interval_days: 30
  max_failures_before_rotation: 5
```

### Provider-Specific Security

**Google OAuth (Gemini):**
- Use service accounts with minimal permissions.
- Rotate OAuth tokens every 7 days.
- Monitor Google Cloud Console for unauthorized access.

**Cerebras/Groq:**
- Use organization-level API keys when available.
- Restrict keys to specific models/usage patterns.
- Enable audit logging in provider dashboards.

---

## Deployment Security

### Docker Security

If using Docker:

```dockerfile
# Use non-root user
RUN useradd -m -u 1000 proxyuser
USER proxyuser

# Don't bake secrets into image
ENV PROXY_API_KEY_FILE=/run/secrets/proxy_api_key
```

**Docker Compose:**
```yaml
services:
  proxy:
    secrets:
      - proxy_api_key
    environment:
      - PROXY_API_KEY_FILE=/run/secrets/proxy_api_key
    read_only: true
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE

secrets:
  proxy_api_key:
    file: ./secrets/proxy_api_key.txt
```

### Health Check Security

The `/health` endpoint exposes system status. Protect it:

```yaml
# config/router_config.yaml
health_check:
  enabled: true
  require_auth: true  # Require PROXY_API_KEY for health checks
  expose_detailed_info: false  # Don't expose provider keys/credentials
```

### Secrets Management

For production deployments, integrate with proper secrets management:

- **AWS:** AWS Secrets Manager or Parameter Store
- **GCP:** Secret Manager
- **Azure:** Key Vault
- **Kubernetes:** Sealed Secrets or External Secrets Operator

---

## Security Checklist

Before deploying to production:

- [ ] `PROXY_API_KEY` is set to a secure random value (32+ chars)
- [ ] `.env` files have permissions `600` (owner read/write only)
- [ ] `.env` files are in `.gitignore`
- [ ] Running behind HTTPS reverse proxy (TLS 1.2+)
- [ ] CORS origins restricted to specific domains (no wildcards)
- [ ] Request logging disabled (`--enable-request-logging` not used)
- [ ] Rate limiting configured at reverse proxy level
- [ ] Provider spending limits configured
- [ ] Non-root user running the process
- [ ] Health endpoint requires authentication (if exposed)
- [ ] Log rotation configured to prevent disk exhaustion
- [ ] Backup strategy for encrypted credentials tested
- [ ] Incident response plan documented

---

## Incident Response

If you suspect a security breach:

1. **Immediate:**
   - Revoke all provider API keys.
   - Rotate `PROXY_API_KEY`.
   - Check access logs for unauthorized IPs.

2. **Investigation:**
   - Review provider dashboards for unusual usage.
   - Check credential modification timestamps.
   - Analyze request logs if available.

3. **Recovery:**
   - Generate new credentials using `credential_tool`.
   - Verify configuration file integrity.
   - Re-deploy with rotated keys.

## Reporting Vulnerabilities

If you discover a security vulnerability in this project, please do not open a public issue. Instead, contact the maintainers directly with details of the vulnerability.

---

**Last Updated:** 2024
**Version:** 1.3.0
