I'll help you fix issue #121 by adding helpful comments to the `.env.example` file. Since I don't have access to the specific repository, I'll provide a comprehensive solution based on common patterns, showing you exactly what changes to make.

## Root Cause

The `.env.example` file lacks explanatory comments, making it difficult for developers to understand:
- What each variable does
- Expected format/values
- Which variables are required vs. optional
- Security implications

## Implementation

Here's how to transform a typical uncommented `.env.example` into a well-documented one:

### Before (Problematic)
```env
APP_NAME=MyApp
APP_ENV=local
APP_KEY=
APP_DEBUG=true
APP_URL=http://localhost

DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE=myapp
DB_USERNAME=root
DB_PASSWORD=

REDIS_HOST=127.0.0.1
REDIS_PASSWORD=null
REDIS_PORT=6379

MAIL_MAILER=smtp
MAIL_HOST=mailpit
MAIL_PORT=1025
```

### After (Fixed)
```env
# ==========================================
# Application Configuration
# ==========================================

# Application name displayed in logs, emails, and UI elements
APP_NAME="MyApp"

# Environment: local, development, staging, or production
# Note: 'production' disables debug mode and enables optimizations
APP_ENV=local

# Application encryption key (32 characters)
# Generate with: php artisan key:generate
# IMPORTANT: Must be unique per environment, keep secret!
APP_KEY=

# Enable detailed error messages and debug information
# WARNING: Set to false in production to avoid leaking sensitive data
APP_DEBUG=true

# Full URL to the application (used for generating links in emails)
APP_URL=http://localhost

# ==========================================
# Database Configuration
# ==========================================

# Database driver: mysql, pgsql, sqlite, or sqlsrv
DB_CONNECTION=mysql

# Database host address
# Use 'db' or service name when running in Docker
DB_HOST=127.0.0.1

# Database port (MySQL: 3306, PostgreSQL: 5432)
DB_PORT=3306

# Database/schema name
DB_DATABASE=myapp

# Database credentials
# Note: Create a dedicated app user instead of using root in production
DB_USERNAME=root
DB_PASSWORD=

# ==========================================
# Cache & Session (Redis)
# ==========================================

# Redis server host
# Use 'redis' when running via Docker Compose
REDIS_HOST=127.0.0.1

# Redis password (leave null if no authentication required)
REDIS_PASSWORD=null

# Redis port (default: 6379)
REDIS_PORT=6379

# Redis database number (0-15) for caching
REDIS_CACHE_DB=0

# Redis database number for sessions
REDIS_SESSION_DB=1

# ==========================================
# Mail Configuration
# ==========================================

# Mail driver: smtp, sendmail, mailgun, ses, postmark, or log
# Use 'log' to write emails to storage/logs for debugging
MAIL_MAILER=smtp

# SMTP host (mailpit for local testing, smtp.mailgun.org, etc.)
MAIL_HOST=mailpit

# SMTP port (587 for TLS, 465 for SSL, 1025 for Mailpit)
MAIL_PORT=1025

# SMTP authentication credentials (if required)
MAIL_USERNAME=null
MAIL_PASSWORD=null

# Encryption method: tls, ssl, or null
MAIL_ENCRYPTION=null

# "From" address for outgoing emails
MAIL_FROM_ADDRESS="hello@example.com"
MAIL_FROM_NAME="${APP_NAME}"

# ==========================================
# Third-Party Services
# ==========================================

# AWS S3 credentials (for file storage)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-1
AWS_BUCKET=myapp-uploads

# Stripe API keys (for payments)
# Test keys start with 'pk_test_' and 'sk_test_'
# Production keys start with 'pk_live_' and 'sk_live_'
STRIPE_KEY=
STRIPE_SECRET=
STRIPE_WEBHOOK_SECRET=

# ==========================================
# Feature Flags (Optional)
# ==========================================

# Enable new beta features (true/false)
ENABLE_BETA_FEATURES=false

# Maintenance mode bypass token (optional)
# Allows accessing the app during maintenance via ?token=XYZ
MAINTENANCE_TOKEN=
```

## Key Changes Made

1. **Section Headers**: Grouped related variables with clear headers for quick scanning
2. **Inline Comments**: 
   - Explained purpose and valid values
   - Noted security implications (e.g., `APP_DEBUG` warning)
   - Provided generation commands (e.g., `php artisan key:generate`)
3. **Contextual Help**: Added notes about Docker service names, port conventions, and environment-specific behavior
4. **Security Warnings**: Flagged sensitive variables like API keys and passwords
5. **Examples**: Showed format examples (e.g., Stripe key prefixes) to prevent confusion

## Best Practices Applied

- **Keep secrets empty**: Left sensitive values like `APP_KEY` and `AWS_SECRET_ACCESS_KEY` empty to force manual entry
- **Consistent formatting**: Used `#` for comments (cross-platform compatible)
- **Hierarchical organization**: Most critical configs (App, DB) at top, optional features at bottom
- **Variable references**: Used `${APP_NAME}` to show relationships between variables

This approach reduces onboarding friction and prevents common configuration errors like using development credentials in production.
