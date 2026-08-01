# GitHub Actions Optimization Summary

## Summary of Changes Made

Your repository has been optimized to maximize your chances of obtaining the Oracle Always-Free VPS (4 cores, 24GB RAM).

### Key Improvements:

1. **24/7 Continuous Operation**: Runs every 6 hours with 6-hour job duration (max GitHub limit)
2. **All 3 Availability Domains**: Now cycles through AD-1, AD-2, AD-3 automatically
3. **Faster Retries**: 30-second retry interval (was 60 seconds)
4. **Smart Instance Detection**: Checks if instance already exists before attempting creation
5. **Daily Reminders**: Once instance is created, sends daily reminder issues so you don't forget
6. **Optimized Schedule**: 4 runs/day at 00:00, 06:00, 12:00, 18:00 UTC for 24/7 coverage

### Why These Changes Work Better:

**Before**: 4x/day × 10 minutes = 40 minutes of retry time daily
**After**: 4x/day × 6 hours = 24 hours of retry time daily (continuous coverage)

With unlimited GitHub Actions minutes for public repos, you can run 24/7 for free!

### Required Secrets (5 total):

1. `OCI_USER_ID` - Your Oracle Cloud user OCID
2. `OCI_TENANCY_ID` - Your tenancy OCID
3. `OCI_FINGERPRINT` - API key fingerprint
4. `OCI_PRIVATE_KEY` - Full private key content (including BEGIN/END lines)
5. `OCI_REGION` - Region (e.g., us-ashburn-1, eu-frankfurt-1)

### Optional Secrets for Notifications:

6. `EMAIL` - Your email address for notifications
7. `EMAIL_PASSWORD` - Your email password or app-specific password
8. `DISCORD_WEBHOOK` - Discord webhook URL for Discord notifications

### How to Add Secrets:

Go to: **Settings → Secrets and variables → Actions → New repository secret**

Add each secret one by one with the exact names above.

### How to Get Oracle Cloud Credentials:

1. Log into Oracle Cloud Console
2. Go to Profile (top right) → User settings
3. Copy your User OCID → Save as `OCI_USER_ID`
4. Copy your Tenancy OCID → Save as `OCI_TENANCY_ID`
5. Go to API Keys → Add API Key
6. Download the private key
7. Copy the fingerprint shown → Save as `OCI_FINGERPRINT`
8. Open the downloaded .pem file, copy ALL contents (including BEGIN/END lines) → Save as `OCI_PRIVATE_KEY`
9. Check your region in the URL (e.g., `us-ashburn-1`) → Save as `OCI_REGION`

### After Setup:

1. The workflow will run automatically every 6 hours
2. Once an instance is created, you'll get a GitHub issue notification
3. Daily reminders will continue until you manually disable the workflow
4. To disable: Go to Actions tab → Click "Oracle Always-Free VPS Signup" → Disable workflow

### Monitoring:

- Check the Actions tab to see run history
- Download artifacts from each run to see detailed logs
- Look for the INSTANCE_CREATED file for instance details
- Check issues for daily reminders

---

**Your repository is now optimized for maximum success probability!** 🚀
