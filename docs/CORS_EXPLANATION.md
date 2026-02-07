# CORS Wildcard Warning - Explanation

## What is CORS?
**CORS (Cross-Origin Resource Sharing)** is a browser security feature. When a web app (like Kobold Lite at https://lite.koboldai.net) tries to call your API (http://40.233.101.233:8000), the browser checks if your API allows requests from that website.

## What is the Wildcard Issue?
In your `.env` file, if someone sets:
```bash
CORS_ORIGINS=*
```

This allows **ANY website** to call your API. This is dangerous because:
- Malicious websites could use your API without permission
- It bypasses the security CORS is supposed to provide
- Anyone on the internet could potentially abuse your gateway

## What I Should Add:
A warning message in the logs when CORS is set to wildcard:
```
⚠️  WARNING: CORS is set to allow all origins (*). This is insecure for production!
```

This alerts users that they're running in an insecure mode.

## Your CORS is Currently Safe:
You have:
```bash
CORS_ORIGINS=https://lite.koboldai.net,https://koboldai.net
```

This is secure - only Kobold Lite can access your API.
