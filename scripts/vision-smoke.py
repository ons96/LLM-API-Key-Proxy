#!/usr/bin/env python3
"""
Vision smoke for smart-router validation (#481/#486).

Run against a deployed gateway (VPS-40 or local) to verify the previously
failing smoke case 3: a data-URI image request must return HTTP 200 via a
working vision-capable candidate.

This is the repo-local successor to /tmp/opencode/vision-test.py. It:
  1. sends a 1x1 PNG data-URI to the configured virtual model(s)
  2. prints HTTP status, latency, usage, and content preview
  3. exits nonzero if no model returned 200

Usage:
    VPS_GATEWAY_API_KEY=... python3 scripts/vision-smoke.py [model ...]

Defaults to: coding-smart coding-fast
Env:
    GATEWAY_URL   base URL (default http://localhost:8000)
    VPS_GATEWAY_API_KEY  gateway bearer token (required)
"""

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

# 1x1 transparent PNG (valid, minimal).
_PNG = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d49444154789c626001000000ffff030000060005"
        "57bfabd40000000049454e44ae426082"
    )
).decode()

DEFAULT_MODELS = ["coding-smart", "coding-fast"]
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8000")


def _call(model: str, api_key: str, timeout: int = 90):
    body = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what color is this pixel?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{_PNG}"},
                },
            ],
        }
    ]
    payload = {"model": model, "messages": body, "max_tokens": 50}
    req = urllib.request.Request(
        f"{GATEWAY_URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        dt = (time.time() - t0) * 1000
        usage = data.get("usage", {})
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        print(
            f"[{model}] HTTP {r.status} {dt:.0f}ms "
            f"in={usage.get('prompt_tokens')} out={usage.get('completion_tokens')} "
            f"content={content[:60]!r}"
        )
        return True
    except urllib.error.HTTPError as e:
        detail = e.read()[:200].decode(errors="replace")
        print(f"[{model}] FAIL HTTP {e.code} {(time.time()-t0)*1000:.0f}ms {detail}")
        return False
    except Exception as e:
        print(f"[{model}] FAIL {(time.time()-t0)*1000:.0f}ms {type(e).__name__}: {e}")
        return False


def main() -> int:
    api_key = os.environ.get("VPS_GATEWAY_API_KEY")
    if not api_key:
        print("VPS_GATEWAY_API_KEY required", file=sys.stderr)
        return 2
    models = sys.argv[1:] or DEFAULT_MODELS
    results = {m: _call(m, api_key) for m in models}
    ok = all(results.values())
    print(f"\nvision-smoke: {'ALL 200' if ok else 'FAILED'} "
          f"({sum(results.values())}/{len(results)} models)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
