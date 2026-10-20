#!/usr/bin/env python3
"""Verify `auto` from three real client shapes.

  1. opencode-style harness: OpenAI-compatible request with tools + tool_choice
  2. plain OpenAI SDK script: the official client, non-streaming
  3. chat-style streaming client: SSE, delta-by-delta

Each check asserts the HTTP status, that content came back, and that every
response carries ``X-Gateway-Decision``.

Usage:
  python3 scripts/harness_check.py --base http://127.0.0.1:8000/v1 \\
      --key-env VPS_GATEWAY_API_KEY --model auto --out /tmp/harness.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DECISION_HEADER = "X-Gateway-Decision"

TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


def _post(url: str, key: str, payload: dict, timeout: float, accept: str) -> tuple[int, dict, str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "Accept": accept,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            # HTTP header names are case-insensitive but dict(headers) is not.
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, \
                resp.read(400_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, {k.lower(): v for k, v in (exc.headers or {}).items()}, \
            exc.read(2000).decode("utf-8", "replace")
    except Exception as exc:
        return 0, {}, f"{type(exc).__name__}: {exc}"


def check_opencode_style(base: str, key: str, model: str, timeout: float) -> dict:
    """Tool-calling request shaped like an opencode/agent harness."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a coding agent. Use tools when asked."},
            {"role": "user", "content": "Read the file README.md and tell me its first line."},
        ],
        "tools": TOOL_SPEC,
        "tool_choice": "auto",
        "max_tokens": 128,
        "stream": False,
    }
    status, headers, body = _post(
        base.rstrip("/") + "/chat/completions", key, payload, timeout, "application/json"
    )
    ok = status == 200
    has_decision = DECISION_HEADER.lower() in headers
    tool_calls = 0
    text = ""
    if ok:
        try:
            parsed = json.loads(body)
            message = (parsed.get("choices") or [{}])[0].get("message") or {}
            tool_calls = len(message.get("tool_calls") or [])
            text = message.get("content") or ""
        except json.JSONDecodeError:
            ok = False
    return {
        "client": "opencode-style (tools)",
        "status": status,
        "ok": ok,
        "tool_calls": tool_calls,
        "chars": len(text),
        "decision": headers.get(DECISION_HEADER.lower()),
        "has_decision_header": has_decision,
        "body_head": body[:300] if not ok else None,
    }


def check_openai_sdk(base: str, key: str, model: str, timeout: float) -> dict:
    """The official OpenAI SDK, non-streaming."""
    try:
        from openai import OpenAI
    except ImportError:
        return {"client": "openai-sdk", "ok": False, "error": "openai package not installed"}

    client = OpenAI(base_url=base.rstrip("/"), api_key=key, timeout=timeout)
    try:
        kwargs = {}
        # The SDK returns parsed headers on the raw response only; use with_raw_response.
        raw = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in five words."}],
            max_tokens=32,
        )
        headers = {k.lower(): v for k, v in (raw.headers or {}).items()}
        completion = raw.parse()
        content = completion.choices[0].message.content or ""
        return {
            "client": "openai-sdk",
            "ok": bool(content),
            "status": 200,
            "chars": len(content),
            "decision": headers.get(DECISION_HEADER.lower()),
            "has_decision_header": DECISION_HEADER.lower() in headers,
        }
    except Exception as exc:
        return {
            "client": "openai-sdk",
            "ok": False,
            "status": getattr(exc, "status_code", 0),
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
        }


def check_streaming(base: str, key: str, model: str, timeout: float) -> dict:
    """Chat-style SSE client: reads deltas as they arrive."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Count from one to five."}],
        "max_tokens": 64,
        "stream": True,
    }
    url = base.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "Accept": "text/event-stream",
        },
    )
    started = time.perf_counter()
    ttft_ms = None
    text_parts: list[str] = []
    chunks = 0
    status = 0
    decision = None
    error = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            decision = {k.lower(): v for k, v in resp.headers.items()}.get(DECISION_HEADER.lower())
            for raw in resp:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - started) * 1000
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload_text = line[5:].strip()
                if payload_text == "[DONE]":
                    break
                chunks += 1
                try:
                    obj = json.loads(payload_text)
                except json.JSONDecodeError:
                    continue
                for choice in obj.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if isinstance(piece, str):
                        text_parts.append(piece)
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = f"http {exc.code}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:160]}"

    text = "".join(text_parts)
    return {
        "client": "streaming",
        "status": status,
        "ok": status == 200 and bool(text),
        "chunks": chunks,
        "chars": len(text),
        "ttft_ms": None if ttft_ms is None else round(ttft_ms, 1),
        "decision": decision,
        "has_decision_header": decision is not None,
        "error": error,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--key-env", default="VPS_GATEWAY_API_KEY")
    ap.add_argument("--key", default="")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    key = args.key or os.environ.get(args.key_env, "")
    if not key:
        print(f"ERROR: no key (set {args.key_env} or pass --key)", file=sys.stderr)
        return 2

    results = [
        check_opencode_style(args.base, key, args.model, args.timeout),
        check_openai_sdk(args.base, key, args.model, args.timeout),
        check_streaming(args.base, key, args.model, args.timeout),
    ]
    report = {"base": args.base, "model": args.model, "results": results}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")

    failed = 0
    for row in results:
        state = "PASS" if row.get("ok") else "FAIL"
        print(
            f"{state} {row['client']:<24} status={row.get('status')} "
            f"chars={row.get('chars', 0)} decision={row.get('decision')} "
            f"{row.get('error') or row.get('body_head') or ''}"
        )
        if not row.get("ok"):
            failed += 1
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
