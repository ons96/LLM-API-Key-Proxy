#!/usr/bin/env python3
"""Gateway streaming smoke + TTFT measurement.

Measures what the gateway actually does, not what the config claims:

  --aliases a,b,c   sequentially stream one request per alias, record TTFT
  --concurrent N    fire N simultaneous streaming requests at one alias and
                    record status codes, TTFT spread and whether any 5xx
  --rss             sample the gateway process RSS (KB) before and after

Uses only stdlib so it runs on VPS-40 without touching the gateway venv.

Usage:
  python3 scripts/gateway_smoke_stream.py --base http://127.0.0.1:8000/v1 \\
      --key-env VPS_GATEWAY_API_KEY --aliases auto,coding-fast,chat-fast \\
      --concurrent 10 --out /tmp/smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_PROMPT = "Reply with exactly one short sentence about tests."
MIN_TOKENS = 16


def rss_kb(pattern: str = "proxy_app.main") -> int | None:
    """RSS of the gateway process, read from ps. None when not found."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "rss,args"], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if pattern in line:
            parts = line.split(None, 1)
            try:
                return int(parts[0])
            except (ValueError, IndexError):
                continue
    return None


def stream_once(base: str, key: str, alias: str, timeout: float) -> dict:
    """One streaming chat completion. Returns timings and status."""
    body = json.dumps(
        {
            "model": alias,
            "messages": [{"role": "user", "content": DEFAULT_PROMPT}],
            "max_tokens": MIN_TOKENS,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "Accept": "text/event-stream",
        },
    )
    started = time.perf_counter()
    ttft_ms = None
    chunks = 0
    text_parts: list[str] = []
    status = 0
    error = None
    decision = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            decision = resp.headers.get("X-Gateway-Decision")
            for raw in resp:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - started) * 1000.0
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunks += 1
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if isinstance(piece, str):
                        text_parts.append(piece)
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = f"http {exc.code}"
        try:
            error += ": " + exc.read(300).decode("utf-8", "replace")
        except Exception:
            pass
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)[:160]
    return {
        "alias": alias,
        "status": status,
        "ttft_ms": None if ttft_ms is None else round(ttft_ms, 1),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
        "chunks": chunks,
        "chars": sum(len(p) for p in text_parts),
        "decision_header": decision,
        "error": error,
    }


def run_sequential(base: str, key: str, aliases: list[str], timeout: float) -> list[dict]:
    return [stream_once(base, key, alias, timeout) for alias in aliases]


def run_concurrent(base: str, key: str, alias: str, count: int, timeout: float) -> dict:
    results: list[dict] = [None] * count  # type: ignore[list-item]

    def worker(idx: int) -> None:
        results[idx] = stream_once(base, key, alias, timeout)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
    started = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_ms = (time.perf_counter() - started) * 1000.0
    done = [r for r in results if r]
    ttfts = [r["ttft_ms"] for r in done if r["ttft_ms"] is not None]
    return {
        "alias": alias,
        "count": count,
        "wall_ms": round(wall_ms, 1),
        "statuses": sorted({r["status"] for r in done}),
        f"http_5xx": sum(1 for r in done if 500 <= r["status"] < 600),
        "errors": [r["error"] for r in done if r["error"]],
        "ttft_p50_ms": round(statistics.median(ttfts), 1) if ttfts else None,
        "ttft_p95_ms": (
            round(sorted(ttfts)[max(0, int(round(0.95 * len(ttfts))) - 1)], 1)
            if ttfts
            else None
        ),
        "results": done,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--key-env", default="VPS_GATEWAY_API_KEY")
    ap.add_argument("--key", default="", help="key value (prefer --key-env)")
    ap.add_argument("--aliases", default="auto")
    ap.add_argument("--concurrent", type=int, default=0)
    ap.add_argument("--concurrent-alias", default="auto")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    key = args.key or os.environ.get(args.key_env, "")
    if not key:
        print(f"ERROR: no key (set {args.key_env} or pass --key)", file=sys.stderr)
        return 2

    report: dict = {"base": args.base, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    report["rss_kb_before"] = rss_kb()
    aliases = [a.strip() for a in args.aliases.split(",") if a.strip()]
    report["sequential"] = run_sequential(args.base, key, aliases, args.timeout)
    if args.concurrent > 0:
        report["concurrent"] = run_concurrent(
            args.base, key, args.concurrent_alias, args.concurrent, args.timeout
        )
    report["rss_kb_after"] = rss_kb()

    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    for row in report["sequential"]:
        print(
            f"{row['alias']:<16} status={row['status']} ttft={row['ttft_ms']}ms "
            f"chars={row['chars']} decision={row['decision_header']} err={row['error']}"
        )
    if "concurrent" in report:
        c = report["concurrent"]
        print(
            f"concurrent n={c['count']} wall={c['wall_ms']}ms statuses={c['statuses']} "
            f"5xx={c['http_5xx']} ttft_p50={c['ttft_p50_ms']} ttft_p95={c['ttft_p95_ms']}"
        )
    print(f"rss_kb before={report['rss_kb_before']} after={report['rss_kb_after']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
