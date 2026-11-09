#!/usr/bin/env python3
"""Probe the Crax watch_models list and write a timestamped status report.

This is deliberately read-only: it never changes router_config.yaml or a
virtual model. It sends one minimal chat request per watched id so a blocked
frontier model can be rechecked later without probing all 65 Crax models.

Usage:
    CRAX_API_KEY=... python scripts/crax_watch.py
    CRAX_API_KEY=... python scripts/crax_watch.py --output-dir /tmp
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "router_config.yaml"
DEFAULT_BASE_URL = "https://gpt.crax.lol/v1"
TIMEOUT_SECONDS = 30
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def ascii_text(value: Any, limit: int = 0) -> str:
    text = str(value).encode("ascii", "backslashreplace").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] if limit else text


def load_watch_models(config_path: Path) -> tuple[str, list[str]]:
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    crax = (config.get("providers") or {}).get("crax") or {}
    base_url = str(crax.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
    models = crax.get("watch_models") or []
    if not isinstance(models, list) or not models:
        raise SystemExit("crax.watch_models is empty or not a list")
    return base_url, [str(model) for model in models if str(model).strip()]


def parse_chat_body(body: str) -> dict[str, Any]:
    """Parse a normal completion or Crax's default SSE response."""
    if body.lstrip().startswith("data:"):
        parts: list[str] = []
        usage = None
        finish_reason = None
        for line in body.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            choices = chunk.get("choices") or []
            usage = chunk.get("usage") or usage
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            content = delta.get("content") or message.get("content") or ""
            if content:
                parts.append(str(content))
            finish_reason = choice.get("finish_reason") or finish_reason
        return {"content": "".join(parts), "usage": usage, "finish_reason": finish_reason}

    payload = json.loads(body)
    choices = payload.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    return {
        "content": message.get("content") or "",
        "usage": payload.get("usage"),
        "finish_reason": choices[0].get("finish_reason") if choices else None,
    }


def clean_error(body: str, key: str) -> str:
    cleaned = body.replace(key, "[redacted]") if key else body
    if "<html" in cleaned.lower():
        return "upstream HTML error (blocked or gateway error)"
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                return ascii_text(err.get("message") or err.get("type") or err, 300)
            if err:
                return ascii_text(err, 300)
            detail = payload.get("detail")
            if detail:
                return ascii_text(detail, 300)
    except ValueError:
        pass
    return ascii_text(cleaned, 300)


def probe(base_url: str, model: str, key: str) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            "max_tokens": 16,
            "temperature": 0,
        }
    ).encode("utf-8")
    req = request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json",
            "User-Agent": USER_AGENT,
        },
    )
    started = time.monotonic()
    try:
        with request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
            status = response.status
            body = response.read().decode("utf-8", "replace")
    except error.HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - report transport failures as data
        return {
            "model": model,
            "http": None,
            "ok": False,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "error": f"{type(exc).__name__}: {exc}"[:300],
        }

    result: dict[str, Any] = {
        "model": model,
        "http": status,
        "ok": False,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    if status == 200:
        try:
            parsed = parse_chat_body(body)
            result.update(parsed)
            content = ascii_text(parsed.get("content") or "", 300)
            result["content"] = content
            result["ok"] = bool(content.strip())
            if not result["ok"]:
                result["error"] = "HTTP 200 but empty text"
        except (TypeError, ValueError) as exc:
            result["error"] = f"HTTP 200 but unparsable body: {exc}"
    else:
        result["error"] = clean_error(body, key)
    return result


def write_report(
    base_url: str,
    results: list[dict[str, Any]],
    timestamp: datetime,
    output_dir: Path,
) -> tuple[Path, Path]:
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"CRAX_WATCH_{stamp}.json"
    md_path = output_dir / f"CRAX_WATCH_{stamp}.md"
    json_path.write_text(
        json.dumps(
            {
                "tested_at_utc": timestamp.isoformat(),
                "endpoint": base_url,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )

    ok = sum(1 for row in results if row.get("ok"))
    lines = [
        "# Crax watch probe",
        "",
        f"Tested: {timestamp.isoformat()}",
        f"Endpoint: `{base_url}`",
        "",
        f"Watch ids: {len(results)}; passing: {ok}",
        "",
        "| Model | HTTP | Seconds | Outcome |",
        "|---|---:|---:|---|",
    ]
    for row in results:
        outcome = row.get("content") if row.get("ok") else row.get("error", "failed")
        lines.append(
            f"| `{ascii_text(row['model'])}` | {row.get('http') or 'transport'} | "
            f"{row['elapsed_seconds']} | {ascii_text(outcome, 300)} |"
        )
    lines.extend(
        [
            "",
            "A passing entry still needs a tool-call and quota check before it is",
            "promoted into `free_tier_models`.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="ascii")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe the Crax watch_models list")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    key = os.environ.get("CRAX_API_KEY", "").strip()
    if not key:
        print("CRAX_API_KEY is not set", file=sys.stderr)
        return 2
    base_url, models = load_watch_models(args.config)
    print(f"watch_models={len(models)} endpoint={base_url}")
    results: list[dict[str, Any]] = []
    for index, model in enumerate(models):
        result = probe(base_url, model, key)
        results.append(result)
        status = result.get("http") or "transport"
        outcome = "OK" if result.get("ok") else result.get("error", "failed")
        print(f"{status} {model} {result['elapsed_seconds']}s {ascii_text(outcome, 100)}")
        if index + 1 < len(models):
            time.sleep(args.interval)
    json_path, md_path = write_report(
        base_url, results, datetime.now(timezone.utc), args.output_dir
    )
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    print(f"ok={sum(1 for row in results if row.get('ok'))}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
