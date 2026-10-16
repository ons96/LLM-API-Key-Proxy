#!/usr/bin/env python3
"""Export the gateway's OpenAPI spec to docs/openapi.json (issue #115).

Usage (from the repo root):

    venv/bin/python scripts/export_openapi.py            # rewrite docs/openapi.json
    venv/bin/python scripts/export_openapi.py --check    # fail if the file is stale

The spec is generated from the live FastAPI app, so it always matches the
routes actually registered. Importing the app needs the same environment the
proxy itself needs (PyYAML, litellm, ...), so run it with the project venv.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "docs" / "openapi.json"


def build_spec() -> dict:
    """Import the gateway app and return its OpenAPI document."""
    os.environ.setdefault("SKIP_LAUNCHER", "true")  # no TUI when imported
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from proxy_app import main  # noqa: PLC0415 - heavy import, done on purpose

    return main.app.openapi()


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the file on disk differs from a fresh export",
    )
    args = parser.parse_args(argv)

    spec = build_spec()
    # 2-space indent, stable key order: regenerating without code changes
    # produces a byte-identical file.
    text = json.dumps(spec, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not args.out.exists():
            print(f"stale: {args.out} does not exist", file=sys.stderr)
            return 1
        if args.out.read_text(encoding="utf-8") != text:
            print(f"stale: {args.out} differs from a fresh export", file=sys.stderr)
            return 1
        print(f"up to date: {args.out}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text)} bytes, {len(spec.get('paths', {}))} paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
