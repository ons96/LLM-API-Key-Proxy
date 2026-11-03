#!/usr/bin/env python3
"""Repeatable operator commands for the New-API sidecar.

Read-only by default. The two mutating paths (`catalog --apply-opencode`,
`priorities --apply`) require an explicit flag. Credentials come from the
environment of an operator-owned env file, never from argv.

    newapi_operator.py audit
    newapi_operator.py catalog --providers groq,cerebras --opencode-config PATH
    newapi_operator.py priorities --order route-a,route-b
    newapi_operator.py boot-check --opencode-config PATH --providers groq

See docs/operator-commands.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from newapi_catalog import (AdminClient, CatalogError, load_provider_metadata,
                            refresh_opencode, refresh_provider)
from newapi_server import channel_key_proof_from_command
from quota_ledger import QuotaLedger, Route

MAX_CONFIG_BYTES = 2 * 1024 * 1024


class OperatorError(RuntimeError):
    """A bounded, non-secret reason an operator command cannot proceed."""


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise OperatorError(name + " is not set")
    return value


def default_key_lookup(row: dict) -> str:
    """Resolve a provider key from the env var named in the metadata row."""
    if row.get("no_api_key_required"):
        return ""
    name = row.get("env_var")
    if not isinstance(name, str) or not name:
        raise OperatorError("provider has no key env var: " + str(row.get("name")))
    value = os.environ.get(name, "")
    if not value:
        raise OperatorError("provider key env var is not set: " + name)
    return value


def run_audit(admin, ledger, relay_key: str, relay_token_id: int) -> dict:
    """Startup-equivalent read-only audit; the same checks `serve()` performs."""
    routes = ledger.routes()
    if not routes:
        raise OperatorError("no configured strict routes")
    digest = admin.audit(routes)
    admin.audit_relay(relay_key, relay_token_id)
    return {
        "routes": len(routes),
        "channels": sorted({route.channel_id for route in routes}),
        "audit_digest": digest,
    }


def refresh_catalogs(ledger, metadata_rows: list[dict], key_lookup=default_key_lookup) -> dict:
    """Fetch provider catalogs into the ledger; never sends completions."""
    catalogs = {}
    for row in metadata_rows:
        catalogs[row["name"]] = refresh_provider(ledger, row, key_lookup(row))
    return {
        "providers": {name: len(rows) for name, rows in catalogs.items()},
        "catalogs": catalogs,
    }


def ordered_routes(ledger, order: list[str]):
    """Map an explicit route order onto the ledger's routes, validating it fully."""
    routes = {route.route_id: route for route in ledger.routes()}
    if len(set(order)) != len(order) or set(order) != set(routes):
        missing = sorted(set(routes) - set(order))
        extra = sorted(set(order) - set(routes))
        raise OperatorError(
            "order must list every route exactly once (missing=%s extra=%s)"
            % (missing, extra)
        )
    if not order:
        raise OperatorError("no configured strict routes")
    return [routes[route_id] for route_id in order]


def _priority_rows(ordered) -> list[dict]:
    total = len(ordered)
    return [
        {"route_id": route.route_id, "channel_id": route.channel_id, "priority": total - index}
        for index, route in enumerate(ordered)
    ]


def preview_priorities(ledger, order: list[str]) -> dict:
    """Show the priority change without touching New-API."""
    return {"apply": False, "order": _priority_rows(ordered_routes(ledger, order))}


def apply_priorities(admin, ledger, order: list[str]) -> dict:
    """Audit, then update channel priorities only."""
    ordered = ordered_routes(ledger, order)
    admin.set_priorities(ordered)
    return {"apply": True, "order": _priority_rows(ordered)}


def load_routes(path: Path) -> list[Route]:
    """Load explicit route definitions; applying them stays an operator choice."""
    try:
        if not path.is_file() or path.stat().st_size > MAX_CONFIG_BYTES:
            raise OperatorError("invalid route file path or size")
        raw = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        raise OperatorError("route file is not readable JSON") from None
    rows = raw.get("routes") if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or not rows:
        raise OperatorError("route file must contain a non-empty list")
    routes = []
    for row in rows:
        if not isinstance(row, dict):
            raise OperatorError("each route must be a JSON object")
        try:
            routes.append(Route(**row))
        except (TypeError, ValueError) as exc:
            raise OperatorError("invalid route definition: " + str(exc)) from None
    if len({route.route_id for route in routes}) != len(routes):
        raise OperatorError("route ids must be unique")
    return routes


def _route_rows(routes: list[Route]) -> list[dict]:
    return [{"route_id": route.route_id, "alias": route.alias,
             "channel_id": route.channel_id, "model_id": route.model_id}
            for route in routes]


def preview_routes(routes: list[Route]) -> dict:
    """Show what would be configured without writing the ledger."""
    return {"apply": False, "routes": _route_rows(routes)}


def apply_routes(ledger, routes: list[Route]) -> dict:
    """Write explicit routes into the ledger; never touches New-API."""
    for route in routes:
        ledger.configure(route)
    return {"apply": True, "routes": _route_rows(routes), "configured": len(routes)}


def boot_check(path: Path, providers: list[str]) -> dict:
    """Validate an OpenCode configuration without writing to it."""
    target = path.expanduser()
    try:
        if not target.is_file() or target.stat().st_size > MAX_CONFIG_BYTES:
            raise OperatorError("invalid OpenCode configuration path or size")
        config = json.loads(target.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        raise OperatorError("OpenCode configuration is not readable JSON") from None
    if not isinstance(config, dict):
        raise OperatorError("OpenCode configuration must be a JSON object")
    registry = config.get("provider")
    if not isinstance(registry, dict):
        raise OperatorError("OpenCode provider map missing")
    counts = {}
    missing = []
    for name in providers:
        provider = registry.get(name)
        models = provider.get("models") if isinstance(provider, dict) else None
        if not isinstance(models, dict) or not models:
            missing.append(name)
            continue
        counts[name] = len(models)
    return {
        "ok": not missing,
        "provider_count": len(registry),
        "models": counts,
        "missing": missing,
    }


def _admin_from_env():
    proof_command = os.environ.get("NEWAPI_CHANNEL_KEY_PROOF_COMMAND", "")
    key_proof = channel_key_proof_from_command(proof_command) if proof_command else None
    return AdminClient(
        _require("NEWAPI_BASE_URL"),
        _require("NEWAPI_ADMIN_TOKEN"),
        int(_require("NEWAPI_ADMIN_USER_ID")),
        key_proof=key_proof,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="New-API sidecar operator commands")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("audit", help="read-only startup-equivalent audit")

    routes = commands.add_parser("routes", help="preview or apply explicit route definitions")
    routes.add_argument("--file", required=True, help="JSON route definitions")
    routes.add_argument("--apply", action="store_true")

    catalog = commands.add_parser("catalog", help="refresh provider catalogs")
    catalog.add_argument("--providers", required=True, help="comma-separated provider names")
    catalog.add_argument("--provider-db", default=os.environ.get("NEWAPI_PROVIDER_DB", ""))
    catalog.add_argument("--opencode-config", default=os.environ.get("OPENCODE_CONFIG", ""))
    catalog.add_argument("--apply-opencode", action="store_true")

    priorities = commands.add_parser("priorities", help="preview or apply channel priorities")
    priorities.add_argument("--order", required=True, help="comma-separated route ids, best first")
    priorities.add_argument("--apply", action="store_true")

    boot = commands.add_parser("boot-check", help="validate an OpenCode configuration")
    boot.add_argument("--opencode-config", required=True)
    boot.add_argument("--providers", default="", help="comma-separated provider names")

    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            admin = _admin_from_env()
            ledger = QuotaLedger(Path(_require("SIDECAR_DB")))
            try:
                result = run_audit(admin, ledger, _require("NEWAPI_API_KEY"),
                                   int(_require("NEWAPI_RELAY_TOKEN_ID")))
            finally:
                ledger.close()
        elif args.command == "routes":
            definitions = load_routes(Path(args.file))
            if args.apply:
                ledger = QuotaLedger(Path(_require("SIDECAR_DB")))
                try:
                    result = apply_routes(ledger, definitions)
                finally:
                    ledger.close()
            else:
                result = preview_routes(definitions)
        elif args.command == "catalog":
            if not args.provider_db:
                raise OperatorError("--provider-db or NEWAPI_PROVIDER_DB is required")
            names = [name.strip() for name in args.providers.split(",") if name.strip()]
            if not names:
                raise OperatorError("at least one provider name is required")
            metadata = load_provider_metadata(Path(args.provider_db), names)
            ledger = QuotaLedger(Path(_require("SIDECAR_DB")))
            try:
                result = refresh_catalogs(ledger, metadata)
            finally:
                ledger.close()
            if args.opencode_config:
                result["opencode"] = refresh_opencode(
                    Path(args.opencode_config), result["catalogs"],
                    apply=args.apply_opencode,
                )
            result.pop("catalogs", None)
        elif args.command == "priorities":
            order = [name.strip() for name in args.order.split(",") if name.strip()]
            ledger = QuotaLedger(Path(_require("SIDECAR_DB")))
            try:
                if args.apply:
                    result = apply_priorities(_admin_from_env(), ledger, order)
                else:
                    result = preview_priorities(ledger, order)
            finally:
                ledger.close()
        else:
            names = [name.strip() for name in args.providers.split(",") if name.strip()]
            result = boot_check(Path(args.opencode_config), names)
    except (OperatorError, CatalogError, OSError, ValueError) as exc:
        print("operator command failed: " + str(exc), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
