#!/usr/bin/env python3
"""Multi-key management + dead-key verification CLI.

Subcommands:
    list [--provider P]                 List keys (all or per-provider) with status.
    add PROVIDER KEY_VALUE [--id ID]    Add key, write to ~/.secrets/, update manifest.
    remove PROVIDER ID                  Mark key removed (NEVER delete the file).
    verify [--provider P] [--timeout T] Ping each key's provider /models endpoint.
                                        Classify: working / dead / rate_limited / unreachable.
    rotate PROVIDER                     Print next working key (round-robin).
    refresh-models PROVIDER [--timeout T] Hit /v1/models, write to manifest.

Storage:
    Manifest at ~/.secrets/keys.json: {version:1, providers:{P:[KeyEntry, ...]}}
    Key files at ~/.secrets/<provider>-key-<id> (content = key value, no newline).

Exit codes: 0=success, 1=error, 2=dry-run.

Refs: task-board #198. Ties into #196 penalty_store for dead-key deprioritization.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SECRETS_DIR = Path(
    os.environ.get("KEYS_SECRETS_DIR", os.path.expanduser("~/.secrets"))
)
MANIFEST_PATH = Path(
    os.environ.get("KEYS_MANIFEST", str(SECRETS_DIR / "keys.json"))
)
DEFAULT_TIMEOUT = float(os.environ.get("KEYS_VERIFY_TIMEOUT", "10"))
MANIFEST_VERSION = 1

# Status enum (kept as plain strings to keep JSON human-readable).
WORKING = "working"
DEAD = "dead"
RATE_LIMITED = "rate_limited"
UNREACHABLE = "unreachable"
UNVERIFIED = "unverified"
REMOVED = "removed"

ALL_STATUSES = {
    WORKING,
    DEAD,
    RATE_LIMITED,
    UNREACHABLE,
    UNVERIFIED,
    REMOVED,
}
ACTIVE_STATUSES = {WORKING, RATE_LIMITED, UNVERIFIED}  # not DEAD/REMOVED/UNREACHABLE


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class KeyEntry:
    """One API key + its metadata."""

    id: str
    env_var: str = ""
    file_path: str = ""
    status: str = UNVERIFIED
    last_verified: float = 0.0
    added_at: float = field(default_factory=time.time)
    last_used: float = 0.0
    detail: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "KeyEntry":
        # ponytail: tolerate missing fields from older manifests.
        return cls(
            id=str(d.get("id", "")),
            env_var=str(d.get("env_var", "")),
            file_path=str(d.get("file_path", "")),
            status=str(d.get("status", UNVERIFIED)),
            last_verified=float(d.get("last_verified", 0.0)),
            added_at=float(d.get("added_at", time.time())),
            last_used=float(d.get("last_used", 0.0)),
            detail=str(d.get("detail", "")),
        )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class KeyManifest:
    """JSON-backed manifest of all keys per provider."""

    def __init__(self, path: Path = MANIFEST_PATH):
        self.path = path
        self.providers: Dict[str, List[KeyEntry]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.providers = {}
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("manifest load failed (%s); starting empty", exc)
            self.providers = {}
            return
        self.providers = {
            name: [KeyEntry.from_dict(e) for e in entries]
            for name, entries in (data.get("providers") or {}).items()
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": MANIFEST_VERSION,
            "updated_at": time.time(),
            "providers": {
                name: [e.to_dict() for e in entries]
                for name, entries in self.providers.items()
            },
        }
        # ponytail: atomic write via tmp + rename.
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(self.path)

    # -- query --

    def list_providers(self) -> List[str]:
        return sorted(self.providers.keys())

    def get_keys(self, provider: str) -> List[KeyEntry]:
        return list(self.providers.get(provider, []))

    def get_active_keys(self, provider: str) -> List[KeyEntry]:
        return [k for k in self.get_keys(provider) if k.status in ACTIVE_STATUSES]

    def next_working_key(self, provider: str) -> Optional[KeyEntry]:
        """Round-robin: pick the working key with the oldest last_used."""
        active = [k for k in self.get_keys(provider) if k.status == WORKING]
        if not active:
            return None
        # ponytail: stable min on last_used. Ties broken by insertion order.
        return min(active, key=lambda k: k.last_used)

    # -- mutate --

    def add_key(
        self,
        provider: str,
        key_value: str,
        key_id: Optional[str] = None,
        env_var: str = "",
        write_file: bool = True,
    ) -> KeyEntry:
        """Add a key. Writes the key to ~/.secrets/<provider>-key-<id> by default."""
        entries = self.providers.setdefault(provider, [])
        if key_id is None:
            # ponytail: numeric ids 1, 2, ... skip collisions.
            existing = {e.id for e in entries}
            n = 1
            while str(n) in existing:
                n += 1
            key_id = str(n)
        file_path = ""
        if write_file:
            file_path = str(SECRETS_DIR / f"{provider}-key-{key_id}")
            SECRETS_DIR.mkdir(parents=True, exist_ok=True)
            Path(file_path).write_text(key_value)
        entry = KeyEntry(
            id=key_id,
            env_var=env_var,
            file_path=file_path,
            status=UNVERIFIED,
            added_at=time.time(),
        )
        entries.append(entry)
        self.save()
        return entry

    def remove_key(self, provider: str, key_id: str) -> Optional[KeyEntry]:
        """Mark a key removed (NEVER delete the file)."""
        for entry in self.providers.get(provider, []):
            if entry.id == key_id:
                entry.status = REMOVED
                entry.detail = f"marked removed at {time.time()}"
                self.save()
                return entry
        return None

    def update_status(
        self,
        provider: str,
        key_id: str,
        status: str,
        detail: str = "",
    ) -> None:
        for entry in self.providers.get(provider, []):
            if entry.id == key_id:
                entry.status = status
                entry.detail = detail
                entry.last_verified = time.time()
        self.save()

    def mark_used(self, provider: str, key_id: str) -> None:
        for entry in self.providers.get(provider, []):
            if entry.id == key_id:
                entry.last_used = time.time()
                break
        self.save()


# ---------------------------------------------------------------------------
# Provider lookup
# ---------------------------------------------------------------------------


def _provider_base_url(provider: str, db_path: Optional[str] = None) -> Tuple[str, str]:
    """Look up (env_var, base_url) for a provider from llm_providers.db.

    Returns ("", "") if not found. db_path defaults to the standard location.
    """
    if db_path is None:
        db_path = os.environ.get(
            "LLM_PROVIDERS_DB",
            os.path.expanduser("~/CodingProjects/llm-provider-manager/llm_providers.db"),
        )
    if not os.path.exists(db_path):
        return ("", "")
    try:
        import sqlite3

        # ponytail: read-only URI to avoid accidental writes.
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT env_var, base_url FROM providers WHERE key_name=? LIMIT 1",
                (provider,),
            ).fetchone()
            if row:
                return (str(row[0] or ""), str(row[1] or ""))
        finally:
            conn.close()
    except Exception as exc:
        logger.debug("provider lookup failed for %s: %s", provider, exc)
    return ("", "")


def _read_key_file(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Verify (async HTTP ping)
# ---------------------------------------------------------------------------


def classify_response(status_code: int, body: str = "") -> str:
    """Map an HTTP response to a key status."""
    if status_code == 200:
        return WORKING
    if status_code in (401, 403):
        return DEAD
    if status_code == 429:
        return RATE_LIMITED
    if 500 <= status_code < 600:
        return UNREACHABLE
    return UNVERIFIED


async def verify_key(
    provider: str,
    base_url: str,
    key_value: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> Tuple[str, str]:
    """Ping a provider's /models endpoint. Returns (status, detail)."""
    if not base_url:
        return (UNREACHABLE, "no base_url for provider")
    if not key_value:
        return (DEAD, "empty key value")
    url = base_url.rstrip("/") + "/models"
    try:
        import httpx
    except ImportError:
        return (UNREACHABLE, "httpx not installed")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {key_value}"})
        status = classify_response(resp.status_code, resp.text[:200])
        return (status, f"HTTP {resp.status_code}")
    except httpx.TimeoutException:
        return (UNREACHABLE, f"timeout after {timeout}s")
    except Exception as exc:
        return (UNREACHABLE, f"{type(exc).__name__}: {exc}")


async def verify_manifest(
    manifest: KeyManifest,
    provider: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
    db_path: Optional[str] = None,
) -> List[Tuple[str, str, str, str]]:
    """Verify all (or one provider's) keys. Returns list of
    (provider, key_id, status, detail)."""
    providers = [provider] if provider else manifest.list_providers()
    results: List[Tuple[str, str, str, str]] = []
    for name in providers:
        env_var, base_url = _provider_base_url(name, db_path)
        for entry in manifest.get_keys(name):
            if entry.status == REMOVED:
                results.append((name, entry.id, REMOVED, "skipped (removed)"))
                continue
            key_value = _read_key_file(entry.file_path)
            status, detail = await verify_key(name, base_url, key_value, timeout)
            manifest.update_status(name, entry.id, status, detail)
            results.append((name, entry.id, status, detail))
            # ponytail: best-effort penalty_store wire; never break verify loop.
            if status == DEAD:
                _try_record_penalty(name, entry.id)
    return results


def _try_record_penalty(provider: str, key_id: str) -> None:
    """Best-effort: record invalid_key failure in penalty_store (#196).

    Wrapped in try/except so verify works even if penalty_store isn't
    installed or importable.
    """
    try:
        # Lazy import: penalty_store lives in src/proxy_app/ on the gateway.
        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), "..", "src")
        )
        from proxy_app.penalty_store import PenaltyStore  # type: ignore

        store = PenaltyStore.get()
        # Run async call from sync context.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                store.arecord_failure(provider, f"key-{key_id}", "invalid_key")
            )
        finally:
            loop.close()
    except Exception as exc:
        logger.debug("penalty_store wire skipped for %s/%s: %s", provider, key_id, exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_list(manifest: KeyManifest, args: argparse.Namespace) -> int:
    providers = (
        [args.provider] if args.provider else manifest.list_providers()
    )
    if not providers:
        print("(no keys in manifest)")
        return 0
    for name in providers:
        entries = manifest.get_keys(name)
        if not entries:
            continue
        print(f"\n[{name}]  ({len(entries)} keys)")
        for e in entries:
            age = ""
            if e.added_at:
                age = f" added {time.strftime('%Y-%m-%d', time.localtime(e.added_at))}"
            verified = ""
            if e.last_verified:
                verified = f" verified {time.strftime('%Y-%m-%d', time.localtime(e.last_verified))}"
            print(
                f"  {e.id:>4}  {e.status:<12}  {e.env_var or '-':<22}  "
                f"{e.detail or ''}{age}{verified}"
            )
    return 0


def _cmd_add(manifest: KeyManifest, args: argparse.Namespace) -> int:
    entry = manifest.add_key(
        args.provider,
        args.key_value,
        key_id=args.id,
        env_var=args.env_var or "",
        write_file=not args.no_file,
    )
    print(
        f"added {args.provider} key id={entry.id} status={entry.status} "
        f"file={entry.file_path or '(not written)'}"
    )
    return 0


def _cmd_remove(manifest: KeyManifest, args: argparse.Namespace) -> int:
    entry = manifest.remove_key(args.provider, args.id)
    if entry is None:
        print(f"error: no key id={args.id} for provider {args.provider}", file=sys.stderr)
        return 1
    print(
        f"removed (marked status=removed, file NOT deleted): "
        f"{args.provider}/{args.id} file={entry.file_path}"
    )
    active = manifest.get_active_keys(args.provider)
    if not active:
        print(
            f"warning: no active keys remain for {args.provider} — "
            "provider will fail until a working key is added",
            file=sys.stderr,
        )
    return 0


def _cmd_verify(manifest: KeyManifest, args: argparse.Namespace) -> int:
    results = asyncio.run(
        verify_manifest(
            manifest,
            provider=args.provider,
            timeout=args.timeout,
            db_path=args.db,
        )
    )
    if not results:
        print("(no keys to verify)")
        return 0
    counts: Dict[str, int] = {}
    for prov, kid, status, detail in results:
        counts[status] = counts.get(status, 0) + 1
        print(f"  {prov:<20} {kid:>4}  {status:<12}  {detail}")
    print("\nsummary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    # Exit nonzero if any dead keys found (useful for CI).
    return 1 if counts.get(DEAD, 0) > 0 else 0


def _cmd_rotate(manifest: KeyManifest, args: argparse.Namespace) -> int:
    entry = manifest.next_working_key(args.provider)
    if entry is None:
        print(
            f"error: no working key for {args.provider} "
            "(run `keys.py verify` first or add a key)",
            file=sys.stderr,
        )
        return 1
    manifest.mark_used(args.provider, entry.id)
    print(
        f"rotate {args.provider}: id={entry.id} "
        f"file={entry.file_path} env_var={entry.env_var or '-'}"
    )
    return 0


def _cmd_refresh_models(manifest: KeyManifest, args: argparse.Namespace) -> int:
    env_var, base_url = _provider_base_url(args.provider, args.db)
    if not base_url:
        print(f"error: no base_url for {args.provider}", file=sys.stderr)
        return 1
    entry = manifest.next_working_key(args.provider)
    if entry is None:
        print(f"error: no working key for {args.provider}", file=sys.stderr)
        return 1
    key_value = _read_key_file(entry.file_path)
    url = base_url.rstrip("/") + "/models"

    async def _fetch() -> Tuple[int, str]:
        import httpx

        async with httpx.AsyncClient(timeout=args.timeout) as client:
            resp = await client.get(
                url, headers={"Authorization": f"Bearer {key_value}"}
            )
            return resp.status_code, resp.text

    try:
        status, body = asyncio.run(_fetch())
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if status != 200:
        print(f"error: HTTP {status} from {url}", file=sys.stderr)
        return 1
    out_path = SECRETS_DIR / f"{args.provider}-models.json"
    out_path.write_text(body)
    print(f"wrote {len(body)} bytes to {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="keys.py", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list keys")
    p_list.add_argument("--provider", default=None)
    p_list.set_defaults(func=_cmd_list)

    p_add = sub.add_parser("add", help="add a key")
    p_add.add_argument("provider")
    p_add.add_argument("key_value")
    p_add.add_argument("--id", default=None, help="explicit key id (default: auto)")
    p_add.add_argument("--env-var", default="", help="env var name to record")
    p_add.add_argument("--no-file", action="store_true", help="don't write key file")
    p_add.set_defaults(func=_cmd_add)

    p_rem = sub.add_parser("remove", help="mark a key removed (file kept)")
    p_rem.add_argument("provider")
    p_rem.add_argument("id")
    p_rem.set_defaults(func=_cmd_remove)

    p_ver = sub.add_parser("verify", help="ping each key's /models endpoint")
    p_ver.add_argument("--provider", default=None)
    p_ver.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    p_ver.add_argument("--db", default=None, help="llm_providers.db path")
    p_ver.set_defaults(func=_cmd_verify)

    p_rot = sub.add_parser("rotate", help="print next working key (round-robin)")
    p_rot.add_argument("provider")
    p_rot.set_defaults(func=_cmd_rotate)

    p_ref = sub.add_parser("refresh-models", help="hit /v1/models, save to ~/.secrets/")
    p_ref.add_argument("provider")
    p_ref.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    p_ref.add_argument("--db", default=None)
    p_ref.set_defaults(func=_cmd_refresh_models)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = KeyManifest()
    return args.func(manifest, args)


if __name__ == "__main__":
    sys.exit(main())
