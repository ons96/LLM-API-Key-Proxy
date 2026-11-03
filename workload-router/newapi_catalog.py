"""Explicit management API and read-only provider catalog inputs for the sidecar."""

from __future__ import annotations

from dataclasses import replace
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import uuid
from typing import Callable
from urllib import error, request

from newapi_runtime import strict_json, validate_endpoint, validate_key
from provider_adapter import _NoRedirect
from quota_ledger import QuotaLedger, Route, token_count


MAX_CATALOG_BYTES = 4 * 1024 * 1024


class CatalogError(Exception):
    """Bounded errors deliberately omit upstream bodies and credential values."""


def key_fingerprint(key: str) -> str:
    """Opaque credential identity; the credential value never enters the sidecar DB."""
    return hashlib.sha256(validate_key(key).encode("ascii")).hexdigest()


def _request(opener, url: str, method: str, headers: dict, payload=None):
    body = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
    req = request.Request(url, data=body, method=method,
                          headers={"Accept": "application/json", "User-Agent": "sidecar-catalog/1.0",
                                   "Content-Type": "application/json", **headers})
    try:
        with opener.open(req, timeout=20) as response:
            raw = response.read(MAX_CATALOG_BYTES + 1)
        if len(raw) > MAX_CATALOG_BYTES:
            raise CatalogError("catalog response exceeds limit")
        return strict_json(raw)
    except error.HTTPError as exc:
        exc.close()
        raise CatalogError("catalog HTTP status " + str(exc.code)) from None
    except (OSError, error.URLError, ValueError, TypeError, RecursionError):
        raise CatalogError("catalog transport or JSON failure") from None


def items(data) -> list[dict]:
    """Support old list responses and current paginated management responses."""
    if isinstance(data, dict):
        data = data.get("items")
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise CatalogError("invalid management list")
    return data


class AdminClient:
    """Management writes require explicit calls; construction and audit are read-only."""

    def __init__(self, base_url: str, token: str, user_id: int, *,
                 key_proof: Callable[[int], str] | None = None) -> None:
        base = validate_endpoint(base_url)
        if not base.endswith("/v1"):
            raise ValueError("New-API base must end in /v1")
        self.base_url = base[:-3]
        self.token = validate_key(token)
        self.user_id = token_count(user_id, "admin_user_id")
        self.key_proof = key_proof
        if self.user_id <= 0:
            raise ValueError("invalid admin user id")
        self.opener = request.build_opener(_NoRedirect())

    def call(self, method: str, path: str, payload=None, *, headers: dict | None = None):
        if not path.startswith("/api/"):
            raise ValueError("invalid management path")
        response = _request(self.opener, self.base_url + path, method,
                            {"Authorization": "Bearer " + self.token,
                             "New-API-User": str(self.user_id), **(headers or {})}, payload)
        if response.get("success") is not True:
            raise CatalogError("management API rejected operation")
        return response.get("data")

    def channels(self) -> list[dict]:
        result = []
        seen = set()
        for page in range(1, 1001):
            data = self.call("GET", f"/api/channel/?p={page}&page_size=100")
            batch = items(data)
            for channel in batch:
                identity = channel.get("id")
                if type(identity) is not int or identity <= 0 or identity in seen:
                    raise CatalogError("invalid or repeated channel page")
                seen.add(identity)
                result.append(channel)
            total = data.get("total") if isinstance(data, dict) else None
            if type(total) is int and len(result) >= total:
                return result
            if len(batch) < 100:
                if type(total) is int and len(result) < total:
                    raise CatalogError("incomplete channel catalog")
                return result
        raise CatalogError("channel catalog exceeds limit")

    def audit(self, routes: list[Route]) -> str:
        """Verify unique aliases, single keys, fixed model mappings and zero retries."""
        if not routes:
            raise CatalogError("no configured strict routes")
        options = {row.get("key"): row.get("value") for row in items(self.call("GET", "/api/option/"))}
        if str(options.get("RetryTimes")) != "0":
            raise CatalogError("strict accounting requires RetryTimes=0")
        channels = self.channels()
        audit_rows = []
        aliases = set()
        for route in routes:
            if route.alias in aliases:
                raise CatalogError("strict aliases must be unique")
            aliases.add(route.alias)
            matches = [row for row in channels if route.alias in [value.strip() for value in str(row.get("models", "")).split(",")]]
            if len(matches) != 1 or matches[0].get("id") != route.channel_id:
                raise CatalogError("strict alias is missing or shared")
            channel = self.call("GET", "/api/channel/" + str(route.channel_id))
            if not isinstance(channel, dict) or channel.get("status") != 1 or channel.get("type") != 1:
                raise CatalogError("strict channel must be enabled and OpenAI-compatible")
            info = channel.get("channel_info")
            if not isinstance(info, dict) or info.get("is_multi_key") is not False:
                raise CatalogError("single-key channel metadata required")
            if info.get("multi_key_size", 0) not in (0, 1):
                raise CatalogError("single-key channel size required")
            if channel.get("group") != "default":
                raise CatalogError("strict channel requires the default group")
            for field in ("param_override", "header_override", "status_code_mapping"):
                if channel.get(field) not in (None, "", "{}"):
                    raise CatalogError("strict channel overrides are not supported")
            raw_mapping = channel.get("model_mapping")
            try:
                mapping = json.loads(raw_mapping) if isinstance(raw_mapping, str) else raw_mapping
            except ValueError:
                raise CatalogError("invalid private model mapping") from None
            if not isinstance(mapping, dict) or mapping.get(route.alias) != route.model_id:
                raise CatalogError("private alias mapping changed")
            key = channel.get("key")
            if not route.key_fingerprint:
                raise CatalogError("channel credential identity is not verifiable")
            if not isinstance(key, str) or not key:
                if self.key_proof is None:
                    raise CatalogError("authorized channel key verification required")
                proof = validate_key(self.key_proof(route.channel_id))
                result = self.call("POST", "/api/channel/" + str(route.channel_id) + "/key",
                                   headers={"X-Security-Proof": proof})
                key = result.get("key") if isinstance(result, dict) else None
            if not isinstance(key, str):
                raise CatalogError("channel credential identity is not verifiable")
            try:
                fingerprint = key_fingerprint(key)
            except ValueError:
                raise CatalogError("ambiguous channel credential") from None
            if not hmac.compare_digest(fingerprint, route.key_fingerprint):
                raise CatalogError("channel credential identity changed")
            if not route.channel_base_url or channel.get("base_url") != route.channel_base_url:
                raise CatalogError("channel endpoint identity changed")
            audit_rows.append({"route": route.route_id, "channel": route.channel_id,
                               "model": route.model_id, "alias": route.alias,
                               "key_fingerprint": fingerprint, "base_url": route.channel_base_url})
        return hashlib.sha256(json.dumps(audit_rows, sort_keys=True).encode()).hexdigest()

    def create_route(self, route: Route, base_url: str, key: str) -> Route:
        """Create a new private single-key channel; preserve all unrelated channels."""
        base = validate_endpoint(base_url)
        if not base.endswith("/v1"):
            raise ValueError("OpenAI-compatible provider base must end in /v1")
        validate_key(key)
        if any(route.alias in str(row.get("models", "")).split(",") for row in self.channels()):
            raise CatalogError("private alias already exists")
        channel = {"name": "sidecar:" + route.route_id, "type": 1, "key": key,
                   "base_url": base[:-3], "models": route.alias,
                   "model_mapping": json.dumps({route.alias: route.model_id}),
                   "group": "default", "groups": ["default"], "status": 1,
                   "priority": 50, "weight": 0, "auto_ban": 0}
        self.call("POST", "/api/channel/", {"mode": "single", "multi_key_mode": "random", "channel": channel})
        matches = [row for row in self.channels() if route.alias in str(row.get("models", "")).split(",")]
        if len(matches) != 1:
            raise CatalogError("created channel cannot be verified")
        return replace(route, channel_id=matches[0]["id"], key_fingerprint=key_fingerprint(key),
                       channel_base_url=base[:-3])

    def audit_relay(self, key: str, token_id: int) -> None:
        """Verify a relay token cannot invoke hidden cross-group retries."""
        identity = token_count(token_id, "relay_token_id")
        if identity <= 0:
            raise ValueError("relay token id must be positive")
        data = self.call("GET", "/api/token/" + str(identity))
        if (not isinstance(data, dict) or data.get("id") != identity or data.get("status") != 1
                or data.get("group") != "default" or data.get("cross_group_retry") is not False):
            raise CatalogError("relay token requires fixed group and no cross-group retry")
        expected = validate_key(key).removeprefix("sk-")
        actual = data.get("key")
        if not isinstance(actual, str) or "*" in actual or "..." in actual:
            # This is the documented, authorized token-key API, not a bypass of
            # New-API's protected channel-key endpoint. Old releases return it inline.
            response = self.call("POST", "/api/token/" + str(identity) + "/key")
            actual = response.get("key") if isinstance(response, dict) else None
        if not isinstance(actual, str) or not hmac.compare_digest(
            hashlib.sha256(actual.removeprefix("sk-").encode()).digest(),
            hashlib.sha256(expected.encode()).digest(),
        ):
            raise CatalogError("relay credential identity changed")

    def set_priorities(self, ordered: list[Route]) -> None:
        """Update only priorities, never keys, models or mappings."""
        self.audit(ordered)
        for index, route in enumerate(ordered):
            self.call("PUT", "/api/channel/", {"id": route.channel_id, "priority": len(ordered) - index})

    def refresh_channel(self, channel_id: int, model_ids: list[str]) -> int:
        """Add discovered models to an ordinary channel without removing its aliases."""
        identity = token_count(channel_id, "channel_id")
        if identity <= 0 or any(not valid_model_id(value) for value in model_ids):
            raise ValueError("invalid channel or model ids")
        channel = self.call("GET", "/api/channel/" + str(identity))
        if not isinstance(channel, dict) or channel.get("id") != identity:
            raise CatalogError("channel identity missing")
        if str(channel.get("name", "")).startswith("sidecar:"):
            raise CatalogError("strict channels must retain private aliases only")
        current = [value.strip() for value in str(channel.get("models", "")).split(",") if value.strip()]
        added = sorted(set(model_ids) - set(current))
        if added:
            self.call("PUT", "/api/channel/", {"id": identity, "models": ",".join(current + added)})
        return len(added)


def load_provider_metadata(database: Path, names: list[str]) -> list[dict]:
    """Read selected endpoint/key-variable metadata without opening credentials."""
    path = database.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        result = []
        for name in names:
            row = connection.execute("SELECT name, base_url, env_var, no_api_key_required FROM providers WHERE name=?",
                                     (name,)).fetchone()
            if row is None:
                raise CatalogError("provider metadata missing")
            data = dict(row)
            data["base_url"] = validate_endpoint(data["base_url"])
            result.append(data)
        return result
    finally:
        connection.close()


def valid_model_id(value: object) -> bool:
    return (isinstance(value, str) and 0 < len(value) <= 512
            and not any(char.isspace() or ord(char) < 32 or char == "," for char in value))


def refresh_provider(ledger: QuotaLedger, metadata: dict, key: str = "") -> list[dict]:
    """A failed/empty fetch preserves the last-good catalog; this sends no completions."""
    base = validate_endpoint(metadata["base_url"])
    headers = {"Authorization": "Bearer " + validate_key(key)} if key else {}
    response = _request(request.build_opener(_NoRedirect()), base + "/models", "GET", headers)
    models = items(response.get("data"))
    if not models or any(not valid_model_id(row.get("id")) for row in models):
        raise CatalogError("invalid or empty provider catalog")
    identities = [row["id"] for row in models]
    if len(set(identities)) != len(identities):
        raise CatalogError("duplicate provider model ids")
    # Whitelist catalog metadata: never persist arbitrary provider response fields.
    safe = [{name: row[name] for name in ("id", "name", "context_length", "pricing", "supported_parameters") if name in row}
            for row in models]
    with ledger.transaction() as db:
        db.execute("INSERT OR REPLACE INTO providers VALUES (?, ?, ?, ?, ?)",
                   (metadata["name"], base, metadata.get("env_var"), "unknown", time.time()))
        db.execute("UPDATE models SET active=0 WHERE provider_id=?", (metadata["name"],))
        for row in safe:
            db.execute("INSERT OR REPLACE INTO models VALUES (?, ?, ?, 1, ?)",
                       (metadata["name"], row["id"], json.dumps(row, allow_nan=False), time.time()))
    return safe


def refresh_opencode(path: Path, catalogs: dict[str, list[dict]], *, apply: bool = False) -> dict:
    """Add missing model ids only; preserve curated flags, options and the config symlink.

    Providers must already exist. The default is a preview; the caller owns the
    OpenCode boot gate before using an applied configuration for real traffic.
    """
    target = path.expanduser().resolve(strict=True)
    if not target.is_file() or target.stat().st_size > MAX_CATALOG_BYTES:
        raise CatalogError("invalid OpenCode configuration")
    lock_path = target.with_name(target.name + ".sidecar.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        end = time.monotonic() + 5
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= end:
                    raise CatalogError("OpenCode configuration is busy") from None
                time.sleep(0.05)
        original = target.read_bytes()
        config = strict_json(original)
        providers = config.get("provider")
        if not isinstance(providers, dict):
            raise CatalogError("OpenCode provider map missing")
        additions = {}
        for name, catalog in catalogs.items():
            provider = providers.get(name)
            if not isinstance(provider, dict):
                raise CatalogError("OpenCode provider must be registered before catalog refresh")
            models = provider.setdefault("models", {})
            if not isinstance(models, dict) or not catalog:
                raise CatalogError("invalid OpenCode model map or empty catalog")
            added = []
            for row in catalog:
                identity = row.get("id") if isinstance(row, dict) else None
                if not valid_model_id(identity):
                    raise CatalogError("invalid catalog model id")
                if identity not in models:
                    # No guessed context limits, pricing or tool/reasoning support.
                    models[identity] = {"name": identity}
                    added.append(identity)
            additions[name] = added
        summary = {"added": additions, "applied": False, "backup": None}
        if not apply or not any(additions.values()):
            return summary
        encoded = (json.dumps(config, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()
        backup = target.with_name(target.name + ".bak.sidecar-" + uuid.uuid4().hex)
        with backup.open("xb") as stream:
            os.chmod(backup, 0o600)
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".sidecar-config-", delete=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
            replacement = Path(stream.name)
        if target.read_bytes() != original:
            raise CatalogError("OpenCode configuration changed during refresh")
        os.replace(replacement, target)
        summary.update(applied=True, backup=str(backup))
        return summary
    finally:
        os.close(descriptor)
