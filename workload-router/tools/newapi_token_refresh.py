#!/usr/bin/env python3
"""Mint and install a New-API admin token for the sidecar.

The sidecar's startup audit needs an admin token. Short-lived login JWTs expire,
which is why the canary previously could not survive a reboot. This helper logs
in with the operator's root password (read from New-API's own env file inside
the host), upgrades the session to a durable access token when the two-factor
flow allows it, and writes it into the sidecar's env file with the same
permissions and a pre-change backup.

Secrecy rules: the password, the login JWT, the security proof and the minted
token never appear in stdout, stderr, argv or the repository. Only bounded
status lines are printed.

Rate-limit rule: New-API throttles this flow, so every run makes at most one
attempt and never retries a 401/403/429.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import shutil
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

TIMEOUT_SECONDS = 10.0
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
ADMIN_KEY = "NEWAPI_ADMIN_TOKEN"
PROOF_KEY = "NEWAPI_PROOF_ADMIN_TOKEN"
PASSWORD_KEY = "ROOT_PASSWORD"
MAX_BODY_BYTES = 65536


class RefreshError(RuntimeError):
    """A bounded, non-secret reason the refresh could not complete."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        raise RefreshError("management API redirect refused")


def _request(method: str, url: str, payload: dict | None = None,
             token: str = "", user_id: int | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    if user_id:
        headers["New-API-User"] = str(user_id)
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_BODY_BYTES)
    except RefreshError:
        raise
    except urllib.error.HTTPError as exc:
        raise RefreshError("management API returned HTTP " + str(exc.code)) from None
    except (urllib.error.URLError, OSError):
        raise RefreshError("management API is unreachable") from None
    try:
        body = json.loads(raw)
    except ValueError:
        raise RefreshError("management API returned invalid JSON") from None
    if not isinstance(body, dict):
        raise RefreshError("management API returned an unexpected body")
    return body


def parse_env(text: str) -> dict[str, str]:
    """Read simple KEY=VALUE lines; quotes are stripped, comments ignored."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        if not key or not all(char.isalnum() or char == "_" for char in key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def update_env_text(text: str, updates: dict[str, str]) -> str:
    """Replace the named keys in place, appending any that are absent."""
    lines = text.splitlines()
    seen: set[str] = set()
    for index, line in enumerate(lines):
        if line.lstrip().startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key in updates and key not in seen:
            lines[index] = key + "=" + updates[key]
            seen.add(key)
    for key, value in updates.items():
        if key not in seen:
            lines.append(key + "=" + value)
    return "\n".join(lines) + "\n"


def write_private(path: Path, text: str, mode: int = 0o600) -> None:
    """Replace a private file atomically, keeping restrictive permissions."""
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def loopback_base(base: str) -> str:
    """Refuse any endpoint that would send credentials off-host."""
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RefreshError("base URL must be an absolute http(s) URL")
    if parsed.hostname not in LOOPBACK_HOSTS:
        raise RefreshError("base URL must stay on loopback")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise RefreshError("base URL must not carry a path or query")
    return base.rstrip("/")


def totp_code(secret: bytes, now: float | None = None) -> str:
    counter = int((time.time() if now is None else now) // 30)
    digest = hmac.new(secret, struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    value = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return str(value % (10 ** 6)).zfill(6)


def load_seed(path: Path) -> bytes:
    try:
        info = path.stat()
    except OSError:
        raise RefreshError("TOTP seed file is unreadable") from None
    if not path.is_file() or info.st_mode & 0o077:
        raise RefreshError("TOTP seed file must be a private regular file")
    try:
        text = path.read_bytes()[:4096].decode("utf-8").strip().replace(" ", "").upper()
    except (OSError, UnicodeDecodeError):
        raise RefreshError("TOTP seed file is not valid UTF-8") from None
    if not text:
        raise RefreshError("TOTP seed file is empty")
    try:
        return base64.b32decode(text + "=" * (-len(text) % 8), casefold=True)
    except (ValueError, TypeError):
        raise RefreshError("TOTP seed is not valid base32") from None


def login(base: str, username: str, password: str) -> dict:
    """Return a session token, or the flow token a 2FA code must be sent with."""
    body = _request("POST", base + "/api/user/login",
                    {"username": username, "password": password})
    if body.get("success") is not True:
        message = str(body.get("message") or "")[:120]
        raise RefreshError("login rejected" + ((": " + message) if message else ""))
    data = body.get("data")
    if not isinstance(data, dict):
        raise RefreshError("login returned no session data")
    token = data.get("access_token") or data.get("token") or ""
    user = data.get("user")
    identity = user.get("id") if isinstance(user, dict) else data.get("id")
    flow = data.get("flow_token") if data.get("require_verification") else ""
    if not isinstance(token, str) or not token:
        if not isinstance(flow, str) or not flow:
            raise RefreshError("login returned neither a session token nor a verification flow")
        return {"token": "", "user_id": 0, "flow_token": flow}
    if type(identity) is not int or identity <= 0:
        raise RefreshError("login did not report the administrator id")
    return {"token": token, "user_id": identity, "flow_token": ""}


def complete_2fa(base: str, flow_token: str, code: str) -> tuple[str, int]:
    """Redeem the single-use login flow with a two-factor code."""
    body = _request("POST", base + "/api/user/login/2fa",
                    {"flow_token": flow_token, "code": code})
    if body.get("success") is not True:
        message = str(body.get("message") or "")[:120]
        raise RefreshError("two-factor login rejected" + ((": " + message) if message else ""))
    data = body.get("data")
    if not isinstance(data, dict):
        raise RefreshError("two-factor login returned no session data")
    token = data.get("access_token") or data.get("token")
    user = data.get("user")
    identity = user.get("id") if isinstance(user, dict) else data.get("id")
    if not isinstance(token, str) or not token:
        raise RefreshError("two-factor login did not issue a session token")
    if type(identity) is not int or identity <= 0:
        raise RefreshError("two-factor login did not report the administrator id")
    return token, identity


def next_window_delay(now: float | None = None) -> float:
    """Seconds to wait so a second TOTP code is not a reuse of the first."""
    moment = time.time() if now is None else now
    return 30 - (moment % 30) + 1


def durable_token(base: str, session: str, user_id: int, code: str) -> str:
    """Exchange the 2FA code for a never-expiring access token."""
    proof_body = _request("POST", base + "/api/verify", {
        "method": "2fa",
        "scope": "access_token.generate",
        "context": {"scopes": {}, "expires_at": 0},
        "code": code,
    }, token=session, user_id=user_id)
    if proof_body.get("success") is not True:
        raise RefreshError("two-factor verification rejected")
    data = proof_body.get("data")
    proof = data.get("proof_token") if isinstance(data, dict) else None
    if not isinstance(proof, str) or not proof.strip():
        raise RefreshError("two-factor verification returned no proof")

    minted = _request("POST", base + "/api/user/access_tokens", {
        "name": "sidecar-audit",
        "expires_at": 0,
        "scopes": {},
    }, token=session, user_id=user_id)
    if minted.get("success") is not True:
        raise RefreshError("access-token creation rejected")
    payload = minted.get("data")
    token = payload if isinstance(payload, str) else None
    if token is None and isinstance(payload, dict):
        for field in ("token", "access_token", "key"):
            if isinstance(payload.get(field), str):
                token = payload[field]
                break
    if not isinstance(token, str) or not token:
        raise RefreshError("access-token creation returned no token")
    return token


def token_is_live(base: str, token: str, user_id: int) -> bool:
    try:
        body = _request("GET", base + "/api/option/", token=token, user_id=user_id)
    except RefreshError:
        return False
    return body.get("success") is True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh the sidecar admin token")
    parser.add_argument("--base-url", default=os.environ.get("NEWAPI_PROOF_BASE_URL",
                                                             "http://127.0.0.1:8001"))
    parser.add_argument("--newapi-env", default="/opt/new-api/new-api.env")
    parser.add_argument("--target-env", default="/etc/newapi-sidecar/sidecar.env")
    parser.add_argument("--totp-seed", default=os.environ.get("NEWAPI_PROOF_TOTP_SECRET_FILE", ""))
    parser.add_argument("--username", default="root")
    parser.add_argument("--durable", action="store_true",
                        help="exchange a TOTP code for a never-expiring token")
    parser.add_argument("--apply", action="store_true", help="write the target env file")
    args = parser.parse_args(argv)

    try:
        base = loopback_base(args.base_url)
        source = Path(args.newapi_env)
        try:
            values = parse_env(source.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            raise RefreshError("New-API env file is unreadable") from None
        password = values.get(PASSWORD_KEY, "").strip()
        if not password:
            names = ", ".join(sorted(values)) or "none"
            raise RefreshError("root password not found; keys present: " + names)
        session = login(base, args.username, password)
        token, user_id = session["token"], session["user_id"]
        source_label = "login-session"
        if not token:
            if not args.totp_seed:
                raise RefreshError("--totp-seed is required to complete two-factor login")
            token, user_id = complete_2fa(base, session["flow_token"],
                                          totp_code(load_seed(Path(args.totp_seed))))
            source_label = "2fa-login-session"
        if args.durable:
            if not args.totp_seed:
                raise RefreshError("--totp-seed is required for --durable")
            if source_label != "login-session":
                time.sleep(next_window_delay())
            try:
                token = durable_token(base, token, user_id,
                                      totp_code(load_seed(Path(args.totp_seed))))
                source_label = "durable-access-token"
            except RefreshError as exc:
                print("durable token unavailable (%s); keeping the session token" % exc,
                      file=sys.stderr)
        if not token_is_live(base, token, user_id):
            raise RefreshError("minted token did not authenticate")
    except RefreshError as exc:
        print("token refresh failed: " + str(exc), file=sys.stderr)
        return 1

    print("login ok user=%d token_source=%s" % (user_id, source_label))
    if not args.apply:
        print("check only; pass --apply to install into " + args.target_env)
        return 0

    target = Path(args.target_env)
    try:
        original = target.read_text(encoding="utf-8")
        info = target.stat()
    except OSError:
        print("token refresh failed: target env file is unreadable", file=sys.stderr)
        return 1
    backup = target.with_name(target.name + ".bak-token-" + time.strftime("%Y%m%dT%H%M%SZ"))
    shutil.copy2(target, backup)
    os.chmod(backup, info.st_mode & 0o777)
    updated = update_env_text(original, {ADMIN_KEY: token, PROOF_KEY: token})
    write_private(target, updated, info.st_mode & 0o777)
    print("installed into %s (backup %s)" % (target, backup.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
