#!/usr/bin/env python3
"""Operator-controlled channel-key proof helper for the New-API sidecar.

Contract (see docs/channel-proof-helper.md):

    stdin  : exactly one line, the numeric New-API channel id
    stdout : exactly one proof token, nothing else
    stderr : bounded diagnostics, never a credential or the proof
    exit   : 0 on success, 1 on any failure (the sidecar treats non-zero as fatal)

Credentials never arrive in argv. The admin token and user id are read from the
environment of an operator-owned env file, and the 2FA code comes either from a
0600 TOTP seed (unattended) or from a terminal prompt (attended).

The helper returns the proof without using it. The sidecar redeems it against
`POST /api/channel/<id>/key`, so consuming it here would break that call.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import hmac
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

TIMEOUT_SECONDS = 10.0
CODE_PERIOD_SECONDS = 30
CODE_DIGITS = 6
MAX_LINE_BYTES = 64
MAX_SECRET_BYTES = 4096
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class ProofError(RuntimeError):
    """A bounded, non-secret reason the proof could not be produced."""


def read_channel_id(stream) -> int:
    """Read and validate the single channel-id line the sidecar writes."""
    line = stream.readline(MAX_LINE_BYTES)
    if not line or not line.endswith("\n"):
        raise ProofError("expected one channel id line on stdin")
    text = line.strip()
    if not text.isdigit() or int(text) <= 0:
        raise ProofError("channel id must be a positive integer")
    if stream.readline(1):
        raise ProofError("unexpected trailing input on stdin")
    return int(text)


def load_totp_secret(path: Path) -> bytes:
    """Load a base32 TOTP seed, refusing anything not private to the operator."""
    try:
        info = path.stat()
    except OSError:
        raise ProofError("TOTP seed file is unreadable") from None
    if not path.is_file():
        raise ProofError("TOTP seed file must be a regular file")
    if info.st_mode & 0o077:
        raise ProofError("TOTP seed file must not be group or world accessible")
    try:
        raw = path.read_bytes()[:MAX_SECRET_BYTES].decode("utf-8")
    except (OSError, UnicodeDecodeError):
        raise ProofError("TOTP seed file is not valid UTF-8") from None
    text = raw.strip().replace(" ", "").upper()
    if not text:
        raise ProofError("TOTP seed file is empty")
    try:
        return base64.b32decode(text + "=" * (-len(text) % 8), casefold=True)
    except (ValueError, TypeError):
        raise ProofError("TOTP seed is not valid base32") from None


def totp_code(secret: bytes, now: float | None = None) -> str:
    """RFC 6238 code for the current 30-second window."""
    counter = int((time.time() if now is None else now) // CODE_PERIOD_SECONDS)
    digest = hmac.new(secret, struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    value = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return str(value % (10 ** CODE_DIGITS)).zfill(CODE_DIGITS)


def prompt_code() -> str:
    """Read a 2FA or backup code from the controlling terminal, never echoed."""
    try:
        code = getpass.getpass("New-API 2FA code: ").strip()
    except (EOFError, OSError):
        raise ProofError("no terminal available for an interactive code") from None
    if not code:
        raise ProofError("no 2FA code provided; use --mode=totp when unattended")
    return code


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        raise ProofError("management API redirect refused")


def _post(url: str, payload: dict, headers: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", **headers},
    )
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read(65536)
    except ProofError:
        raise
    except urllib.error.HTTPError as exc:
        raise ProofError("management API returned HTTP " + str(exc.code)) from None
    except (urllib.error.URLError, OSError):
        raise ProofError("management API is unreachable") from None
    try:
        body = json.loads(raw)
    except ValueError:
        raise ProofError("management API returned invalid JSON") from None
    if not isinstance(body, dict) or body.get("success") is not True:
        raise ProofError("management API rejected the verification")
    data = body.get("data")
    return data if isinstance(data, dict) else {}


def loopback_base(base: str) -> str:
    """Reject any endpoint that would send the admin token off-host."""
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ProofError("base URL must be an absolute http(s) URL")
    if parsed.hostname not in LOOPBACK_HOSTS:
        raise ProofError("base URL must stay on loopback")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ProofError("base URL must not carry a path or query")
    return base.rstrip("/")


def request_proof(base_url: str, token: str, user_id: int, channel_id: int, code: str) -> str:
    data = _post(
        base_url + "/api/verify",
        {
            "method": "2fa",
            "scope": "channel.key.read",
            "context": {"channel_id": channel_id},
            "code": code,
        },
        {"Authorization": "Bearer " + token, "New-API-User": str(user_id)},
    )
    proof = data.get("proof_token")
    if not isinstance(proof, str):
        raise ProofError("verification did not return a proof token")
    proof = proof.strip()
    if not proof:
        raise ProofError("verification returned an empty proof token")
    if "\n" in proof or "\r" in proof or "\x00" in proof:
        raise ProofError("proof token contains an invalid character")
    if len(proof) > 512:
        raise ProofError("proof token is implausibly long")
    return proof


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="New-API channel-key proof helper")
    parser.add_argument("--mode", choices=("interactive", "totp"), default="interactive")
    parser.add_argument(
        "--totp-secret-file",
        default=os.environ.get("NEWAPI_PROOF_TOTP_SECRET_FILE", ""),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("NEWAPI_PROOF_BASE_URL", "http://127.0.0.1:8001"),
    )
    parser.add_argument(
        "--admin-user-id",
        default=os.environ.get("NEWAPI_PROOF_ADMIN_USER_ID", ""),
    )
    args = parser.parse_args(argv)

    try:
        channel_id = read_channel_id(sys.stdin)
        token = os.environ.get("NEWAPI_PROOF_ADMIN_TOKEN", "").strip()
        if not token:
            raise ProofError("NEWAPI_PROOF_ADMIN_TOKEN is not set")
        if not str(args.admin_user_id).isdigit() or int(args.admin_user_id) <= 0:
            raise ProofError("--admin-user-id or NEWAPI_PROOF_ADMIN_USER_ID is required")
        base_url = loopback_base(args.base_url)
        if args.mode == "totp":
            if not args.totp_secret_file:
                raise ProofError("--totp-secret-file is required for totp mode")
            code = totp_code(load_totp_secret(Path(args.totp_secret_file)))
        else:
            code = prompt_code()
        proof = request_proof(base_url, token, int(args.admin_user_id), channel_id, code)
    except ProofError as exc:
        print("channel proof failed: " + str(exc), file=sys.stderr)
        return 1

    sys.stdout.write(proof + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
