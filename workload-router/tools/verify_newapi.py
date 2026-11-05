"""Exercise an unmodified New-API binary against local scripted providers.

The command enters a fresh network namespace. Artifacts stay in an owner-only directory;
no existing gateway, provider credential or production database is accessed.
This is gateway integration evidence, not a live-provider availability test.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from http.client import HTTPConnection
from http.cookiejar import CookieJar
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
import hmac
import os
from pathlib import Path
import secrets
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from urllib import error, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from newapi_catalog import AdminClient, CatalogError, _request, items, refresh_opencode, refresh_provider
from newapi_benchmark import CASES, benchmark
from newapi_runtime import Runtime
from newapi_server import Server
from provider_adapter import _NoRedirect
from quota_ledger import QuotaLedger, Route
from router_state import RouterState


@contextmanager
def running(server):
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def private_namespace(parent_inode: int) -> None:
    """The binary listens on all interfaces; require a loopback-only namespace."""
    links = json.loads(subprocess.check_output(["ip", "-json", "link", "show"]))
    if [link["ifname"] for link in links] != ["lo"]:
        raise ValueError("run this check in a fresh unshare --net namespace")
    if parent_inode <= 0 or os.stat("/proc/self/ns/net").st_ino == parent_inode:
        raise ValueError("network namespace isolation is required")
    subprocess.run(["ip", "link", "set", "lo", "up"], check=True)


def bootstrap(base: str):
    opener = request.build_opener(request.ProxyHandler({}), request.HTTPCookieProcessor(CookieJar()), _NoRedirect())
    setup = _request(opener, base + "/api/setup", "GET", {})
    data = setup.get("data", {})
    if setup.get("success") is not True or data.get("status") is not False or data.get("root_init"):
        raise CatalogError("fresh uninitialized New-API state required")
    password = secrets.token_urlsafe(32)
    setup = _request(opener, base + "/api/setup", "POST", {}, {
        "username": "sidecarproof", "password": password, "confirmPassword": password,
        "SelfUseModeEnabled": True, "DemoSiteEnabled": False,
    })
    if setup.get("success") is not True:
        raise CatalogError("local setup rejected")
    login = _request(opener, base + "/api/user/login", "POST", {},
                     {"username": "sidecarproof", "password": password})
    if login.get("success") is not True or not isinstance(login.get("data"), dict):
        raise CatalogError("local login rejected")
    data = login["data"]
    user = data.get("user", data)
    identity = user.get("id") if isinstance(user, dict) else None
    if type(identity) is not int or identity <= 0:
        raise CatalogError("local administrator identity missing")
    token = data.get("access_token")
    if not isinstance(token, str) or not token:
        generated = _request(opener, base + "/api/user/token", "GET", {"New-API-User": str(identity)})
        if generated.get("success") is not True or not isinstance(generated.get("data"), str):
            raise CatalogError("installed release does not expose a usable admin token")
        token = generated["data"]
    admin = AdminClient(base + "/v1", token, identity)
    return admin, password


def totp(secret: str, timestamp: int, digits: int = 6) -> str:
    """RFC 6238 SHA-1; used only for the generated disposable test account."""
    key = base64.b32decode(secret + "=" * (-len(secret) % 8))
    digest = hmac.new(key, (timestamp // 30).to_bytes(8, "big"), hashlib.sha1).digest()
    offset = digest[-1] & 15
    number = int.from_bytes(digest[offset:offset + 4], "big") & 0x7fffffff
    return str(number % (10 ** digits)).zfill(digits)


def enroll_local_proof(admin: AdminClient, password: str) -> str:
    """Enroll the disposable account and return its seed only to the fixture."""
    proof = admin.call("POST", "/api/verify", {"method": "password", "scope": "2fa.setup", "password": password})
    if not isinstance(proof, dict) or not isinstance(proof.get("proof_token"), str):
        raise CatalogError("local enrollment proof missing")
    setup = admin.call("POST", "/api/user/2fa/setup", {}, headers={"X-Security-Proof": proof["proof_token"]})
    if not isinstance(setup, dict) or not isinstance(setup.get("secret"), str):
        raise CatalogError("local two-factor enrollment unavailable")
    secret = setup["secret"]
    result = admin.call("POST", "/api/user/2fa/enable",
                        {"flow_token": setup["flow_token"], "code": totp(secret, int(time.time()))})
    if not isinstance(result, dict) or not isinstance(result.get("access_token"), str):
        raise CatalogError("local enrollment did not return renewed authentication")
    admin.token = result["access_token"]
    # Backup codes are issued by the enrollment API and can be consumed once.
    # They avoid OTP reuse in the same time step while auditing several channels.
    codes = iter(setup.get("backup_codes", []))

    def channel_proof(channel_id):
        try:
            code = next(codes)
        except StopIteration:
            raise CatalogError("local proof codes exhausted") from None
        proof = admin.call(
            "POST",
            "/api/verify",
            {
                "method": "2fa",
                "scope": "channel.key.read",
                "context": {"channel_id": channel_id},
                "code": code,
            },
        )
        if not isinstance(proof, dict) or not isinstance(proof.get("proof_token"), str):
            raise CatalogError("local channel verification proof missing")
        return proof["proof_token"]
    admin.key_proof = channel_proof
    return secret


@contextmanager
def entrypoint_process(
    directory: Path, env: dict[str, str], name: str
) -> Iterator[subprocess.Popen[bytes]]:
    """Run the unchanged production entrypoint; keep all diagnostics private."""
    entrypoint = Path(__file__).resolve().parents[1] / "newapi_server.py"
    with (directory / (name + ".log")).open("xb") as log:
        process = subprocess.Popen(
            [sys.executable, str(entrypoint)],
            cwd=directory,
            env=env,
            stdout=log,
            stderr=log,
            umask=0o077,
        )
        try:
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)


def get_json(address: tuple[str, int], path: str, key: str = "") -> tuple[int, dict]:
    """Read a bounded, prompt-free fixture endpoint without logging its body."""
    connection = HTTPConnection(*address, timeout=0.2)
    try:
        headers = {"Authorization": "Bearer " + key} if key else {}
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        raw = response.read(65537)
        if len(raw) > 65536:
            raise CatalogError("sidecar probe response exceeds limit")
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise CatalogError("sidecar probe response must be an object")
        return response.status, body
    finally:
        connection.close()


def wait_entrypoint(
    process: subprocess.Popen[bytes], address: tuple[str, int], *, fail_closed: bool
) -> None:
    """Reject any listening socket before a failed audit, and bound boot time."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            status, body = get_json(address, "/healthz")
        except OSError:
            pass
        else:
            if fail_closed:
                raise CatalogError("sidecar listened before successful authorization")
            if status != 200 or body.get("service") != "newapi-sidecar":
                raise CatalogError("unexpected sidecar health response")
            return
        result = process.poll()
        if result is not None:
            if fail_closed and result != 0:
                return
            raise CatalogError("sidecar entrypoint exited unexpectedly")
        time.sleep(0.05)
    raise CatalogError("sidecar entrypoint startup timed out")


def verify_entrypoint(
    directory: Path, env: dict[str, str], calls: list[str], ledger: QuotaLedger
) -> dict:
    """Prove helper authorization, actual boot, graceful stop and durable restart."""
    address = ("127.0.0.1", int(env["SIDECAR_PORT"]))
    before = len(calls)
    result: dict = {}
    missing = {
        name: value
        for name, value in env.items()
        if name != "NEWAPI_CHANNEL_KEY_PROOF_COMMAND"
    }
    invalid = {
        **env,
        "NEWAPI_CHANNEL_KEY_PROOF_COMMAND": shlex.join(
            [sys.executable, "-c", "print('invalid-proof')"]
        ),
    }
    for name, environment in (("missing-helper", missing), ("invalid-helper", invalid)):
        with entrypoint_process(directory, environment, name) as process:
            wait_entrypoint(process, address, fail_closed=True)
        failure = (directory / (name + ".log")).read_text()
        expected = (
            "authorized channel key verification required"
            if name == "missing-helper"
            else "catalog HTTP status 403"
        )
        if expected not in failure:
            raise CatalogError("entrypoint failed for an unexpected reason")
        result[name.replace("-", "_") + "_fail_closed"] = True
    if len(calls) != before:
        raise CatalogError("upstream dispatch occurred before authorization")
    result["unauthorized_upstream_calls"] = 0

    persisted = None
    quota_snapshot = None
    for name in ("startup", "restart"):
        window = int(time.time()) // 30
        with entrypoint_process(directory, env, name) as process:
            wait_entrypoint(process, address, fail_closed=False)
            same_window = window == int(time.time()) // 30
            if get_json(address, "/v1/models")[0] != 401:
                raise CatalogError("entrypoint did not require client authentication")
            status, diagnostics = get_json(
                address, "/v1/router/diagnostics", env["SIDECAR_API_KEY"]
            )
            if status != 200:
                raise CatalogError("entrypoint diagnostics unavailable")
            if persisted is not None and diagnostics != persisted:
                raise CatalogError(
                    "accounting or cooldown state did not survive restart"
                )
            current_quota = list(
                ledger.db.execute(
                    "SELECT window_type, COUNT(*), SUM(amount) FROM quota_windows "
                    "GROUP BY window_type ORDER BY window_type"
                )
            )
            if quota_snapshot is not None and current_quota != quota_snapshot:
                raise CatalogError("quota reservations did not survive restart")
            checks: dict = {
                "health": 200,
                "unauthenticated_models": 401,
                "totp_channels": 3,
                "same_totp_window": same_window,
            }
            checks["chat"] = chat(address, env["SIDECAR_API_KEY"])
            if name == "startup":
                checks["coding"] = chat(
                    address, env["SIDECAR_API_KEY"], model="auto-coding"
                )
                checks["stream"] = chat(address, env["SIDECAR_API_KEY"], stream=True)
            status, persisted = get_json(
                address, "/v1/router/diagnostics", env["SIDECAR_API_KEY"]
            )
            if status != 200 or sum(persisted["attempts"].values()) <= sum(
                diagnostics["attempts"].values()
            ):
                raise CatalogError("entrypoint completion was not accounted")
            checks["accounting"] = persisted["attempts"]
            checks["active_cooldowns"] = persisted["active_cooldowns"]
            quota_snapshot = list(
                ledger.db.execute(
                    "SELECT window_type, COUNT(*), SUM(amount) FROM quota_windows "
                    "GROUP BY window_type ORDER BY window_type"
                )
            )
        if process.returncode != 0:
            raise CatalogError("sidecar did not shut down cleanly on SIGTERM")
        checks["sigterm_exit"] = process.returncode
        result[name] = checks
    sensitive = [
        env[name]
        for name in ("SIDECAR_API_KEY", "NEWAPI_API_KEY", "NEWAPI_ADMIN_TOKEN")
    ]
    sensitive.append(Path(env["NEWAPI_PROOF_TOTP_SECRET_FILE"]).read_text().strip())
    for name in ("missing-helper", "invalid-helper", "startup", "restart"):
        log = (directory / (name + ".log")).read_text()
        if any(value in log for value in sensitive):
            raise CatalogError("sidecar entrypoint logged a generated credential")
    result["credentials_absent_from_logs"] = True
    result["quota_windows_persisted"] = True
    result["restart_persisted"] = True
    return result


def fixture_command(
    directory: Path,
    env: dict[str, str],
    name: str,
    argv: list[str],
    *,
    expected: int = 0,
) -> bytes:
    """Bound operator/client subprocesses and keep their output in private artifacts."""
    try:
        with (directory / (name + ".out")).open("xb") as out, (
            directory / (name + ".err")
        ).open("xb") as err:
            process = subprocess.run(
                argv,
                cwd=directory,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                timeout=45,
                umask=0o077,
            )
    except (OSError, subprocess.SubprocessError):
        raise CatalogError("fixture command could not complete: " + name) from None
    if process.returncode != expected:
        raise CatalogError("fixture command failed: " + name)
    stdout = directory / (name + ".out")
    stderr = directory / (name + ".err")
    if max(stdout.stat().st_size, stderr.stat().st_size) > 1024 * 1024:
        raise CatalogError("fixture command output exceeds limit")
    raw, error_output = stdout.read_bytes(), stderr.read_bytes()
    credentials = [
        env[key].encode()
        for key in ("SIDECAR_API_KEY", "NEWAPI_API_KEY", "NEWAPI_ADMIN_TOKEN")
        if env.get(key)
    ]
    if any(value in raw or value in error_output for value in credentials):
        raise CatalogError("fixture command logged a generated credential")
    return raw


def verify_operators(
    directory: Path,
    env: dict[str, str],
    admin: AdminClient,
    ledger: QuotaLedger,
    routes: list[Route],
    config: Path,
    calls: list[str],
    catalog_faults: set[str],
) -> dict:
    """Exercise operator preview/apply, catalogs and protected priority-only writes."""
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "newapi_operator.py"),
    ]
    before_calls = len(calls)
    route_file = directory / "routes.json"
    route_file.write_text(json.dumps({"routes": [asdict(route) for route in routes]}))
    metadata = directory / "providers.db"
    with sqlite3.connect(metadata) as db:
        db.execute(
            "CREATE TABLE providers (key_name, base_url, env_var, no_api_key_required)"
        )
        for route in routes:
            db.execute(
                "INSERT INTO providers VALUES (?, ?, ?, ?)",
                (route.provider_id, route.channel_base_url + "/v1", None, 1),
            )
    result = {}
    original_config = config.read_bytes()
    metadata_before = metadata.read_bytes()
    result["routes_preview"] = json.loads(
        fixture_command(
            directory,
            env,
            "routes-preview",
            command + ["routes", "--file", str(route_file)],
        )
    )
    result["routes_apply"] = json.loads(
        fixture_command(
            directory,
            env,
            "routes-apply",
            command + ["routes", "--file", str(route_file), "--apply"],
        )
    )
    # JSON route files store tuple profiles as lists.
    expected_routes = {route.route_id: asdict(route) for route in routes}
    for metadata_row in expected_routes.values():
        metadata_row["profiles"] = list(metadata_row["profiles"])
    if {route.route_id: asdict(route) for route in ledger.routes()} != expected_routes:
        raise CatalogError("operator route seeding changed configured identities")
    catalog_config = directory / "operator.stock.json"
    catalog_config.write_bytes(original_config)
    catalog_link = directory / "operator.json"
    catalog_link.symlink_to(catalog_config.name)
    catalog = command + [
        "catalog",
        "--provider-db",
        str(metadata),
        "--providers",
        "one,two,three",
        "--opencode-config",
        str(catalog_link),
    ]
    result["catalog_preview"] = json.loads(
        fixture_command(directory, env, "catalog-preview", catalog)
    )
    if (
        catalog_link.read_bytes() != original_config
        or result["catalog_preview"]["opencode"]["applied"]
    ):
        raise CatalogError("catalog preview changed OpenCode configuration")
    result["catalog_apply"] = json.loads(
        fixture_command(directory, env, "catalog-apply", catalog + ["--apply-opencode"])
    )
    if not result["catalog_apply"]["opencode"]["applied"]:
        raise CatalogError("operator catalog apply did not update configuration")
    if metadata.read_bytes() != metadata_before:
        raise CatalogError("operator wrote the provider metadata database")
    updated = json.loads(catalog_link.read_bytes())
    if not catalog_link.is_symlink():
        raise CatalogError("operator catalog refresh replaced the OpenCode symlink")
    original = json.loads(original_config)
    for name in ("one", "two", "three"):
        if (
            updated["provider"][name]["models"]["curated"]
            != original["provider"][name]["models"]["curated"]
        ):
            raise CatalogError("operator refresh overwrote curated model metadata")
        if name + "-model" not in updated["provider"][name]["models"]:
            raise CatalogError("operator refresh omitted a catalog model")
    last_good = list(
        ledger.db.execute("SELECT * FROM models ORDER BY provider_id, model_id")
    )
    applied_config = catalog_link.read_bytes()
    catalog_faults.add("one")
    try:
        fixture_command(directory, env, "catalog-empty", catalog, expected=1)
    finally:
        catalog_faults.clear()
    if (
        list(ledger.db.execute("SELECT * FROM models ORDER BY provider_id, model_id"))
        != last_good
    ):
        raise CatalogError("empty catalog replaced last-good provider metadata")
    if catalog_link.read_bytes() != applied_config:
        raise CatalogError("failed catalog changed OpenCode configuration")
    result["last_good_preserved"] = True
    result["boot_check"] = json.loads(
        fixture_command(
            directory,
            env,
            "boot-check",
            command
            + [
                "boot-check",
                "--opencode-config",
                str(catalog_link),
                "--providers",
                "one,two,three",
            ],
        )
    )
    result["missing_provider_boot_check"] = json.loads(
        fixture_command(
            directory,
            env,
            "boot-missing",
            command
            + [
                "boot-check",
                "--opencode-config",
                str(catalog_link),
                "--providers",
                "missing",
            ],
            expected=1,
        )
    )
    result["audit"] = json.loads(
        fixture_command(directory, env, "operator-audit", command + ["audit"])
    )
    channel_ids = {route.channel_id for route in routes}
    before = {row["id"]: row for row in admin.channels() if row["id"] in channel_ids}
    invalid_order = command + ["priorities", "--order", "two,missing", "--apply"]
    fixture_command(directory, env, "priority-invalid", invalid_order, expected=1)
    if {
        row["id"]: row for row in admin.channels() if row["id"] in channel_ids
    } != before:
        raise CatalogError("invalid priority order changed New-API channels")
    result["invalid_priority_order_fail_closed"] = True
    order = command + ["priorities", "--order", "two,one,three"]
    result["priority_preview"] = json.loads(
        fixture_command(directory, env, "priority-preview", order)
    )
    if {
        row["id"]: row for row in admin.channels() if row["id"] in channel_ids
    } != before:
        raise CatalogError("priority preview changed New-API channels")
    result["priority_apply"] = json.loads(
        fixture_command(directory, env, "priority-apply", order + ["--apply"])
    )
    after = {row["id"]: row for row in admin.channels() if row["id"] in channel_ids}
    for update in result["priority_apply"]["order"]:
        identity = update["channel_id"]
        if after[identity].get("priority") != update["priority"]:
            raise CatalogError("New-API priority update was not applied")
        if {
            key: value for key, value in before[identity].items() if key != "priority"
        } != {
            key: value for key, value in after[identity].items() if key != "priority"
        }:
            raise CatalogError("priority update changed another channel field")
    if len(calls) != before_calls:
        raise CatalogError("operator command dispatched an upstream completion")
    result.update(no_completion_calls=True, priorities_preserved_channel_fields=True)
    return result


def verify_opencode_boot(
    directory: Path, executable: Path, config: Path, entry_env: dict[str, str]
) -> dict:
    """Boot the installed client and complete a scripted read tool loop in isolation."""
    home = directory / "client-home"
    home.mkdir()
    models = directory / "client-models.json"
    models.write_text("{}")
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "OPENCODE_TEST_HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_DATA_HOME": str(home / ".local/share"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_STATE_HOME": str(home / ".local/state"),
        "OPENCODE_CONFIG": str(config),
        "OPENCODE_MODELS_PATH": str(models),
        "OPENCODE_DISABLE_MODELS_FETCH": "true",
        "OPENCODE_DISABLE_AUTOUPDATE": "true",
        "OPENCODE_DISABLE_DEFAULT_PLUGINS": "true",
        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "true",
        "OPENCODE_DISABLE_CLAUDE_CODE": "true",
        "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
        "OPENCODE_DISABLE_CHANNEL_DB": "true",
    }
    version = fixture_command(
        directory, env, "opencode-version", [str(executable), "--version"]
    )
    result: dict = {"version": version.decode().strip(), "providers": {}}
    for name in ("one", "two", "three"):
        output = fixture_command(
            directory,
            env,
            "opencode-models-" + name,
            [str(executable), "models", name, "--pure"],
        )
        entries = output.decode().splitlines()
        if (
            name + "/curated" not in entries
            or name + "/" + name + "-model" not in entries
        ):
            raise CatalogError("OpenCode did not boot with refreshed catalog")
        result["providers"][name] = len(entries)
    client_config = directory / "client.json"
    client_key = directory / "client.key"
    client_key.write_text(entry_env["SIDECAR_API_KEY"])
    client_key.chmod(0o600)
    configuration = json.loads(config.read_bytes())
    configuration["provider"]["sidecar"] = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Disposable sidecar",
        "options": {
            "baseURL": "http://127.0.0.1:" + entry_env["SIDECAR_PORT"] + "/v1",
            "apiKey": "{file:" + str(client_key) + "}",
        },
        "models": {
            "auto-coding": {
                "name": "Auto coding",
                "tool_call": True,
                "limit": {"context": 131072, "output": 2048},
            }
        },
    }
    configuration["permission"] = {
        "*": "deny",
        "read": {"*": "deny", "proof.txt": "allow"},
    }
    configuration["model"] = "sidecar/auto-coding"
    configuration["small_model"] = "sidecar/auto-coding"
    client_config.write_text(json.dumps(configuration))
    env["OPENCODE_CONFIG"] = str(client_config)
    env["OPENCODE_DISABLE_LSP_DOWNLOAD"] = "true"
    work = directory / "client-work"
    work.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", str(work)],
        check=True,
        env={"PATH": os.environ["PATH"], "HOME": str(home)},
    )
    (work / "proof.txt").write_text("hello-sidecar-proof\n")
    address = ("127.0.0.1", int(entry_env["SIDECAR_PORT"]))
    with entrypoint_process(directory, entry_env, "opencode-sidecar") as process:
        wait_entrypoint(process, address, fail_closed=False)
        output = fixture_command(
            directory,
            env,
            "opencode-loop",
            [
                str(executable),
                "run",
                "--pure",
                "--dir",
                str(work),
                "--format",
                "json",
                "--model",
                "sidecar/auto-coding",
                "--title",
                "local-proof",
                "Use read to inspect proof.txt and reply with its exact content.",
            ],
        )
    if process.returncode != 0:
        raise CatalogError("OpenCode sidecar did not stop cleanly")
    secret = entry_env["SIDECAR_API_KEY"].encode()
    if secret in output or secret in (directory / "opencode-loop.err").read_bytes():
        raise CatalogError("OpenCode logged its generated client credential")
    events = [json.loads(line) for line in output.splitlines() if line.strip()]
    completed = [
        event
        for event in events
        if event.get("type") == "tool_use"
        and event.get("part", {}).get("tool") == "read"
        and event["part"].get("state", {}).get("status") == "completed"
    ]
    if len(completed) != 1 or not any(
        event.get("type") == "text"
        and "hello-sidecar-proof" in event.get("part", {}).get("text", "")
        for event in events
    ):
        raise CatalogError("OpenCode did not complete the scripted read tool loop")
    result.update(network_isolated=True, tool_loops=1, read_only_permissions=True)
    return result


def relay_token(admin: AdminClient, aliases: list[str]) -> tuple[int, str]:
    name = "proof-" + secrets.token_hex(4)
    admin.call(
        "POST",
        "/api/token/",
        {
            "name": name,
            "remain_quota": 1000000,
            "expired_time": -1,
            "unlimited_quota": True,
            "model_limits_enabled": True,
            "model_limits": ",".join(aliases),
            "group": "default",
            "cross_group_retry": False,
        },
    )
    matches = [row for row in items(admin.call("GET", "/api/token/?p=1&page_size=100")) if row.get("name") == name]
    if len(matches) != 1:
        raise CatalogError("created relay token cannot be identified")
    token = matches[0]
    key = token.get("key")
    if not isinstance(key, str) or "*" in key or "..." in key:
        result = admin.call("POST", "/api/token/" + str(token["id"]) + "/key")
        key = result.get("key") if isinstance(result, dict) else None
    if not isinstance(key, str) or not key:
        raise CatalogError("created relay credential cannot be read")
    return token["id"], key


def chat(
    address: tuple[str, int],
    key: str,
    *,
    model: str = "auto-chat",
    stream: bool = False,
) -> dict:
    connection = HTTPConnection(*address, timeout=5)
    payload = {"model": model, "messages": [{"role": "user", "content": "Reply with exactly ok."}],
               "max_tokens": 32, "stream": stream}
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(payload),
                           {"Content-Type": "application/json", "Authorization": "Bearer " + key,
                            "X-Session-Id": "local-proof"})
        response = connection.getresponse()
        body = response.read().decode()
        if response.status != 200:
            raise CatalogError("sidecar completion failed with HTTP " + str(response.status))
        if stream:
            if "[DONE]" not in body or '"content":"ok"' not in body:
                raise CatalogError("sidecar stream did not complete")
        elif json.loads(body)["choices"][0]["message"]["content"] != "ok":
            raise CatalogError("sidecar completion content mismatch")
        return {"status": response.status, "decision": response.getheader("X-Router-Decision"),
                "route": response.getheader("X-Router-Deployment")}
    finally:
        connection.close()


def verify(
    binary: Path, parent: Path, parent_inode: int, opencode: Path | None = None
) -> dict:
    private_namespace(parent_inode)
    if not binary.is_file() or not os.access(binary, os.X_OK) or not parent.is_dir():
        raise ValueError("executable binary and existing artifact parent required")
    directory = Path(tempfile.mkdtemp(prefix="newapi-proof-", dir=parent))
    report: dict = {
        "artifact_dir": str(directory),
        "upstreams": "scripted-local-only",
        "live_provider_test": False,
        "passed": False,
    }
    calls = []
    faults = set()
    catalog_faults: set[str] = set()
    grading = [False]
    client_active = [False]
    client_events = []

    class Upstream(BaseHTTPRequestHandler):
        def send_json(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if status == 429:
                self.send_header("Retry-After", "120")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            name = self.path.split("/")[1]
            models = (
                []
                if name in catalog_faults
                else [{"id": name + "-model", "object": "model"}]
            )
            self.send_json(200, {"data": models})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            model = payload["model"]
            calls.append(model)
            if model in faults:
                self.send_json(429, {"error": {"message": "rate limit exceeded", "type": "rate_limit_exceeded"}})
                return
            usage = {"prompt_tokens": 12, "completion_tokens": 2}
            case = next((case for case in CASES if case.prompt == payload["messages"][0].get("content")), None)
            message = {"role": "assistant", "content": "ok"}
            if grading[0] and case is not None:
                correct = model == "two-model"
                if case.tool:
                    arguments = case.expected if correct else {"path": "wrong", "start": 0, "count": 1}
                    message = {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "proof-call", "type": "function", "function": {
                            "name": "read_file", "arguments": json.dumps(arguments)}}]}
                else:
                    message["content"] = json.dumps(case.expected if correct else {"wrong": True})
            if client_active[0]:
                tools = payload.get("tools", [])
                tool_messages = [
                    row for row in payload["messages"] if row.get("role") == "tool"
                ]
                if tool_messages:
                    if "hello-sidecar-proof" not in json.dumps(tool_messages):
                        self.send_json(
                            400, {"error": {"message": "fixture tool result missing"}}
                        )
                        return
                    message["content"] = "hello-sidecar-proof"
                    client_events.append(
                        {"step": "result", "model": model, "tools": len(tools)}
                    )
                else:
                    if not any(
                        row.get("function", {}).get("name") == "read" for row in tools
                    ):
                        self.send_json(
                            400, {"error": {"message": "fixture read tool missing"}}
                        )
                        return
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_read_proof",
                                "type": "function",
                                "function": {
                                    "name": "read",
                                    "arguments": json.dumps(
                                        {
                                            "filePath": str(
                                                directory / "client-work/proof.txt"
                                            )
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                    client_events.append(
                        {"step": "call", "model": model, "tools": len(tools)}
                    )
            if payload.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                final_delta = ({"tool_calls": [{**message["tool_calls"][0], "index": 0}]}
                               if message.get("tool_calls") else {"content": message["content"]})
                for delta in ({"role": "assistant"}, {"reasoning_content": "checking"}, final_delta):
                    event = {"id": "proof", "object": "chat.completion.chunk", "model": model,
                             "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                    self.wfile.flush()
                    time.sleep(0.005)
                event = {
                    "id": "proof",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": (
                                "tool_calls" if message.get("tool_calls") else "stop"
                            ),
                        }
                    ],
                    "usage": usage,
                }
                self.wfile.write(("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode())
                return
            self.send_json(
                200,
                {
                    "id": "proof",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [
                        {"index": 0, "message": message, "finish_reason": "stop"}
                    ],
                    "usage": usage,
                },
            )

        def log_message(self, *_args):
            return

    base = "http://127.0.0.1:8002"
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(directory),
        "LANG": "C.UTF-8",
        "SQLITE_PATH": str(directory / "one-api.db"),
        "PORT": "8002",
        "SESSION_SECRET": secrets.token_urlsafe(32),
        "CRYPTO_SECRET": secrets.token_urlsafe(32),
        "GOMEMLIMIT": "160MiB",
        "GOMAXPROCS": "2",
        # Fixture-only budget for enrollment and repeated operator/startup audits.
        "CRITICAL_RATE_LIMIT": "64",
        "GLOBAL_WEB_RATE_LIMIT": "512",
    }
    process = None
    ledger = state = None
    try:
        with (directory / "newapi.log").open("xb") as log, running(ThreadingHTTPServer(("127.0.0.1", 0), Upstream)) as upstream:
            process = subprocess.Popen([str(binary)], cwd=directory, env=env, stdout=log, stderr=log, umask=0o077)
            for _ in range(200):
                try:
                    with request.urlopen(base + "/api/setup", timeout=0.2) as response:
                        if response.status == 200:
                            break
                except (OSError, error.URLError):
                    if process.poll() is not None:
                        raise CatalogError("local New-API process exited") from None
                    time.sleep(0.05)
            else:
                raise CatalogError("local New-API startup timed out")
            admin, password = bootstrap(base)
            seed = enroll_local_proof(admin, password)
            admin.call("PUT", "/api/option/", {"key": "RetryTimes", "value": "0"})
            ledger = QuotaLedger(directory / "sidecar.db")
            routes = []
            catalogs = {}
            configuration = {
                "provider": {
                    name: {"models": {"curated": {"tool_call": False}}}
                    for name in ("one", "two", "three")
                }
            }
            target = directory / "opencode.stock.json"
            target.write_text(json.dumps(configuration))
            config_link = directory / "opencode.json"
            config_link.symlink_to(target.name)
            for index, name in enumerate(("one", "two", "three"), 1):
                endpoint = f"http://127.0.0.1:{upstream.server_address[1]}/{name}/v1"
                catalogs[name] = refresh_provider(ledger, {"name": name, "base_url": endpoint, "env_var": None})
                route = Route(name, name, name + "-model", "proof-key-" + name, "sc.proof." + name, index,
                              quota_type="paid" if name == "three" else "unlimited", context_window=131072,
                              output_limit=4096, input_price=1 if name == "three" else 0,
                              output_price=1 if name == "three" else 0, stream=True, tool_call=True,
                              structured_output=True)
                route = admin.create_route(route, endpoint, secrets.token_urlsafe(24))
                ledger.configure(route)
                routes.append(route)
            report["catalog_models"] = ledger.diagnostics()["catalog_models"]
            report["opencode_refresh"] = refresh_opencode(config_link, catalogs, apply=True)
            if not config_link.is_symlink():
                raise CatalogError("catalog refresh replaced the config symlink")
            report["audit_digest"] = admin.audit(routes)
            token_id, relay = relay_token(admin, [route.alias for route in routes])
            admin.audit_relay(relay, token_id)
            state = RouterState(directory / "sidecar.db")
            runtime = Runtime(ledger, state, base + "/v1", relay, explore=False)
            client_key = secrets.token_urlsafe(24)
            with running(Server(("127.0.0.1", 0), runtime, client_key)) as server:
                report["chat"] = chat(server.server_address, client_key)
                report["coding"] = chat(
                    server.server_address, client_key, model="auto-coding"
                )
                report["stream"] = chat(server.server_address, client_key, stream=True)
                faults.add("one-model")
                before = len(calls)
                report["failover"] = chat(server.server_address, client_key)
                report["fault_attempts"] = calls[before:]
                if report["fault_attempts"] != ["one-model", "two-model"]:
                    raise CatalogError("hidden retry or incorrect fallback order detected")
            faults.clear()
            # This seed belongs only to the newly enrolled, disposable account.
            seed_file = directory / "totp.seed"
            seed_file.write_text(seed + "\n")
            seed_file.chmod(0o600)
            helper = Path(__file__).resolve().with_name("newapi_channel_proof.py")
            entry_env = {
                **env,
                "SIDECAR_DB": str(directory / "sidecar.db"),
                "SIDECAR_HOST": "127.0.0.1",
                "SIDECAR_PORT": "8003",
                "SIDECAR_API_KEY": client_key,
                "NEWAPI_BASE_URL": base + "/v1",
                "NEWAPI_ADMIN_TOKEN": admin.token,
                "NEWAPI_ADMIN_USER_ID": str(admin.user_id),
                "NEWAPI_API_KEY": relay,
                "NEWAPI_RELAY_TOKEN_ID": str(token_id),
                "NEWAPI_CHANNEL_KEY_PROOF_COMMAND": shlex.join(
                    [sys.executable, str(helper), "--mode=totp"]
                ),
                "NEWAPI_PROOF_BASE_URL": base,
                "NEWAPI_PROOF_ADMIN_TOKEN": admin.token,
                "NEWAPI_PROOF_ADMIN_USER_ID": str(admin.user_id),
                "NEWAPI_PROOF_TOTP_SECRET_FILE": str(seed_file),
            }
            operator_config = directory / "operator-input.json"
            operator_config.write_text(json.dumps(configuration))
            report["operators"] = verify_operators(
                directory,
                entry_env,
                admin,
                ledger,
                routes,
                operator_config,
                calls,
                catalog_faults,
            )
            report["entrypoint"] = verify_entrypoint(
                directory, entry_env, calls, ledger
            )
            if opencode is not None:
                before_client = len(calls)
                accounted_before = ledger.diagnostics()["attempts"].get("success", 0)
                client_active[0] = True
                try:
                    report["opencode_boot"] = verify_opencode_boot(
                        directory, opencode, directory / "operator.json", entry_env
                    )
                finally:
                    client_active[0] = False
                if [row["step"] for row in client_events] != ["call", "result"]:
                    raise CatalogError("unexpected OpenCode tool loop dispatch count")
                if (
                    len(calls) - before_client != 2
                    or ledger.diagnostics()["attempts"].get("success", 0)
                    - accounted_before
                    != 2
                ):
                    raise CatalogError(
                        "OpenCode tool loop was not independently accounted"
                    )
                if len({row["model"] for row in client_events}) != 1:
                    raise CatalogError("OpenCode tool continuation changed deployments")
                report["opencode_boot"].update(
                    upstream_steps=client_events,
                    accounted_attempts=2,
                    stable_deployment=True,
                )
            # Fresh graded phase in the owned fixture DB: clear decisions, cooldowns
            # and quota usage so before/after ranking shows grade-driven reordering.
            # Route config and learned limits remain. Do NOT do this in production.
            with ledger.transaction() as db:
                db.execute("DELETE FROM quota_windows")
                db.execute("DELETE FROM routing_decisions")
                db.execute("DELETE FROM cooldowns")
                db.execute("DELETE FROM benchmark_runs")
            grading[0] = True
            report["benchmark"] = benchmark(runtime, routes, seed=7, stream=True, max_requests=15)
            if report["benchmark"]["metrics"]["two"]["correct"] != 5:
                raise CatalogError("graded workload did not complete")
            if not report["benchmark"]["ranking_changed"]:
                raise CatalogError("graded telemetry did not change ordering")
            report["diagnostics"] = ledger.diagnostics()
            report["paid_excluded"] = "three-model" not in calls
            if not report["paid_excluded"]:
                raise CatalogError("paid route was dispatched")
            for line in Path(f"/proc/{process.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    report["newapi_rss_kib"] = int(line.split()[1])
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    report["python_fixture_and_sidecar_rss_kib"] = int(line.split()[1])
            report["passed"] = True
    except (CatalogError, ValueError) as exc:
        report["error"] = str(exc)
    finally:
        if state is not None:
            state.close()
        if ledger is not None:
            ledger.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("/tmp/opencode"))
    parser.add_argument(
        "--opencode",
        type=Path,
        help="optional installed client for isolated boot and scripted tool loop",
    )
    parser.add_argument("--namespace-parent-inode", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.namespace_parent_inode is None:
        inode = os.stat("/proc/self/ns/net").st_ino
        sys.exit(subprocess.call(["unshare", "--user", "--map-root-user", "--net",
                                  sys.executable, *sys.argv, "--namespace-parent-inode", str(inode)]))
    try:
        report = verify(
            args.binary.resolve(),
            args.artifacts.resolve(),
            args.namespace_parent_inode,
            args.opencode.resolve() if args.opencode else None,
        )
        print(json.dumps(report, indent=2))
        sys.exit(0 if report["passed"] else 1)
    except (CatalogError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}))
        sys.exit(1)
