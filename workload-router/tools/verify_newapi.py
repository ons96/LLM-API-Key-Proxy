"""Exercise an unmodified New-API binary against local scripted providers.

The command enters a fresh network namespace. Artifacts stay in an owner-only directory;
no existing gateway, provider credential or production database is accessed.
This is gateway integration evidence, not a live-provider availability test.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from http.client import HTTPConnection
from http.cookiejar import CookieJar
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
import hmac
import os
from pathlib import Path
import secrets
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


def enroll_local_proof(admin: AdminClient, password: str):
    """Use the installed account's authorized password->2FA enrollment flow."""
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
        proof = admin.call("POST", "/api/verify", {"method": "2fa", "scope": "channel.key.read",
                                                  "context": {"channel_id": channel_id}, "code": code})
        if not isinstance(proof, dict) or not isinstance(proof.get("proof_token"), str):
            raise CatalogError("local channel verification proof missing")
        return proof["proof_token"]
    admin.key_proof = channel_proof


def relay_token(admin: AdminClient, aliases: list[str]) -> tuple[int, str]:
    name = "proof-" + secrets.token_hex(4)
    admin.call("POST", "/api/token/", {"name": name, "remain_quota": 1000000,
                "expired_time": -1, "unlimited_quota": True, "model_limits_enabled": True,
                "model_limits": ",".join(aliases), "group": "default", "cross_group_retry": False})
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


def chat(server: Server, key: str, *, model: str = "auto-chat", stream: bool = False) -> dict:
    connection = HTTPConnection(*server.server_address, timeout=5)
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


def verify(binary: Path, parent: Path, parent_inode: int) -> dict:
    private_namespace(parent_inode)
    if not binary.is_file() or not os.access(binary, os.X_OK) or not parent.is_dir():
        raise ValueError("executable binary and existing artifact parent required")
    directory = Path(tempfile.mkdtemp(prefix="newapi-proof-", dir=parent))
    report = {"artifact_dir": str(directory), "upstreams": "scripted-local-only", "live_provider_test": False}
    calls = []
    faults = set()
    grading = [False]

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
            self.send_json(200, {"data": [{"id": name + "-model", "object": "model"}]})

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
                event = {"id": "proof", "object": "chat.completion.chunk", "model": model,
                         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage}
                self.wfile.write(("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode())
                return
            self.send_json(200, {"id": "proof", "object": "chat.completion", "model": model,
                                  "choices": [{"index": 0, "message": message,
                                               "finish_reason": "stop"}], "usage": usage})

        def log_message(self, *_args):
            return

    base = "http://127.0.0.1:8002"
    env = {"PATH": os.environ["PATH"], "HOME": str(directory), "LANG": "C.UTF-8",
           "SQLITE_PATH": str(directory / "one-api.db"), "PORT": "8002",
           "SESSION_SECRET": secrets.token_urlsafe(32), "CRYPTO_SECRET": secrets.token_urlsafe(32),
           "GOMEMLIMIT": "160MiB", "GOMAXPROCS": "2"}
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
            enroll_local_proof(admin, password)
            admin.call("PUT", "/api/option/", {"key": "RetryTimes", "value": "0"})
            ledger = QuotaLedger(directory / "sidecar.db")
            routes = []
            catalogs = {}
            configuration = {"provider": {name: {"models": {"curated": {"tool_call": False}}}
                                           for name in ("one", "two", "three")}}
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
                report["chat"] = chat(server, client_key)
                report["coding"] = chat(server, client_key, model="auto-coding")
                report["stream"] = chat(server, client_key, stream=True)
                faults.add("one-model")
                before = len(calls)
                report["failover"] = chat(server, client_key)
                report["fault_attempts"] = calls[before:]
                if report["fault_attempts"] != ["one-model", "two-model"]:
                    raise CatalogError("hidden retry or incorrect fallback order detected")
            faults.clear()
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
    parser.add_argument("--namespace-parent-inode", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.namespace_parent_inode is None:
        inode = os.stat("/proc/self/ns/net").st_ino
        sys.exit(subprocess.call(["unshare", "--user", "--map-root-user", "--net",
                                  sys.executable, *sys.argv, "--namespace-parent-inode", str(inode)]))
    try:
        print(json.dumps(verify(args.binary.resolve(), args.artifacts.resolve(), args.namespace_parent_inode), indent=2))
    except (CatalogError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}))
        sys.exit(1)
