"""Provider-free startup and resource smoke test for the VPS target."""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request


def request(url: str, method: str = "GET", body: bytes | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method)
    with urllib.request.urlopen(req, timeout=2) as response:
        return response.status, response.read()


def main() -> int:
    port = 8765
    env = os.environ.copy()
    env["ROUTER_PORT"] = str(port)
    process = subprocess.Popen(
        [sys.executable, "router_server.py"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                status, body = request(f"http://127.0.0.1:{port}/healthz")
                if status == 200:
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(0.05)
        else:
            raise RuntimeError("router did not become ready")

        route_body = json.dumps({
            "model": "auto",
            "messages": [{"role": "user", "content": "answer briefly"}],
            "session_id": "resource-smoke",
        }).encode()
        status, body = request(
            f"http://127.0.0.1:{port}/v1/router/route", "POST", route_body
        )
        if status != 200 or json.loads(body)["deployment"] != "general":
            raise RuntimeError("route smoke check failed")

        with open(f"/proc/{process.pid}/status", encoding="utf-8") as status_file:
            rss_kb = int(status_file.read().split("VmRSS:", 1)[1].split()[0])
        print(json.dumps({"health": 200, "route": 200, "rss_kb": rss_kb}))
        return 0
    finally:
        process.terminate()
        process.wait(timeout=3)


if __name__ == "__main__":
    raise SystemExit(main())
