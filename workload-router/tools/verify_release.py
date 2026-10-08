"""Run the provider-free checks required before a release or VPS copy."""

import subprocess
import sys


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> int:
    run([sys.executable, "-m", "pytest", "-q"])
    run([sys.executable, "tools/validate_session.py"])
    run([sys.executable, "tools/resource_smoke.py"])
    run([sys.executable, "-m", "py_compile", "router_core.py", "router_config.py",
         "router_state.py", "provider_adapter.py", "router_server.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
