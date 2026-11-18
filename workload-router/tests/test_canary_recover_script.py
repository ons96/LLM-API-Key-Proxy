"""The recovery script is the only thing that revives a `failed` canary.

It is shell, so these tests run it for real against a stubbed `systemctl` and
`logger` on PATH: they cover the decision (revive only `failed`), the ordering
that makes a revive possible (`reset-failed` before `start`), and the exit code
that tells the timer the revive did not work.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "newapi-sidecar-canary-recover"

STUB_SYSTEMCTL = """#!/bin/sh
echo "systemctl $*" >> "$CALLS"
if [ "$1" = show ]; then
  count=$(cat "$COUNT" 2>/dev/null || echo 0)
  count=$((count + 1))
  echo "$count" > "$COUNT"
  if [ "$count" -ge 2 ]; then echo "$FAKE_STATE_AFTER"; else echo "$FAKE_STATE"; fi
fi
exit 0
"""

STUB_LOGGER = """#!/bin/sh
echo "logger $*" >> "$CALLS"
"""


def run_script(tmp_path: Path, state: str, after: str) -> tuple[int, list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("systemctl", STUB_SYSTEMCTL), ("logger", STUB_LOGGER)):
        stub = bin_dir / name
        stub.write_text(body, encoding="utf-8")
        stub.chmod(0o755)

    calls = tmp_path / "calls"
    calls.write_text("", encoding="utf-8")
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CALLS": str(calls),
        "COUNT": str(tmp_path / "count"),
        "FAKE_STATE": state,
        "FAKE_STATE_AFTER": after,
        "SIDECAR_CANARY_RECOVER_DELAY": "0",
    }
    completed = subprocess.run(
        ["bash", str(SCRIPT)], capture_output=True, text=True, env=environment, check=False
    )
    return completed.returncode, calls.read_text(encoding="utf-8").splitlines()


def test_healthy_unit_is_left_alone(tmp_path):
    code, calls = run_script(tmp_path, "active", "active")

    assert code == 0
    assert calls == ["systemctl show --property=ActiveState --value newapi-sidecar-canary.service"]


def test_operator_stop_is_not_fought(tmp_path):
    """`systemctl stop` leaves the unit inactive; a timer must not undo that."""
    code, calls = run_script(tmp_path, "inactive", "inactive")

    assert code == 0
    assert not any("reset-failed" in call or call.startswith("systemctl start") for call in calls)


def test_failed_unit_is_reset_before_start(tmp_path):
    code, calls = run_script(tmp_path, "failed", "active")

    assert code == 0
    reset = calls.index("systemctl reset-failed newapi-sidecar-canary.service")
    start = calls.index("systemctl start newapi-sidecar-canary.service")
    assert reset < start, "without reset-failed the start is refused as 'too quickly'"
    # The failed canary pulled the token refresh in on every attempt, so that
    # unit is usually at its own limit; leaving it there revives a canary whose
    # stale token fails the audit immediately.
    token_reset = calls.index("systemctl reset-failed newapi-sidecar-canary-token.service")
    assert token_reset < start


def test_still_failed_after_start_reports_failure(tmp_path):
    """A canary that cannot stay up (for example a rate-limited login) must make
    the oneshot fail, so the journal records it rather than looking healthy."""
    code, calls = run_script(tmp_path, "failed", "failed")

    assert code == 1
    assert any("recovery finished state=failed" in call for call in calls)
