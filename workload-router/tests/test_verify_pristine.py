"""Acceptance tests for the pristine-worktree oracle (protocol section 8.7)."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from tools.verify_pristine import post_verdict, verify

GIT = ["git", "-c", "user.email=oracle@example.invalid", "-c", "user.name=oracle"]


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([*GIT, "-C", str(repo), *args], check=True,
                          capture_output=True, text=True)


def make_repo(tmp_path: Path, *, manifest: str = "pyproject") -> Path:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "mod.py").write_text("def add(a, b):\n    return 4\n")
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_mod.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
        "from pkg.mod import add\n\n"
        "def test_add():\n"
        "    assert add(1, 2) == 3\n"
    )
    if manifest == "pyproject":
        (repo / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\ntestpaths = [\"tests\"]\n")
    elif manifest == "make":
        (repo / "Makefile").write_text("check:\n\tpython3 -m pytest -q\n")
    git(repo, "init", "-b", "main")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base")
    return repo


def grade(repo: Path, **options):
    options.setdefault("cache_dir", repo.parent / "cache")
    return verify(repo, **options)


def test_acceptance_1_test_edit_is_rejected(tmp_path):
    repo = make_repo(tmp_path)
    test_file = repo / "tests" / "test_mod.py"
    test_file.write_text(test_file.read_text().replace("== 3", "== 4"))
    result = grade(repo)
    assert result["verdict"] in {"fail_scope", "fail_tamper"}
    assert result["verdict"] != "pass"


def test_acceptance_2_skip_insertion_is_tamper(tmp_path):
    repo = make_repo(tmp_path)
    test_file = repo / "tests" / "test_mod.py"
    test_file.write_text(test_file.read_text().replace(
        "def test_add():", "@pytest.mark.skip(reason=\"x\")\ndef test_add():"))
    result = grade(repo)
    assert result["verdict"] == "fail_tamper"


def test_acceptance_3_correct_patch_passes(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "pkg" / "mod.py").write_text("def add(a, b):\n    return 3\n")
    result = grade(repo)
    assert result["verdict"] == "pass"
    assert result["oracle"] == "V2"


def test_acceptance_4_syntax_error_fails_build(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "pkg" / "mod.py").write_text("def add(a, b)\n    return 3\n")
    result = grade(repo)
    assert result["verdict"] == "fail_build"


def test_acceptance_5_runner_comes_from_base_commit(tmp_path):
    repo = make_repo(tmp_path, manifest="make")
    (repo / "Makefile").write_text("check:\n\t@true\n")
    result = grade(repo)
    # A patched green target cannot grade itself: the base Makefile still runs pytest.
    assert result["verdict"] == "fail_test"


def test_acceptance_6_repo_without_runner_is_unknown(tmp_path):
    repo = tmp_path / "empty"
    repo.mkdir()
    (repo / "README.md").write_text("no runner here\n")
    git(repo, "init", "-b", "main")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base")
    (repo / "README.md").write_text("changed\n")
    result = verify(repo, cache_dir=tmp_path / "cache")
    assert result["verdict"] == "unknown"

    script = Path(__file__).resolve().parents[1] / "tools" / "verify_pristine.py"
    process = subprocess.run(
        [sys.executable, str(script), str(repo), "--cache-dir", str(tmp_path / "cache2")],
        capture_output=True, text=True,
    )
    assert process.returncode == 2
    assert json.loads(process.stdout)["verdict"] == "unknown"


def test_acceptance_7_same_diff_is_identical_and_cached(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "pkg" / "mod.py").write_text("def add(a, b):\n    return 3\n")
    first = grade(repo)
    second = grade(repo)
    assert first["verdict"] == second["verdict"] == "pass"
    assert second["cached"] is True


def test_acceptance_8_flaky_suite_is_flagged_and_clean_suite_is_stable(tmp_path):
    flaky = make_repo(tmp_path / "flaky")
    (flaky / "tests" / "test_mod.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
        "from pkg.mod import add\n\n"
        "def test_add():\n"
        "    marker = Path(__file__).with_suffix('.marker')\n"
        "    assert not marker.exists()\n"
        "    marker.write_text('seen')\n"
        "    assert add(1, 2) == 3\n"
    )
    git(flaky, "add", ".")
    git(flaky, "commit", "-m", "flaky fixture")
    (flaky / "pkg" / "mod.py").write_text("def add(a, b):\n    return 3\n")
    result = verify(flaky, flake_runs=2, use_cache=False,
                    cache_dir=tmp_path / "cache")
    assert result["verdict"] == "unknown"
    assert result["flaky"] is True

    clean = make_repo(tmp_path / "clean")
    (clean / "pkg" / "mod.py").write_text("def add(a, b):\n    return 3\n")
    verdicts = [verify(clean, use_cache=False)["verdict"] for _ in range(3)]
    assert verdicts == ["pass", "pass", "pass"]


def test_post_verdict_sends_only_bounded_fields():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        code = post_verdict(
            f"http://127.0.0.1:{server.server_port}/v1/router/outcome",
            "a" * 32, "pass", "V2", 12, token="client-key")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert code == 200
    assert seen["path"] == "/v1/router/outcome"
    assert seen["auth"] == "Bearer client-key"
    assert seen["body"] == {
        "attempt_id": "a" * 32, "grade": True, "verdict": "pass",
        "oracle": "V2", "duration_ms": 12,
    }
    with pytest.raises(ValueError, match="unsupported verdict"):
        post_verdict("http://127.0.0.1:1/", "a" * 32, "PASS", "V2", 1)
