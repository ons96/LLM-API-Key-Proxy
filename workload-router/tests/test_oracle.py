"""Tests for the two-grader oracle.

Covers command detection for each repository shape, the validator grader for
edits with no repository or no tests (the opencode.json case), and the
"nothing to grade" outcome that must not be reported as a pass.
"""

import json
import subprocess

from oracle import detect_test_command, grade, validate_file


def test_detects_npm_test_script(tmp_path):
    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest run"}}), encoding="utf-8"
    )
    assert detect_test_command(tmp_path) == ["npm", "test", "--silent"]


def test_ignores_npm_placeholder_test(tmp_path):
    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"test": "echo \"Error: no test specified\" && exit 1"}}),
        encoding="utf-8",
    )
    assert detect_test_command(tmp_path) is None


def test_detects_makefile_test_target(tmp_path):
    (tmp_path / "Makefile").write_text("build:\n\ttrue\n\ntest:\n\tpytest -q\n", encoding="utf-8")
    assert detect_test_command(tmp_path) == ["make", "test"]


def test_detects_pyproject_pytest(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\naddopts = '-q'\n", encoding="utf-8"
    )
    assert detect_test_command(tmp_path) == ["python", "-m", "pytest", "-q"]


def test_detects_tests_directory(tmp_path):
    (tmp_path / "tests").mkdir()
    assert detect_test_command(tmp_path) == ["python", "-m", "pytest", "-q"]


def test_repo_without_tests_has_no_command(tmp_path):
    (tmp_path / "README.md").write_text("no tests here", encoding="utf-8")
    assert detect_test_command(tmp_path) is None


def test_grade_runs_detected_command_and_reports_failure(tmp_path):
    (tmp_path / "Makefile").write_text("test:\n\ttrue\n", encoding="utf-8")
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs["cwd"]))
        return subprocess.CompletedProcess(command, 1, stdout="1 failed", stderr="")

    result = grade(tmp_path, runner=fake_runner)

    assert calls == [(["make", "test"], str(tmp_path))]
    assert result.grader == "tests"
    assert result.passed is False
    assert "exit 1" in result.detail and "1 failed" in result.detail


def test_grade_reports_missing_test_binary_as_failure(tmp_path):
    (tmp_path / "Makefile").write_text("test:\n\ttrue\n", encoding="utf-8")

    def missing(command, **kwargs):
        raise FileNotFoundError("make")

    result = grade(tmp_path, runner=missing)
    assert result.passed is False
    assert "not installed" in result.detail


def test_grade_reports_timeout_as_failure(tmp_path):
    (tmp_path / "Makefile").write_text("test:\n\ttrue\n", encoding="utf-8")

    def slow(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 1)

    result = grade(tmp_path, runner=slow)
    assert result.passed is False
    assert "timed out" in result.detail


def test_config_edit_with_no_repo_falls_back_to_validator(tmp_path):
    config = tmp_path / "opencode.json"
    config.write_text(json.dumps({"provider": {"x": {"models": {}}}}), encoding="utf-8")

    result = grade(config, changed=[config])

    assert result.grader == "validator"
    assert result.passed is True


def test_broken_config_edit_fails_validation(tmp_path):
    config = tmp_path / "opencode.json"
    config.write_text('{"provider": {"x": ', encoding="utf-8")

    result = grade(config, changed=[config])

    assert result.grader == "validator"
    assert result.passed is False
    assert "JSONDecodeError" in result.detail


def test_yaml_edit_is_validated(tmp_path):
    good = tmp_path / "router_config.yaml"
    good.write_text("providers:\n  groq:\n    enabled: true\n", encoding="utf-8")
    assert validate_file(good) == (True, "router_config.yaml: parsed")

    bad = tmp_path / "bad.yaml"
    bad.write_text("providers: [unclosed", encoding="utf-8")
    ok, detail = validate_file(bad)
    assert ok is False and "bad.yaml" in detail


def test_nothing_to_grade_is_not_a_pass(tmp_path):
    note = tmp_path / "notes.txt"
    note.write_text("just text", encoding="utf-8")

    result = grade(note, changed=[note])

    assert result.grader == "none"
    assert result.passed is None
    assert "nothing to grade" in result.detail


def test_repo_with_tests_prefers_tests_over_validator(tmp_path):
    (tmp_path / "tests").mkdir()
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")

    def fake_runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="3 passed", stderr="")

    result = grade(tmp_path, changed=[config], runner=fake_runner)
    assert result.grader == "tests"
    assert result.passed is True
