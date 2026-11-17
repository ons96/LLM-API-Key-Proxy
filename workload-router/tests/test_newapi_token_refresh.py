import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import newapi_token_refresh as refresh  # noqa: E402


def test_parse_env_handles_quotes_comments_and_spaces():
    text = (
        "# comment\n"
        "ROOT_PASSWORD=\"s3cret value\"\n"
        "NEWAPI_CHANNEL_KEY_PROOF_COMMAND=/usr/bin/python3 /opt/x/proof.py --mode=totp\n"
        "EMPTY=\n"
    )
    values = refresh.parse_env(text)
    assert values["ROOT_PASSWORD"] == "s3cret value"
    assert values["NEWAPI_CHANNEL_KEY_PROOF_COMMAND"].endswith("--mode=totp")
    assert values["EMPTY"] == ""


def test_update_env_text_replaces_and_appends():
    text = "A=1\nNEWAPI_ADMIN_TOKEN=old\nB=2\n"
    updated = refresh.update_env_text(text, {"NEWAPI_ADMIN_TOKEN": "new",
                                             "NEWAPI_PROOF_ADMIN_TOKEN": "new"})
    assert "NEWAPI_ADMIN_TOKEN=new" in updated
    assert "NEWAPI_PROOF_ADMIN_TOKEN=new" in updated
    assert "A=1" in updated and "B=2" in updated
    assert "old" not in updated


def test_update_env_text_leaves_unrelated_spaced_values_intact():
    text = "NEWAPI_CHANNEL_KEY_PROOF_COMMAND=/usr/bin/python3 /opt/x.py --mode=totp\n"
    updated = refresh.update_env_text(text, {"NEWAPI_ADMIN_TOKEN": "t"})
    assert "/usr/bin/python3 /opt/x.py --mode=totp" in updated


def test_write_private_is_atomic_and_restrictive(tmp_path):
    target = tmp_path / "sidecar.env"
    target.write_text("OLD=1\n")
    refresh.write_private(target, "NEW=2\n")
    assert target.read_text() == "NEW=2\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not (tmp_path / "sidecar.env.tmp").exists()


def test_totp_matches_rfc6238_vector():
    secret = b"12345678901234567890"
    assert refresh.totp_code(secret, now=59) == "287082"
    assert refresh.totp_code(secret, now=1111111109) == "081804"


def test_loopback_base_rejects_remote_and_paths():
    assert refresh.loopback_base("http://127.0.0.1:8001") == "http://127.0.0.1:8001"
    for bad in ("http://10.0.0.5:8001", "http://example.com", "http://127.0.0.1:8001/api",
                "file:///etc/passwd"):
        with pytest.raises(refresh.RefreshError):
            refresh.loopback_base(bad)


def test_login_requires_token_flow_or_id(monkeypatch):
    monkeypatch.setattr(refresh, "_request",
                        lambda *a, **k: {"success": True, "data": {"user": {"id": 1}}})
    with pytest.raises(refresh.RefreshError, match="neither a session token"):
        refresh.login("http://127.0.0.1:8001", "root", "pw")


def test_login_returns_flow_token_when_2fa_is_required(monkeypatch):
    monkeypatch.setattr(refresh, "_request", lambda *a, **k: {
        "success": True,
        "data": {"require_verification": True, "flow_token": "flow"},
    })
    assert refresh.login("http://127.0.0.1:8001", "root", "pw") == {
        "token": "", "user_id": 0, "flow_token": "flow"}


def test_complete_2fa_returns_token_and_id(monkeypatch):
    seen = {}

    def fake(method, url, payload=None, token="", user_id=None):
        seen["url"] = url
        seen["payload"] = payload
        return {"success": True, "data": {"access_token": "jwt", "user": {"id": 7}}}

    monkeypatch.setattr(refresh, "_request", fake)
    assert refresh.complete_2fa("http://127.0.0.1:8001", "flow", "123456") == ("jwt", 7)
    assert seen["url"].endswith("/api/user/login/2fa")
    assert seen["payload"] == {"flow_token": "flow", "code": "123456"}


def test_next_window_delay_never_reuses_the_current_step():
    assert 1 <= refresh.next_window_delay(now=1000) <= 31
    assert refresh.next_window_delay(now=1029) > refresh.next_window_delay(now=1000)


def test_login_reports_rejection_without_echoing_password(monkeypatch):
    monkeypatch.setattr(refresh, "_request",
                        lambda *a, **k: {"success": False, "message": "wrong password"})
    with pytest.raises(refresh.RefreshError) as exc:
        refresh.login("http://127.0.0.1:8001", "root", "hunter2")
    assert "hunter2" not in str(exc.value)


def test_durable_token_reads_string_and_object_payloads(monkeypatch):
    calls = []

    def fake(method, url, payload=None, token="", user_id=None):
        calls.append(url)
        if url.endswith("/api/verify"):
            return {"success": True, "data": {"proof_token": "proof"}}
        return {"success": True, "data": {"token": "minted"}}

    monkeypatch.setattr(refresh, "_request", fake)
    assert refresh.durable_token("http://127.0.0.1:8001", "s", 1, "123456") == "minted"
    assert calls[-1].endswith("/api/user/access_tokens")


def test_main_check_mode_never_writes(tmp_path, monkeypatch, capsys):
    source = tmp_path / "new-api.env"
    source.write_text("ROOT_PASSWORD=pw\n")
    target = tmp_path / "sidecar.env"
    target.write_text("NEWAPI_ADMIN_TOKEN=old\n")
    monkeypatch.setattr(refresh, "login",
                        lambda *a: {"token": "session", "user_id": 3, "flow_token": ""})
    monkeypatch.setattr(refresh, "token_is_live", lambda *a: True)
    assert refresh.main(["--newapi-env", str(source), "--target-env", str(target)]) == 0
    assert target.read_text() == "NEWAPI_ADMIN_TOKEN=old\n"
    assert "check only" in capsys.readouterr().out


def test_main_apply_writes_token_and_backup(tmp_path, monkeypatch):
    source = tmp_path / "new-api.env"
    source.write_text("ROOT_PASSWORD=pw\n")
    target = tmp_path / "sidecar.env"
    target.write_text("NEWAPI_ADMIN_TOKEN=old\nOTHER=1\n")
    os.chmod(target, 0o600)
    monkeypatch.setattr(refresh, "login",
                        lambda *a: {"token": "session", "user_id": 3, "flow_token": ""})
    monkeypatch.setattr(refresh, "token_is_live", lambda *a: True)
    assert refresh.main(["--newapi-env", str(source), "--target-env", str(target),
                         "--apply"]) == 0
    text = target.read_text()
    assert "NEWAPI_ADMIN_TOKEN=session" in text
    assert "NEWAPI_PROOF_ADMIN_TOKEN=session" in text
    assert "OTHER=1" in text
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    backups = list(tmp_path.glob("sidecar.env.bak-token-*"))
    assert len(backups) == 1


def test_main_fails_closed_without_password(tmp_path, capsys):
    source = tmp_path / "new-api.env"
    source.write_text("SOMETHING_ELSE=1\n")
    assert refresh.main(["--newapi-env", str(source),
                         "--target-env", str(tmp_path / "sidecar.env")]) == 1
    assert "root password not found" in capsys.readouterr().err
