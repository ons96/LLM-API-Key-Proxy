"""Management preflight and last-good catalog checks without live providers."""

from dataclasses import replace
import json
import sqlite3

import pytest

from newapi_catalog import AdminClient, CatalogError, key_fingerprint, load_provider_metadata, refresh_opencode, refresh_provider
from quota_ledger import QuotaLedger, Route


def candidate():
    return Route("r", "provider", "model", "opaque", "sc.r", 7,
                 quota_type="unlimited", input_price=0, output_price=0,
                 context_window=32768, output_limit=4096,
                 key_fingerprint=key_fingerprint("test-only-key"),
                 channel_base_url="http://127.0.0.1:9999")


@pytest.fixture
def management(monkeypatch):
    client = AdminClient("http://127.0.0.1:8002/v1", "test-only-admin", 1)
    channel = {"id": 7, "status": 1, "type": 1, "models": "sc.r", "group": "default",
               "channel_info": {"is_multi_key": False, "multi_key_size": 0},
               "key": "test-only-key", "model_mapping": '{"sc.r":"model"}',
               "base_url": "http://127.0.0.1:9999"}
    options = [{"key": "RetryTimes", "value": "0"}]
    listing = [channel]
    operations = []
    def call(method, path, payload=None):
        operations.append((method, path, payload))
        if path == "/api/option/":
            return options
        if path.startswith("/api/channel/?"):
            return {"items": listing, "total": len(listing)}
        if path == "/api/channel/7":
            return channel
        if path == "/api/channel/" and method == "PUT":
            return None
        if path == "/api/token/9":
            return {"id": 9, "status": 1, "group": "default", "cross_group_retry": False, "key": "relay-only"}
        raise AssertionError((method, path))
    monkeypatch.setattr(client, "call", call)
    return client, channel, options, listing, operations


def test_complete_audit_is_read_only(management):
    client, _, _, _, operations = management
    assert len(client.audit([candidate()])) == 64
    client.audit_relay("sk-relay-only", 9)
    assert all(method == "GET" for method, _, _ in operations)


@pytest.mark.parametrize("changes", [
    {"key": "different-key"}, {"key": "key1\nkey2"}, {"key": None},
    {"channel_info": {"is_multi_key": True}}, {"channel_info": {}},
    {"status": 2}, {"type": 2}, {"group": "auto"},
    {"model_mapping": '{"sc.r":"other"}'}, {"param_override": '{"model":"other"}'},
    {"header_override": '{"Authorization":"other"}'}, {"base_url": "http://127.0.0.1:9998"},
])
def test_unsafe_channel_fails_closed(management, changes):
    client, channel, _, _, _ = management
    channel.update(changes)
    with pytest.raises(CatalogError):
        client.audit([candidate()])


def test_missing_key_fingerprint_and_duplicate_alias_fail(management):
    client, _, options, listing, _ = management
    with pytest.raises(CatalogError):
        client.audit([replace(candidate(), key_fingerprint=None)])
    listing.append({"id": 8, "models": "other, sc.r"})
    with pytest.raises(CatalogError, match="shared"):
        client.audit([candidate()])
    listing.pop()
    options[0]["value"] = "1"
    with pytest.raises(CatalogError, match="RetryTimes"):
        client.audit([candidate()])


def test_priority_update_preserves_every_other_field(management):
    client, _, _, _, operations = management
    client.set_priorities([candidate()])
    writes = [payload for method, _, payload in operations if method == "PUT"]
    assert writes == [{"id": 7, "priority": 1}]


def test_three_catalogs_transactional_and_last_good(monkeypatch, tmp_path):
    ledger = QuotaLedger(tmp_path / "sidecar.db")
    responses = {name: {"data": [{"id": name + "-model", "name": "model", "untrusted": "must-not-store"}]}
                 for name in ("one", "two", "three")}
    def fetch(_opener, url, *_args):
        name = url.split("/")[-2]
        return responses[name]
    monkeypatch.setattr("newapi_catalog._request", fetch)
    try:
        for name in responses:
            metadata = {"name": name, "base_url": "http://127.0.0.1:9999/" + name, "env_var": None}
            refresh_provider(ledger, metadata)
        assert ledger.diagnostics()["catalog_models"] == 3
        assert "must-not-store" not in json.dumps([tuple(row) for row in ledger.db.execute("SELECT * FROM models")])
        responses["one"] = {"data": []}
        with pytest.raises(CatalogError):
            refresh_provider(ledger, {"name": "one", "base_url": "http://127.0.0.1:9999/one"})
        assert ledger.diagnostics()["catalog_models"] == 3
        responses["one"] = {"data": [{"id": "same"}, {"id": "same"}]}
        with pytest.raises(CatalogError):
            refresh_provider(ledger, {"name": "one", "base_url": "http://127.0.0.1:9999/one"})
        assert ledger.diagnostics()["catalog_models"] == 3
    finally:
        ledger.close()


def test_provider_database_is_read_only(tmp_path):
    path = tmp_path / "canonical.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE providers (name, base_url, env_var, no_api_key_required)")
        db.execute("INSERT INTO providers VALUES (?, ?, ?, ?)",
                   ("local", "http://127.0.0.1:9999/v1", "LOCAL_KEY", 0))
    before = path.read_bytes()
    data = load_provider_metadata(path, ["local"])
    assert data[0]["env_var"] == "LOCAL_KEY" and path.read_bytes() == before
    with pytest.raises(CatalogError):
        load_provider_metadata(path, ["missing"])
    assert path.read_bytes() == before


def test_three_provider_opencode_refresh_is_additive_and_symlink_safe(tmp_path):
    target = tmp_path / "opencode.stock.json"
    config = {"provider": {name: {"options": {"apiKey": "{env:TEST_KEY}"}, "models": {
        "curated": {"tool_call": False, "reasoning": True, "limit": {"context": 12345, "output": 321}}
    }} for name in ("one", "two", "three")}, "plugin": ["preserve-me"]}
    target.write_text(json.dumps(config))
    link = tmp_path / "opencode.json"
    link.symlink_to(target.name)
    catalogs = {name: [{"id": "curated"}, {"id": name + "-new"}] for name in config["provider"]}
    before = target.read_bytes()
    preview = refresh_opencode(link, catalogs)
    assert not preview["applied"] and target.read_bytes() == before
    result = refresh_opencode(link, catalogs, apply=True)
    assert result["applied"] and link.is_symlink()
    from pathlib import Path
    assert Path(result["backup"]).read_bytes() == before
    after = json.loads(target.read_bytes())
    assert after["plugin"] == config["plugin"]
    for name, provider in after["provider"].items():
        assert provider["options"] == config["provider"][name]["options"]
        assert provider["models"]["curated"] == config["provider"][name]["models"]["curated"]
        assert provider["models"][name + "-new"] == {"name": name + "-new"}
    assert target.stat().st_mode & 0o777 == 0o600
    assert not refresh_opencode(link, catalogs, apply=True)["applied"]


def test_normal_channel_refresh_preserves_aliases_and_model_mapping(monkeypatch):
    client = AdminClient("http://127.0.0.1:8002/v1", "test-admin", 1)
    writes = []
    def call(method, _path, payload=None):
        if method == "GET":
            return {"id": 7, "name": "ordinary", "models": "original,alias",
                    "model_mapping": '{"alias":"original"}'}
        writes.append(payload)
    monkeypatch.setattr(client, "call", call)
    assert client.refresh_channel(7, ["new", "original"]) == 1
    assert writes == [{"id": 7, "models": "original,alias,new"}]
