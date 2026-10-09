import sqlite3

from provider_metadata import load_provider_group_deployments
from router_core import Capability


def test_provider_metadata_exports_ordered_free_chain_without_secrets(tmp_path):
    path = tmp_path / "providers.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE virtual_models (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE fallback_chains (
            id INTEGER PRIMARY KEY, virtual_model_id INTEGER, provider_key TEXT,
            model_id TEXT, priority INTEGER, capabilities TEXT
        );
        CREATE TABLE providers (
            id INTEGER, key_name TEXT, base_url TEXT, env_var TEXT, enabled INTEGER,
            free_tier INTEGER, no_api_key_required INTEGER, free_unlimited INTEGER,
            free_daily INTEGER
        );
        CREATE TABLE models (
            id INTEGER, provider_id INTEGER, model_id TEXT, context_window INTEGER,
            tps INTEGER, free_tier INTEGER, capabilities TEXT
        );
        INSERT INTO virtual_models VALUES (1, 'chat-fast');
        INSERT INTO providers VALUES (1, 'demo', 'https://provider.invalid/v1', 'DEMO_API_KEY', 1, 1, 0, 0, 0);
        INSERT INTO models VALUES (1, 1, 'fast-model', 32768, 100, 1, '["json"]');
        INSERT INTO fallback_chains VALUES (1, 1, 'demo', 'fast-model', 1, '["structured_output"]');
        """
    )
    connection.commit()
    connection.close()

    deployments = load_provider_group_deployments(path, ["chat-fast"])

    assert len(deployments) == 1
    deployment = deployments[0]
    assert deployment.group == "chat-fast"
    assert deployment.capability == Capability.FAST_GENERAL
    assert deployment.base_url == "https://provider.invalid/v1"
    assert deployment.env_prefix == "DEMO"
    assert deployment.structured_output is True
    assert deployment.chain_priority == 1
