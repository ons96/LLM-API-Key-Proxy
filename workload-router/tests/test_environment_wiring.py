from unittest.mock import patch

from provider_adapter import adapters_from_environment
from router_core import Capability, Deployment


def test_environment_wiring_skips_unconfigured_and_requires_url():
    deployments = [
        Deployment("local", Capability.FAST_GENERAL, 1000),
        Deployment("remote", Capability.FAST_GENERAL, 1000, provider="demo"),
    ]
    with patch.dict("os.environ", {"DEMO_BASE_URL": "https://provider.invalid"}, clear=True):
        adapters = adapters_from_environment(deployments)
    assert set(adapters) == {"remote"}


def test_environment_wiring_does_not_require_api_key_for_public_endpoint():
    deployment = Deployment("remote", Capability.FAST_GENERAL, 1000, provider="demo")
    with patch.dict("os.environ", {"DEMO_BASE_URL": "https://provider.invalid"}, clear=True):
        adapter = adapters_from_environment([deployment])["remote"]
    assert adapter.base_url == "https://provider.invalid"
    assert adapter.api_key == ""
