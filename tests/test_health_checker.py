"""Health-checker blocklist tests.

Pins the behaviour that a provider listed in ``config/dead_providers.yaml`` is
never pinged, so a dead route stops producing a failure line every interval.
"""

import asyncio

from proxy_app.health_checker import HealthChecker, _blocked_provider_names


def test_blocklist_names_are_lowercased(tmp_path):
    path = tmp_path / "dead.yaml"
    path.write_text("blocked_providers:\n- provider: Supacoder\n- provider: crax\n")
    assert _blocked_provider_names(str(path)) == {"supacoder", "crax"}


def test_missing_file_yields_empty_set(tmp_path):
    assert _blocked_provider_names(str(tmp_path / "absent.yaml")) == set()


def test_malformed_file_yields_empty_set(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("blocked_providers: [unterminated\n")
    assert _blocked_provider_names(str(path)) == set()


def test_blocklisted_provider_is_not_pinged(tmp_path, monkeypatch):
    path = tmp_path / "dead.yaml"
    path.write_text("blocked_providers:\n- provider: deadprov\n")
    monkeypatch.setenv("DEAD_PROVIDERS_PATH", str(path))
    monkeypatch.delenv("HEALTH_CHECK_SKIP", raising=False)

    pinged = []

    class FakeAdapter:
        def __init__(self, name):
            self.name = name

        async def list_models(self):
            return [f"{self.name}-model"]

    class FakeIntegration:
        adapters = {
            "deadprov": FakeAdapter("deadprov"),
            "liveprov": FakeAdapter("liveprov"),
        }

    checker = HealthChecker(FakeIntegration())

    async def fake_ping(provider, model):
        pinged.append(provider)
        return True

    monkeypatch.setattr(checker, "_ping_provider", fake_ping)

    asyncio.run(checker._check_all_providers())

    assert pinged == ["liveprov"]
