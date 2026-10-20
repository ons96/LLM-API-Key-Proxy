"""M2: the free invariant.

Proves three things:
  1. with free-only on, the model listing exposes zero paid models
  2. a request naming a paid model is rejected with a typed error, not a silent
     fallback
  3. no automatic (virtual-model) chain contains a tier-4 model
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from proxy_app.routing import free_policy  # noqa: E402
from proxy_app.router_core import RouterCore  # noqa: E402


@pytest.fixture(scope="module")
def tiers():
    return free_policy.load_tiers(REPO_ROOT, force=True)


@pytest.fixture(scope="module")
def catalog():
    path = REPO_ROOT / "config" / "catalog.json.gz"
    import gzip

    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


# --------------------------------------------------------------------------- #
# 1. the listing hides paid models
# --------------------------------------------------------------------------- #


def test_paid_provider_models_are_hidden_from_listing(tiers, catalog):
    paid_providers = {
        p["provider_id"]
        for p in catalog["providers"]
        if int(p["quota_tier"]) >= free_policy.PAID_TIER
    }
    assert paid_providers, "fixture expects at least one paid provider"

    listing = []
    for provider in catalog["providers"]:
        for model in provider["models"]:
            listing.append(f"{provider['provider_id']}/{model['model_id']}")

    visible = free_policy.filter_model_ids(listing, tiers, free_only=True)

    leaked = [
        mid for mid in visible if mid.split("/", 1)[0] in paid_providers
    ]
    assert leaked == [], f"paid models leaked into /v1/models: {leaked[:5]}"
    assert len(visible) < len(listing)


def test_listing_unchanged_when_free_only_off(tiers):
    listing = ["openai/gpt-5", "groq/llama-3.3-70b", "auto"]
    assert free_policy.filter_model_ids(listing, tiers, free_only=False) == listing


def test_aliases_survive_the_filter(tiers):
    visible = free_policy.filter_model_ids(["auto", "coding-fast"], tiers, free_only=True)
    assert visible == ["auto", "coding-fast"]


def test_paid_provider_is_refused(tiers):
    allowed, reason = free_policy.allowed_in_free_mode(
        tiers, "openai", "gpt-5", free_only=True
    )
    assert allowed is False
    assert "paid" in reason


def test_one_time_credit_is_allowed_as_last_resort(tiers):
    """Tier 3 is reserved, not excluded: ordering keeps it behind tier 0/1/2."""
    allowed, reason = free_policy.allowed_in_free_mode(
        tiers, "koyeb", "any-model", free_only=True
    )
    assert allowed is True
    assert "last" in reason


# --------------------------------------------------------------------------- #
# 2. a paid model is a typed error
# --------------------------------------------------------------------------- #


def test_paid_model_error_payload_is_typed():
    err = free_policy.PaidModelError("openai", "gpt-5", 4, "provider_type=paid")
    assert err.status_code == 402
    payload = err.to_payload()["error"]
    assert payload["type"] == "paid_model_rejected"
    assert payload["code"] == "paid_model_rejected"
    assert payload["tier"] == 4
    assert payload["provider"] == "openai"


class _RouterStandIn:
    """Only the attributes ``_reject_paid_model`` needs."""

    free_only_mode = True
    config: dict = {"providers": {}}

    _reject_paid_model = RouterCore._reject_paid_model
    _free_tiers = RouterCore._free_tiers


def test_router_rejects_paid_model_before_any_candidate_lookup():
    router = _RouterStandIn()
    with pytest.raises(free_policy.PaidModelError) as excinfo:
        router._reject_paid_model("openai/gpt-5")
    assert excinfo.value.provider == "openai"
    assert excinfo.value.status_code == 402


def test_router_does_not_reject_free_model():
    router = _RouterStandIn()
    router._reject_paid_model("groq/llama-3.3-70b-versatile")  # must not raise


def test_router_ignores_rejection_when_free_only_off():
    class Off(_RouterStandIn):
        free_only_mode = False

    Off()._reject_paid_model("openai/gpt-5")  # must not raise


@pytest.mark.asyncio
async def test_route_request_converts_to_typed_http_error(monkeypatch):
    """route_request turns PaidModelError into a 402 with a typed detail."""
    from fastapi import HTTPException

    router = _RouterStandIn()
    router.config = {"providers": {}}

    async def _explode(request, request_id):
        raise free_policy.PaidModelError("openai", "gpt-5", 4, "provider_type=paid")

    monkeypatch.setattr(router, "_route_request_inner", _explode, raising=False)
    monkeypatch.setenv("GATEWAY_RETRY_UNTIL_DONE", "false")

    with pytest.raises(HTTPException) as excinfo:
        await RouterCore.route_request(
            router, {"model": "openai/gpt-5"}, "test-request"
        )
    assert excinfo.value.status_code == 402
    assert excinfo.value.detail["type"] == "paid_model_rejected"


# --------------------------------------------------------------------------- #
# 3. no automatic chain contains a tier-4 model
# --------------------------------------------------------------------------- #


def test_no_automatic_chain_contains_a_paid_model(tiers):
    vm_path = REPO_ROOT / "config" / "virtual_models.yaml"
    data = yaml.safe_load(vm_path.read_text(encoding="utf-8")) or {}
    chains = data.get("virtual_models") or {}

    offenders: list[str] = []
    checked = 0

    for alias, config in chains.items():
        if not isinstance(config, dict):
            continue
        # A static_virtual_models entry is still automatic: nothing asks for it.
        for entry in config.get("fallback_chain") or []:
            if not isinstance(entry, dict):
                continue
            provider = entry.get("provider")
            model = entry.get("model")
            if not provider or not model:
                continue
            checked += 1
            tier = free_policy.provider_tier(tiers, provider)
            if tier >= free_policy.PAID_TIER:
                offenders.append(f"{alias} -> {provider}/{model} (tier {tier})")

    assert checked > 0, "no chain entries found; the test would be vacuous"
    assert offenders == [], f"automatic chains contain paid models: {offenders[:10]}"


def test_config_free_only_mode_is_on():
    data = yaml.safe_load(
        (REPO_ROOT / "config" / "router_config.yaml").read_text(encoding="utf-8")
    )
    assert data["free_only_mode"] is True


def test_catalog_artifact_is_present_and_parsable(catalog):
    assert catalog["schema_version"] == 1
    assert catalog["providers"], "catalog artifact is empty"
    assert catalog["totals"]["models"] > 0
