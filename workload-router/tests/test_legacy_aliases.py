"""Legacy gateway alias compatibility at the sidecar request boundary."""

import pytest

from newapi_runtime import LEGACY_ALIASES, MODEL_ALIASES, PROFILE_ALIASES, prepare


def body(model: str, **options) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}], **options}


@pytest.mark.parametrize("model", sorted(LEGACY_ALIASES))
def test_every_legacy_alias_is_accepted(model):
    prepared = prepare(body(model))
    profile, step = LEGACY_ALIASES[model]

    assert prepared.work.profile == profile
    assert prepared.work.step == step
    assert prepared.model == model


def test_alias_inventory_covers_the_live_gateway():
    assert len(LEGACY_ALIASES) == 19
    assert len(MODEL_ALIASES) == 22
    assert set(PROFILE_ALIASES) <= set(MODEL_ALIASES)


def test_explicit_route_step_overrides_the_legacy_default():
    prepared = prepare(body("coding-elite"), step="debug")

    assert prepared.work.profile == "auto-coding"
    assert prepared.work.step == "debug"


def test_public_profiles_keep_their_default_steps():
    assert prepare(body("auto-coding")).work.step == "implement"
    assert prepare(body("auto-code")).work.step == "implement"
    assert prepare(body("auto-chat")).work.step == "chat"


@pytest.mark.parametrize("model", ["gpt-4o", "coding-elite ", "", "AUTO"])
def test_unknown_names_still_fail_closed(model):
    with pytest.raises(ValueError, match="model must be one of"):
        prepare(body(model))
