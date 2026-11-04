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


@pytest.mark.parametrize(("prompt", "profile", "step"), [
    ("hi", "auto-chat", "chat"),
    ("Implement a function that parses JSON", "auto-coding", "implement"),
    ("The pytest suite is failing", "auto-coding", "debug"),
    ("Debug this traceback", "auto-coding", "debug"),
    ("Verify the change", "auto-coding", "verify"),
    ("Prove this step by step", "auto-chat", "chat"),
])
def test_auto_classifies_the_latest_user_message(prompt, profile, step):
    prepared = prepare({"model": "auto", "messages": [{"role": "user", "content": prompt}]})

    assert prepared.work.profile == profile
    assert prepared.work.step == step


def test_auto_explicit_step_selects_the_matching_profile():
    assert prepare(body("auto"), step="verify").work.profile == "auto-coding"
    assert prepare(body("auto"), step="verify").work.step == "verify"
    assert prepare(body("auto"), step="chat").work.profile == "auto-chat"
    assert prepare(body("auto"), step="chat").work.step == "chat"


def test_auto_tools_stay_on_the_chat_profile_and_preserve_the_capability():
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
    prepared = prepare(body("auto", tools=tools))

    assert prepared.work.profile == "auto-chat"
    assert prepared.work.step == "chat"
    assert prepared.work.tools is True


def test_auto_tools_do_not_override_an_explicit_code_intent():
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
    prepared = prepare(body("auto", tools=tools, messages=[{
        "role": "user", "content": "Implement the parser change"
    }]))

    assert prepared.work.profile == "auto-coding"
    assert prepared.work.step == "implement"
    assert prepared.work.tools is True


def test_explicit_profiles_do_not_use_the_auto_classifier():
    prompt = {"role": "user", "content": "Implement a function that parses JSON"}
    prepared = prepare({"model": "auto-chat", "messages": [prompt]})

    assert prepared.work.profile == "auto-chat"
    assert prepared.work.step == "chat"


@pytest.mark.parametrize("model", ["gpt-4o", "coding-elite ", "", "AUTO"])
def test_unknown_names_still_fail_closed(model):
    with pytest.raises(ValueError, match="model must be one of"):
        prepare(body(model))
