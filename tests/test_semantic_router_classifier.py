"""Wiring tests for #949 slice 2: classifier as an opt-in step in resolve_auto."""

import pytest

from proxy_app import prompt_classifier as pc
from proxy_app.semantic_router import resolve_auto


def _req(text, **extra):
    return {"messages": [{"role": "user", "content": text}], **extra}


@pytest.fixture(autouse=True)
def _no_phase_header(monkeypatch):
    # Phase header short-circuits before the classifier; keep it out of the way.
    monkeypatch.delenv("PROMPT_CLASSIFIER_ENABLED", raising=False)


def test_disabled_by_default(monkeypatch):
    monkeypatch.setenv("AUTO_ROUTE_MODE", "off")  # no semantic deps in unit env
    r = resolve_auto(_req("write a python function to parse csv"))
    assert r.source != "classifier"


def test_enabled_code_prompt_uses_classifier(monkeypatch):
    monkeypatch.setenv("PROMPT_CLASSIFIER_ENABLED", "1")
    monkeypatch.setenv("AUTO_ROUTE_MODE", "off")
    r = resolve_auto(_req("def parse(path):\n    return open(path).read()"))
    assert r.source == "classifier"
    assert r.chain == pc.CATEGORY_CHAIN[pc.PromptCategory.CODE]


def test_enabled_with_tools_forces_tool_capable_chain(monkeypatch):
    monkeypatch.setenv("PROMPT_CLASSIFIER_ENABLED", "1")
    monkeypatch.setenv("AUTO_ROUTE_MODE", "off")
    r = resolve_auto(_req("write a python function that parses csv rows", tools=[{"type": "function"}]))
    assert r.source == "classifier"
    assert r.chain in pc.TOOL_CAPABLE


def test_enabled_but_unsure_falls_through(monkeypatch):
    monkeypatch.setenv("PROMPT_CLASSIFIER_ENABLED", "1")
    monkeypatch.setenv("AUTO_ROUTE_MODE", "off")
    r = resolve_auto({})  # empty request -> classifier fallback
    assert r.source != "classifier"


def test_phase_header_still_wins(monkeypatch):
    monkeypatch.setenv("PROMPT_CLASSIFIER_ENABLED", "1")
    r = resolve_auto(_req("def x(): pass"), phase_header="plan")
    assert r.source == "phase-header"
