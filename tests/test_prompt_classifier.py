"""Tests for the #949 prompt classifier (slice 1: corpus + heuristic baseline)."""

import json
from pathlib import Path

from proxy_app import prompt_classifier as pc

CORPUS = Path(__file__).parent / "fixtures" / "prompt_classifier_eval.jsonl"
# ponytail: floor, not target. Raise it when the corpus grows.
MIN_ACCURACY = 0.9


def _req(text, **extra):
    return {"messages": [{"role": "user", "content": text}], **extra}


def test_corpus_accuracy_floor():
    rows = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
    hits = sum(pc.classify(_req(r["text"])).category.value == r["label"] for r in rows)
    assert hits / len(rows) >= MIN_ACCURACY, f"{hits}/{len(rows)}"


def test_tools_force_tool_capable_chain():
    r = pc.classify(_req("hi", tools=[{"type": "function"}]))
    assert r.suggested_chain in pc.TOOL_CAPABLE


def test_empty_and_garbage_fall_back_safely():
    for req in ({}, None, [], {"messages": "nope"}, _req("")):
        r = pc.classify(req)
        assert r.fallback is True
        assert r.suggested_chain == pc._default_chain()


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("PROMPT_CLASSIFIER_ENABLED", raising=False)
    assert pc.is_enabled() is False
