#!/usr/bin/env python3
"""Rebuild selected virtual-model chains without changing provider catalogs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import yaml

from chain_policy import candidate_key, blocked_reason, load_policy, sanitize_chain


DEFAULT_CONFIG = Path("/tmp/opencode/virtual_models.yaml")
DEFAULT_WORKING_MODELS = Path("/tmp/opencode/working_models.json")


def load_working_models(path: Path, policy: Mapping[str, Any]) -> Dict[str, List[Dict]]:
    """Load telemetry candidates, excluding only policy-blocked chain members."""
    try:
        raw = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, list):
        raise ValueError(f"working models must be a list: {path}")

    working: Dict[str, List[Dict]] = defaultdict(list)
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        provider = entry.get("provider")
        model = entry.get("model")
        if not isinstance(provider, str) or not isinstance(model, str):
            continue
        if not blocked_reason(provider, model, policy):
            working[provider].append(entry)
    return dict(working)


def is_good_coding_model(model: str, ttft: float, tps: float, samples: int) -> bool:
    """Return whether telemetry describes a usable coding candidate."""
    if ttft > 30000 or samples < 2:
        return False
    banned_terms = ("rp", "uncensored", "roleplay", "flux", "sdxl", "dall-e", "imagen")
    return not any(term in model.lower() for term in banned_terms)


def is_good_chat_model(model: str, ttft: float, tps: float, samples: int) -> bool:
    """Return whether telemetry describes a usable chat candidate."""
    return ttft <= 30000 and samples >= 2


def get_top_models(
    working: Mapping[str, Iterable[Dict]], count: int, filter_fn: Any
) -> List[Tuple[str, Dict]]:
    """Return fastest telemetry candidates across providers."""
    candidates = []
    for provider, entries in working.items():
        for entry in entries:
            if filter_fn(entry["model"], entry["ttft"], entry["tps"], entry["n"]):
                candidates.append((provider, entry))
    return sorted(candidates, key=lambda item: (-item[1]["tps"], item[1]["ttft"]))[:count]


def default_new_tops() -> Dict[str, List[Tuple[str, str]]]:
    """Return bounded manual recovery candidates for the non-agent virtual models."""
    return {
        "coding-fast": [
            ("groq", "llama-3.1-8b-instant"),
            ("groq", "llama-3.3-70b-versatile"),
            ("groq", "qwen/qwen3-32b"),
            ("mistral", "codestral-latest"),
            ("aihubmix", "coding-minimax-m3-free"),
            ("aihubmix", "coding-glm-5.1-free"),
            ("nvidia", "nvidia/nemotron-mini-4b-instruct"),
            ("gemini", "gemini-2.5-flash"),
        ],
        "coding-smart": [
            ("groq", "llama-3.3-70b-versatile"),
            ("groq", "qwen/qwen3-32b"),
            ("mistral", "codestral-latest"),
            ("futureppo", "kimi-k2.6"),
            ("aihubmix", "coding-glm-5.1-free"),
            ("gemini", "gemini-2.5-flash"),
            ("tokenrouter", "MiniMax-M3"),
            ("evolvex", "qwen/qwen3.5-397b-a17b"),
        ],
        "coding-elite": [
            ("futureppo", "kimi-k2.6"),
            ("nianhua", "claude-opus-4-8"),
            ("evolvex", "qwen/qwen3.5-397b-a17b"),
            ("tokenrouter", "MiniMax-M3"),
            ("pooled", "qwen3.7-max-official"),
            ("futureppo", "qwen3.7-max-t"),
            ("blaze_free", "MiniMax-M3-official"),
            ("aihubmix", "coding-minimax-m3-free"),
        ],
        "coding-free": [
            ("groq", "llama-3.3-70b-versatile"),
            ("groq", "llama-3.1-8b-instant"),
            ("groq", "qwen/qwen3-32b"),
            ("mistral", "codestral-latest"),
            ("gemini", "gemini-2.5-flash"),
            ("aihubmix", "coding-minimax-m3-free"),
        ],
        "chat-fast": [
            ("groq", "llama-3.1-8b-instant"),
            ("gemini", "gemini-2.5-flash"),
            ("groq", "llama-3.3-70b-versatile"),
            ("aihubmix", "coding-minimax-m3-free"),
            ("aihubmix", "coding-glm-5.1-free"),
            ("nvidia", "nvidia/nemotron-mini-4b-instruct"),
        ],
        "chat-smart": [
            ("gemini", "gemini-2.5-flash"),
            ("groq", "llama-3.3-70b-versatile"),
            ("futureppo", "kimi-k2.6"),
            ("tokenrouter", "MiniMax-M3"),
            ("blaze_free", "MiniMax-M3-official"),
            ("nvidia", "z-ai/glm-5.1"),
        ],
        "chat-elite": [
            ("nianhua", "claude-opus-4-8"),
            ("futureppo", "kimi-k2.6"),
            ("evolvex", "qwen/qwen3.5-397b-a17b"),
            ("pooled", "qwen3.7-max-official"),
            ("tokenrouter", "MiniMax-M3"),
            ("blaze_free", "MiniMax-M3-official"),
        ],
        "glm5-elite": [
            ("nvidia", "z-ai/glm-5.1"),
            ("ktai-paid", "z-ai/glm-5.1"),
            ("blaze_free", "glm-5.2-china"),
            ("pooled", "glm-5.1-official"),
        ],
        "title-fast": [
            ("groq", "llama-3.1-8b-instant"),
            ("gemini", "gemini-2.5-flash"),
            ("groq", "llama-3.3-70b-versatile"),
        ],
        "auto": [
            ("groq", "llama-3.3-70b-versatile"),
            ("gemini", "gemini-2.5-flash"),
            ("groq", "llama-3.1-8b-instant"),
            ("mistral", "codestral-latest"),
            ("futureppo", "kimi-k2.6"),
            ("groq", "qwen/qwen3-32b"),
        ],
        "chat-rp": [
            ("groq", "llama-3.1-8b-instant"),
            ("groq", "llama-3.3-70b-versatile"),
            ("gemini", "gemini-2.5-flash"),
            ("nvidia", "z-ai/glm-5.1"),
        ],
    }


def filter_new_tops_by_observed_models(
    new_tops: Mapping[str, Iterable[Tuple[str, str]]],
    working: Mapping[str, Iterable[Mapping[str, Any]]],
) -> Dict[str, List[Tuple[str, str]]]:
    """Keep static recovery candidates only when current telemetry observed them."""
    observed = {
        candidate_key(provider, entry["model"])
        for provider, entries in working.items()
        for entry in entries
        if isinstance(entry, Mapping) and isinstance(entry.get("model"), str)
    }
    if not observed:
        return {}
    return {
        name: [
            (provider, model)
            for provider, model in entries
            if candidate_key(provider, model) in observed
        ]
        for name, entries in new_tops.items()
    }


def rebuild_document(
    document: Mapping[str, Any],
    new_tops: Mapping[str, Iterable[Tuple[str, str]]],
    policy: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Return a target-only, policy-safe rebuild and names changed.

    Catalogs and unrelated virtual models are retained untouched.
    """
    if not isinstance(document, Mapping):
        raise ValueError("virtual model config must be a mapping")
    virtual_models = document.get("virtual_models")
    if not isinstance(virtual_models, Mapping):
        raise ValueError("virtual_models must be a mapping")

    rebuilt = dict(document)
    rebuilt_models = dict(virtual_models)
    changed = []
    for name, entries in new_tops.items():
        virtual_model = virtual_models.get(name)
        if not isinstance(virtual_model, Mapping):
            continue
        existing_chain = virtual_model.get("fallback_chain", [])
        if not isinstance(existing_chain, list):
            raise ValueError(f"{name} fallback_chain must be a list")
        candidates = [
            {"provider": provider, "model": model} for provider, model in entries
        ]
        sanitized = sanitize_chain(candidates + existing_chain, policy)
        updated_model = dict(virtual_model)
        updated_model["fallback_chain"] = sanitized
        if updated_model != virtual_model:
            rebuilt_models[name] = updated_model
            changed.append(name)
    rebuilt["virtual_models"] = rebuilt_models
    return rebuilt, changed


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild selected policy-safe fallback chains")
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--working-models", type=Path, default=DEFAULT_WORKING_MODELS)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    policy = load_policy(args.policy)
    working = load_working_models(args.working_models, policy)
    new_tops = filter_new_tops_by_observed_models(default_new_tops(), working)
    print(f"Telemetry providers available: {len(working)}")
    if not working:
        print("No observed telemetry candidates; refusing static rebuild.")
        return 0
    document = yaml.safe_load(args.config.read_text()) or {}
    rebuilt, changed = rebuild_document(document, new_tops, policy)

    if not changed:
        print("No targeted chains changed.")
        return 0
    print(f"Policy-safe chains: {', '.join(changed)}")
    if args.dry_run:
        print(f"Dry run: would update {args.config}")
        return 0

    backup = args.config.with_name(f"{args.config.name}.bak-pre-rebuild")
    backup.write_text(args.config.read_text())
    args.config.write_text(yaml.safe_dump(rebuilt, default_flow_style=False, sort_keys=False))
    print(f"Backup: {backup}")
    print(f"Wrote: {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
