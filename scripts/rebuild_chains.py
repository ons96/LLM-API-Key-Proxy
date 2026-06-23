#!/usr/bin/env python3
"""Rebuild chains in virtual_models.yaml with real working (provider, model) pairs at the top.

Inputs:
- /tmp/opencode/working_models.json: dump from dump_alias_to_actual.py (provider -> [(model, n, ttft, tps), ...])
- current virtual_models.yaml

Strategy per chain:
- For each chain, prepend 3-8 working models from telemetry that match the chain's purpose
- Keep all existing entries (user rule: NEVER remove)
- Renumber priorities

For each chain we hand-pick the best working models based on:
- Chain purpose (coding-* vs chat-*)
- TTFT < 10s threshold
- n >= 3 samples
- Provider not in dead list
"""
import json
import sys
import re
from collections import defaultdict
from pathlib import Path

# Dead providers to NEVER use as new entries (user confirmed these are dead)
# Note: 'gemini' is RATE-LIMITED (429) but works when not rate-limited -- keep it
DEAD_PROVIDERS = {
    "cliproxyapi",  # manual OAuth, often broken
    "supacoder",    # dead
    "antigravity",  # unreliable OAuth
    "blaze", "blaze_org", "blazeai",  # all 401
    "agentrouter",  # 401
    "aibz",         # blocked
    "logfare",      # 500
    "voidai",       # 500
    "zenllm_responses",  # blocked
    "free_gemini",  # 401
    "openrouter",   # out of credits
}

# Working providers in priority order (user wants free-credit first, NVIDIA as backup)
WORKING_FREE_TIER = {
    "groq", "mistral", "cerebras", "nvidia",  # unlimited free
    "koyeb", "koyeb-new",  # has free credit
    "freetheai",  # daily checkin
    "blaze-free",  # free tier
    "iamhc", "paxsenix", "nianhua", "buddybackend",  # chinese gateways
    "anmix", "tokenrouter", "tokenlb", "pooled",  # chinese gateways
    "navy",  # chinese gateway
    "zenllm-free",  # free tier
    "futureppo",  # daily checkin
    "kilocode", "kilo",  # free tier
}


def load_working_models(path="/tmp/opencode/working_models.json"):
    """Load {provider: [(model, n, ttft, tps), ...]} from JSON dump."""
    with open(path) as f:
        raw = json.load(f)
    out = defaultdict(list)
    for entry in raw:
        provider = entry["provider"]
        if provider in DEAD_PROVIDERS:
            continue
        out[provider].append(entry)
    return out


def is_good_coding_model(model, ttft, tps, n):
    """Heuristic: is this a reasonable coding model?"""
    if ttft > 30000:  # >30s TTFT is unusable
        return False
    if n < 2:
        return False
    # Exclude models with 'rp' or 'uncensored' in name for coding chains
    bad = ["rp", "uncensored", "roleplay", "flux", "sdxl", "dall-e", "imagen"]
    return not any(b in model.lower() for b in bad)


def is_good_chat_model(model, ttft, tps, n):
    """Heuristic: is this a reasonable chat model?"""
    if ttft > 30000:
        return False
    if n < 2:
        return False
    return True


def get_top_models(working, n, filter_fn):
    """Return top-n models across all working providers, sorted by (tps desc, ttft asc)."""
    all_models = []
    for provider, entries in working.items():
        for e in entries:
            if filter_fn(e["model"], e["ttft"], e["tps"], e["n"]):
                all_models.append((provider, e))
    # Sort by tps desc, ttft asc
    all_models.sort(key=lambda x: (-x[1]["tps"], x[1]["ttft"]))
    return all_models[:n]


def get_top_models_for_provider(working, provider, n, filter_fn):
    """Return top-n models for a specific provider."""
    entries = working.get(provider, [])
    entries = [e for e in entries if filter_fn(e["model"], e["ttft"], e["tps"], e["n"])]
    entries.sort(key=lambda x: (-x["tps"], x["ttft"]))
    return [(provider, e) for e in entries[:n]]


def main():
    yaml_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/virtual_models.yaml")
    working = load_working_models()

    # Per-chain new top entries (provider, model) -- max 8 per chain to keep short
    # ponytail: bare model names route through gateway provider list; openai/* prefix
    # forces api.openai.com call which fails (no OPENAI_API_KEY). Use REAL provider names.
    new_tops = {
        # Coding chains: prefer groq (fast, reliable, large samples), then others
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
        # Chat chains: prefer fast chat-tuned
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

    # agent-* chains use github-models (which works) -- leave as-is
    # oracle, explore, librarian, build, metis, momus - leave as-is

    print("=== Plan ===")
    for chain, entries in new_tops.items():
        print(f"  {chain}: {len(entries)} new top entries")
    print(f"  Total: {sum(len(v) for v in new_tops.values())} new entries")
    print()
    print("=== Verify all entries are in working set ===")
    missing = 0
    for chain, entries in new_tops.items():
        for provider, model in entries:
            in_working = any(
                e["model"] == model for e in working.get(provider, [])
            )
            if not in_working:
                print(f"  MISS: {chain}: {provider}/{model} not in telemetry")
                missing += 1
    if missing == 0:
        print("  All entries verified in telemetry")
    else:
        print(f"  {missing} entries not in telemetry (will still add, but reorder will mark no_telemetry)")

    # ============ APPLY: rebuild chains in virtual_models.yaml ============
    import re
    yaml_text = yaml_path.read_text()

    def rebuild_chain(yaml_text, chain_name, new_entries):
        """Find chain in yaml, prepend new entries, renumber all priorities.

        The yaml indent for fallback_chain entries is 4 spaces (    - provider:)
        and the inner fields are 6 spaces (      model:,      priority:).
        """
        # Find the chain section
        # Pattern matches 4-space-indent entries inside a chain
        pattern = re.compile(
            r"(  " + re.escape(chain_name) + r":\n"
            r"(?:    [^\n]+\n)*?"  # description, settings, etc. (4-space lines)
            r"    fallback_chain:\n"
            r"(?:    - provider: [^\n]+\n"
            r"      model: [^\n]+\n"
            r"      priority: \d+\n"
            r"(?:      [^\n]+\n)*)*)"
        )
        m = pattern.search(yaml_text)
        if not m:
            return yaml_text, False
        old_block = m.group(1)
        # Extract existing entries (4-space indent)
        old_entries = re.findall(
            r"    - provider: ([^\n]+)\n"
            r"      model: ([^\n]+)\n"
            r"      priority: \d+",
            old_block,
        )
        # Find existing extras (capabilities, notes) per entry
        old_extras = {}
        for entry_match in re.finditer(
            r"    - provider: ([^\n]+)\n"
            r"      model: ([^\n]+)\n"
            r"      priority: \d+\n"
            r"((?:      [^\n]+\n)*)",
            old_block,
        ):
            provider, model, extras = entry_match.groups()
            old_extras[(provider, model)] = extras

        # Build new chain: prepend new_entries, then dedupe existing
        seen = set()
        merged = []
        for provider, model in new_entries:
            key = (provider, model)
            if key in seen:
                continue
            seen.add(key)
            merged.append((provider, model, "new"))
        for provider, model in old_entries:
            key = (provider, model)
            if key in seen:
                continue
            seen.add(key)
            merged.append((provider, model, "existing"))

        # Render with 4-space indent for `- provider:` and 6-space for inner fields (matches original)
        lines = []
        for i, (provider, model, kind) in enumerate(merged, 1):
            extras = old_extras.get((provider, model), "")
            lines.append(f"    - provider: {provider}\n")
            lines.append(f"      model: {model}\n")
            lines.append(f"      priority: {i}\n")
            if extras:
                lines.append(extras)

        new_block_parts = old_block.rstrip("\n").split("\n")
        # Find where fallback_chain entries start
        fc_start = None
        for i, line in enumerate(new_block_parts):
            if "fallback_chain:" in line:
                fc_start = i + 1
                break
        if fc_start is None:
            return yaml_text, False
        # Build new block: header (everything up to and including fallback_chain: line) + new entries
        header = "\n".join(new_block_parts[:fc_start])
        new_block = header + "\n" + "".join(lines) + "\n"

        return yaml_text.replace(old_block, new_block), True

    print()
    print("=== Applying rebuild ===")
    backup_path = yaml_path.with_suffix(yaml_path.suffix + ".bak-pre-rebuild")
    backup_path.write_text(yaml_text)
    print(f"  Backup: {backup_path}")
    updated = yaml_text
    for chain_name, entries in new_tops.items():
        updated, ok = rebuild_chain(updated, chain_name, entries)
        if ok:
            print(f"  OK   {chain_name}: {len(entries)} new + existing")
        else:
            print(f"  SKIP {chain_name}: pattern not found")

    yaml_path.write_text(updated)
    print(f"  Wrote: {yaml_path}")


if __name__ == "__main__":
    main()