#!/usr/bin/env python3
"""
Virtual Model Generator

Generates intelligent fallback chains for virtual models based on:
- SWE-bench / LiveCodeBench (coding benchmarks)
- Artificial Analysis (intelligence, TPS, efficiency)
- UGI Leaderboard (uncensored models for RP)
- models.dev (free provider discovery)

Outputs: config/virtual_models_generated.yaml
"""

import sys
import yaml
import requests
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

WEIGHTS = {
    "coding-elite": {
        "swe_bench": 0.35,
        "livecodebench": 0.25,
        "humaneval": 0.15,
        "tps": 0.10,
        "hallucination_penalty": 0.15,
    },
    "coding-smart": {
        "swe_bench": 0.25,
        "livecodebench": 0.20,
        "humaneval": 0.15,
        "tps": 0.15,
        "agentic": 0.10,
        "hallucination_penalty": 0.15,
    },
    "coding-fast": {
        "tps": 0.55,
        "swe_bench": 0.20,
        "humaneval": 0.10,
        "hallucination_penalty": 0.15,
    },
    "chat-elite": {
        "intelligence": 0.55,
        "mmlu": 0.20,
        "arena_elo": 0.15,
        "hallucination_penalty": 0.10,
    },
    "chat-smart": {
        "intelligence": 0.35,
        "mmlu": 0.15,
        "arena_elo": 0.15,
        "tps": 0.15,
        "hallucination_penalty": 0.20,
    },
    "chat-fast": {
        "tps": 0.60,
        "intelligence": 0.20,
        "hallucination_penalty": 0.20,
    },
    "chat-rp": {
        "tps": 0.50,
        "ugi": 0.25,
        "writing": 0.25,
    },
}

FREE_PROVIDERS = [
    "groq",
    "cerebras",
    "gemini",
    "together",
    "g4f",
    "nvidia",
    "github-models",
    "kilo",
    "modal",
]
RP_PROVIDERS = ["g4f", "together", "deepinfra"]

PROVIDER_PRIORITY = {
    "speed": ["cerebras", "groq", "kilo", "gemini", "together", "g4f"],
    "coding": ["kilo", "groq", "cerebras", "gemini", "together", "openrouter", "g4f"],
    "chat": ["groq", "gemini", "cerebras", "together", "kilo", "g4f"],
    "rp": ["g4f", "together", "deepinfra"],
}


def load_yaml(filename: str) -> Dict:
    path = CONFIG_DIR / filename
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def fetch_models_dev_free() -> List[Dict]:
    try:
        resp = requests.get("https://models.dev/api.json", timeout=60)
        resp.raise_for_status()
        data = resp.json()

        free_models = []
        for provider_id, provider_data in data.items():
            if not isinstance(provider_data, dict):
                continue
            models = provider_data.get("models", {})
            for model_id, model_info in models.items():
                if not isinstance(model_info, dict):
                    continue
                cost = model_info.get("cost", {})
                if cost.get("input", 1) == 0 and cost.get("output", 1) == 0:
                    free_models.append(
                        {
                            "provider": provider_id,
                            "model": model_id,
                            "name": model_info.get("name", model_id),
                            "context": model_info.get("limit", {}).get("context", 0),
                            "supports_tools": model_info.get("tool_call", False),
                        }
                    )
        return free_models
    except Exception as e:
        print(f"Warning: Could not fetch models.dev: {e}")
        return []


def calculate_score(model: Dict, weights: Dict) -> float:
    total = 0.0
    for metric, weight in weights.items():
        if metric == "hallucination_penalty":
            hallucination_rate = model.get("hallucination_rate", 10.0)
            if hallucination_rate > 0:
                normalized_penalty = min(hallucination_rate / 20.0, 1.0)
                total -= normalized_penalty * weight
            continue

        value = model.get(metric, 0)
        if isinstance(value, (int, float)) and value > 0:
            if metric in ["tps", "arena_elo"]:
                normalized = min(value / 200, 1.0)
            elif metric == "efficiency":
                normalized = min(value / 50, 1.0)
            else:
                normalized = min(value / 100, 1.0)
            total += normalized * weight
    return total


def load_coding_models() -> List[Dict]:
    rankings = load_yaml("model_rankings.yaml")
    models = []
    for m in rankings.get("models", []):
        scores = m.get("scores", {})
        model_entry = {
            "id": m.get("id", ""),
            "name": m.get("name", ""),
            "swe_bench": scores.get("swe_bench", scores.get("swe_bench_verified", 0)),
            "livecodebench": scores.get("livebench_coding", 0),
            "humaneval": scores.get("humaneval", 0),
            "tps": scores.get("speed_tps", 50),
            "agentic": scores.get("agentic_coding", 0),
            "hallucination_rate": scores.get("hallucination_rate", 10.0),
            "verified": scores.get("verified", True),
        }

        if not model_entry["verified"]:
            model_entry["swe_bench"] *= 0.5
            model_entry["agentic"] *= 0.5

        models.append(model_entry)
    return models


def load_chat_models() -> List[Dict]:
    rankings = load_yaml("chat_model_rankings.yaml")
    models = []
    for m in rankings.get("models", []):
        models.append(
            {
                "id": m.get("id", ""),
                "name": m.get("name", ""),
                "intelligence": m.get("intelligence_score", 0),
                "mmlu": m.get("intelligence_score", 0) * 0.9,
                "arena_elo": m.get("chat_arena_score", 0) / 15,
                "tps": 1000 / max(m.get("response_time_seconds", 5), 0.5),
                "efficiency": m.get("efficiency_score", 0),
                "hallucination_rate": m.get("hallucination_rate", 10.0),
            }
        )
    return models


def load_ugi_models() -> List[Dict]:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    try:
        from rotator_library.ugi_scraper import get_top_rp_models

        ugi_models = get_top_rp_models(limit=50, min_ugi=25.0)
        models = []
        for m in ugi_models:
            models.append(
                {
                    "id": f"g4f/{m['model']}",
                    "name": m["full_name"],
                    "ugi": m["ugi_score"],
                    "writing": m["writing"],
                    "intelligence": m["natint"],
                }
            )
        return models
    except Exception as e:
        print(f"Warning: Could not load UGI models: {e}")
        return []


def get_provider_for_model(model_id: str, model_type: str) -> str:
    if model_id.startswith("g4f/"):
        return "g4f"

    model_lower = model_id.lower()

    if "gemini" in model_lower:
        return "gemini"
    elif "llama" in model_lower:
        if model_type == "rp":
            return "g4f"
        return "groq"
    elif "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
        return "g4f"
    elif "claude" in model_lower:
        return "g4f"
    elif "qwen" in model_lower or "qwq" in model_lower:
        return "together"
    elif "deepseek" in model_lower:
        return "together"
    elif "mistral" in model_lower:
        return "together"

    return "g4f"


def generate_fallback_chain(
    models: List[Dict],
    weights: Dict,
    model_type: str,
    min_score: float = 0.3,
    max_models: int = 15,
    min_intelligence: float = 0.0,
    min_ugi: float = 0.0,
    min_ugi_entertainment: float = 0.0,
    max_hallucination: float = 100.0,
) -> List[Dict]:
    scored = []
    for m in models:
        if min_intelligence > 0 and m.get("intelligence", 0) < min_intelligence:
            continue
        if min_ugi > 0 and m.get("ugi", 0) < min_ugi:
            continue
        if min_ugi_entertainment > 0 and m.get("writing", 0) < min_ugi_entertainment:
            continue
        if m.get("hallucination_rate", 0) > max_hallucination:
            continue

        score = calculate_score(m, weights)
        if score >= min_score:
            provider = get_provider_for_model(m.get("id", ""), model_type)
            scored.append(
                {
                    "provider": provider,
                    "model": m.get("id", m.get("name", "")).split("/")[-1],
                    "score": score,
                    "original_id": m.get("id", ""),
                }
            )

    scored.sort(key=lambda x: x["score"], reverse=True)

    seen = set()
    chain = []
    for m in scored:
        key = f"{m['provider']}/{m['model']}"
        if key not in seen and m["provider"] in FREE_PROVIDERS:
            seen.add(key)
            chain.append(
                {
                    "provider": m["provider"],
                    "model": m["model"],
                    "priority": len(chain) + 1,
                }
            )
        if len(chain) >= max_models:
            break

    return chain


def generate_virtual_models_yaml() -> Dict:
    print("Loading ranking data...")
    coding_models = load_coding_models()
    chat_models = load_chat_models()
    ugi_models = load_ugi_models()

    print(f"  Coding models: {len(coding_models)}")
    print(f"  Chat models: {len(chat_models)}")
    print(f"  UGI models: {len(ugi_models)}")

    virtual_models = {}

    print("\nGenerating coding-elite...")
    chain = generate_fallback_chain(
        coding_models, WEIGHTS["coding-elite"], "coding", min_score=0.4
    )
    virtual_models["coding-elite"] = {
        "description": "Best agentic coding models (SWE-bench + LiveCodeBench weighted)",
        "fallback_chain": chain,
        "settings": {"timeout_ms": 180000, "retry_on_rate_limit": True},
    }
    print(f"  Generated {len(chain)} models")

    print("\nGenerating coding-smart...")
    chain = generate_fallback_chain(
        coding_models, WEIGHTS["coding-smart"], "coding", min_score=0.35
    )
    virtual_models["coding-smart"] = {
        "description": "High-quality coding with balanced performance",
        "fallback_chain": chain,
        "settings": {"timeout_ms": 120000, "retry_on_rate_limit": True},
    }
    print(f"  Generated {len(chain)} models")

    print("\nGenerating coding-fast...")
    chain = generate_fallback_chain(
        coding_models, WEIGHTS["coding-fast"], "coding", min_score=0.3
    )
    virtual_models["coding-fast"] = {
        "description": "Fastest coding models (TPS priority with quality floor)",
        "fallback_chain": chain,
        "settings": {"timeout_ms": 30000, "retry_on_rate_limit": True},
    }
    print(f"  Generated {len(chain)} models")

    print("\nGenerating chat-smart...")
    chain = generate_fallback_chain(
        chat_models, WEIGHTS["chat-smart"], "chat", min_score=0.4
    )
    virtual_models["chat-smart"] = {
        "description": "Best intelligence-to-speed ratio (smart AND reasonably fast)",
        "fallback_chain": chain,
        "settings": {"timeout_ms": 120000, "retry_on_rate_limit": True},
    }
    print(f"  Generated {len(chain)} models")

    print("\nGenerating chat-elite...")
    chain = generate_fallback_chain(
        chat_models, WEIGHTS["chat-elite"], "chat", min_score=0.5
    )
    virtual_models["chat-elite"] = {
        "description": "Most intelligent models regardless of speed (pure intelligence ranking)",
        "fallback_chain": chain,
        "settings": {"timeout_ms": 300000, "retry_on_rate_limit": True},
    }
    print(f"  Generated {len(chain)} models")

    print("\nGenerating chat-fast...")
    chain = generate_fallback_chain(
        chat_models,
        WEIGHTS["chat-fast"],
        "chat",
        min_score=0.3,
        min_intelligence=30.0,
        max_hallucination=25.0,
    )
    virtual_models["chat-fast"] = {
        "description": "Fastest models that aren't stupid (TPS priority, min intelligence threshold)",
        "fallback_chain": chain,
        "settings": {"timeout_ms": 15000, "retry_on_rate_limit": True},
    }
    print(f"  Generated {len(chain)} models")

    print("\nGenerating chat-rp...")
    if ugi_models:
        chain = generate_fallback_chain(
            ugi_models,
            WEIGHTS["chat-rp"],
            "rp",
            min_score=0.15,
            max_models=20,
            min_ugi=25.0,
            min_ugi_entertainment=15.0,
        )
    else:
        chain = [
            {"provider": "g4f", "model": "mn-violet-lotus-12b", "priority": 1},
            {"provider": "g4f", "model": "cydonia-24b-v4.1", "priority": 2},
            {"provider": "g4f", "model": "anubis-70b-v1.1", "priority": 3},
            {"provider": "g4f", "model": "broken-tutu-24b-unslop-v2.0", "priority": 4},
            {
                "provider": "g4f",
                "model": "pantheon-rp-1.8-24b-small-3.1",
                "priority": 5,
            },
            {"provider": "cerebras", "model": "llama-3.1-8b", "priority": 6},
        ]
    virtual_models["chat-rp"] = {
        "description": "Uncensored RP models (min UGI threshold, then sorted by TPS for fast responses)",
        "fallback_chain": chain,
        "settings": {"timeout_ms": 30000, "retry_on_rate_limit": True},
    }
    print(f"  Generated {len(chain)} models")

    providers = load_yaml("virtual_models.yaml").get("providers", {})

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "sources": [
                "model_rankings.yaml",
                "chat_model_rankings.yaml",
                "UGI Leaderboard",
            ],
        },
        "virtual_models": virtual_models,
        "providers": providers,
    }

    return output


def main():
    print("=" * 70)
    print("VIRTUAL MODEL GENERATOR")
    print("=" * 70)

    output = generate_virtual_models_yaml()

    output_path = CONFIG_DIR / "virtual_models_generated.yaml"
    with open(output_path, "w") as f:
        yaml.dump(
            output, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    print(f"\n✓ Generated: {output_path}")

    print("\n" + "=" * 70)
    print("SUMMARY: TOP 5 MODELS PER VIRTUAL MODEL")
    print("=" * 70)

    for vm_name, vm_data in output["virtual_models"].items():
        print(f"\n{vm_name}:")
        chain = vm_data.get("fallback_chain", [])[:5]
        for i, m in enumerate(chain, 1):
            print(f"  {i}. {m['provider']}/{m['model']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
