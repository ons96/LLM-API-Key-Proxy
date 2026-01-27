import pandas as pd
import yaml
import os
import re
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("sync_leaderboard")

LEADERBOARD_CSV = (
    "/home/owens/CodingProjects/llm-leaderboard/llm_aggregated_leaderboard.csv"
)
RANKINGS_YAML = (
    "/home/owens/CodingProjects/LLM-API-Key-Proxy/config/model_rankings.yaml"
)

ALL_BENCHMARKS = [
    "livebench_coding",
    "aider",
    "swe_bench",
    "swe_rebench",
    "humaneval",
    "bigcodebench",
    "gso_bench",
    "ts_bench",
    "vals_ai",
]

# Comprehensive mapping to LiteLLM provider/model format
LITELLM_MAP = {
    # Claude 4.x Series
    "claude 4.5 opus thinking high effort": ("anthropic", "claude-opus-4-5-thinking"),
    "claude 4.5 opus medium effort": ("anthropic", "claude-opus-4-5-thinking"),
    "claude 4.5 opus thinking": ("anthropic", "claude-opus-4-5-thinking"),
    "claude 4.5 opus": ("anthropic", "claude-opus-4-5"),
    "claude 4.5 sonnet thinking": ("anthropic", "claude-sonnet-4-5-thinking"),
    "claude 4.5 sonnet": ("anthropic", "claude-sonnet-4-5"),
    "claude 4.5 haiku thinking": ("anthropic", "claude-haiku-4-5"),
    "claude 4.5 haiku": ("anthropic", "claude-haiku-4-5"),
    "claude 4 opus thinking": ("anthropic", "claude-opus-4"),
    "claude 4 opus": ("anthropic", "claude-opus-4"),
    "claude 4 sonnet thinking": ("anthropic", "claude-sonnet-4"),
    "claude 4 sonnet": ("anthropic", "claude-sonnet-4"),
    "claude 4.1 opus thinking": ("anthropic", "claude-opus-4-1"),
    "claude 4.1 opus": ("anthropic", "claude-opus-4-1"),
    "claude 4.1 sonnet": ("anthropic", "claude-sonnet-4-1"),
    # Claude 3.x Series
    "claude 3.7 sonnet extended thinking": ("anthropic", "claude-3-7-sonnet"),
    "claude 3.7 sonnet standard": ("anthropic", "claude-3-7-sonnet"),
    "claude 3.5 sonnet": ("anthropic", "claude-3-5-sonnet"),
    "claude 3.5 haiku": ("anthropic", "claude-3-5-haiku"),
    "claude 3 opus": ("anthropic", "claude-3-opus"),
    "claude code": ("anthropic", "claude-code"),
    # GPT-5 Series
    "gpt-5.2 (2025-12-11) (high reasoning)": ("openai", "gpt-5-2"),
    "gpt-5.2 (2025-12-11)": ("openai", "gpt-5-2"),
    "gpt-5.2 high": ("openai", "gpt-5-2"),
    "gpt-5.2 no thinking": ("openai", "gpt-5-2"),
    "gpt-5.2": ("openai", "gpt-5-2"),
    "gpt-5.1 codex max xhigh": ("openai", "gpt-5-1"),
    "gpt-5.1 codex max": ("openai", "gpt-5-1"),
    "gpt-5.1 codex": ("openai", "gpt-5-1"),
    "gpt-5.1 no thinking": ("openai", "gpt-5-1"),
    "gpt-5.1": ("openai", "gpt-5-1"),
    "gpt-5": ("openai", "gpt-5"),
    "gpt-4-1": ("openai", "gpt-4-1"),
    "gpt-4o": ("openai", "gpt-4o"),
    # Gemini 3.x Series
    "gemini 3 pro preview (2025-11-18)": ("google", "gemini-3-pro"),
    "gemini 3 pro preview high": ("google", "gemini-3-pro"),
    "gemini 3 pro preview": ("google", "gemini-3-pro"),
    "gemini 3 pro": ("google", "gemini-3-pro"),
    "gemini 3 flash preview high": ("google", "gemini-3-flash"),
    "gemini 3 flash preview": ("google", "gemini-3-flash"),
    "gemini 3 flash": ("google", "gemini-3-flash"),
    "gemini 3 pro (11/25)": ("google", "gemini-3-pro"),
    # Gemini 2.x Series
    "gemini 2.5 pro (max thinking)": ("google", "gemini-2-5-pro"),
    "gemini 2.5 pro": ("google", "gemini-2-5-pro"),
    "gemini 2.5 flash (max thinking) (2025-09-25)": ("google", "gemini-2-5-flash"),
    "gemini 2.5 flash lite (max thinking) (2025-06-17)": ("google", "gemini-2-5-flash"),
    "gemini 2.5 flash": ("google", "gemini-2-5-flash"),
    # Gemini 1.x Series
    "gemini 1.5 pro": ("google", "gemini-1-5-pro"),
    "gemini 1.5 flash": ("google", "gemini-1-5-flash"),
    # Groq/Llama
    "llama 3.3 70b": ("groq", "llama-3-3-70b-versatile"),
    "llama 3.1 8b": ("groq", "llama-3-1-8b-instant"),
    # DeepSeek
    "deepseek v3.2 thinking": ("deepseek", "deepseek-chat"),
    "deepseek v3.2 exp thinking": ("deepseek", "deepseek-chat"),
    "deepseek v3.2 exp": ("deepseek", "deepseek-chat"),
    "deepseek v3.2": ("deepseek", "deepseek-chat"),
    # Grok
    "grok-4": ("grok", "grok-4"),
    "grok code fast": ("grok", "grok-code-fast"),
    # Qwen
    "qwen 3 235b a22b instruct 2507": ("qwen", "qwen-plus"),
    "qwen 3 235b a22b thinking 2507": ("qwen", "qwen-plus"),
    "qwen 3 235b a22b instruct": ("qwen", "qwen-plus"),
    "qwen 3 235b a22b thinking": ("qwen", "qwen-plus"),
    "qwen 3 next 80b a3b instruct": ("qwen", "qwen-plus"),
    "qwen 3 32b": ("qwen", "qwen-plus"),
    "qwen 3": ("qwen", "qwen-plus"),
    # GLM
    "glm-4.7": ("cerebras", "glm-4-7b"),
    "glm-4.6": ("cerebras", "glm-4-6b"),
    # Devstral
    "devstral": ("mistral", "devstral"),
    "devstral small": ("mistral", "devstral-small"),
    # Kimi
    "kimi k2 thinking": ("moonshot", "kimi-k2-thinking"),
}

PROVIDER_PREFIXES = {
    "anthropic/": "anthropic",
    "openai/": "openai",
    "google/": "google",
    "groq/": "groq",
    "deepseek/": "deepseek",
    "mistral/": "mistral",
    "cerebras/": "cerebras",
    "qwen/": "qwen",
    "grok/": "grok",
    "opencode/": "opencode",
    "moonshot/": "moonshot",
}

# Speed data (tokens per second) from artificial analysis and benchmark sources
# Sources: LiveBench performance data, provider documentation, third-party benchmarks
MODEL_SPEEDS = {
    # OpenAI models - fastest providers
    "openai/gpt-5-2": 187.0,  # GPT-5.2 - fastest model (3.8x faster than Claude)
    "openai/gpt-5-1": 150.0,  # GPT-5.1 - very fast
    "openai/gpt-5": 120.0,  # GPT-5 - fast
    "openai/gpt-4-1": 100.0,  # GPT-4.1 - moderate
    "openai/gpt-4o": 80.0,  # GPT-4o - moderate
    # Anthropic models - slower but more intelligent
    "anthropic/claude-opus-4-5-thinking": 49.0,  # Claude 4.5 Opus Thinking - slower but best coding
    "anthropic/claude-opus-4-5": 49.0,  # Claude 4.5 Opus - slower but best coding
    "anthropic/claude-sonnet-4-5-thinking": 52.0,  # Claude Sonnet 4.5 Thinking
    "anthropic/claude-sonnet-4-5": 52.0,  # Claude Sonnet 4.5
    "anthropic/claude-haiku-4-5": 65.0,  # Claude Haiku 4.5 - faster
    "anthropic/claude-opus-4-1": 45.0,  # Claude 4.1 Opus
    "anthropic/claude-sonnet-4-1": 48.0,  # Claude 4.1 Sonnet
    "anthropic/claude-opus-4": 42.0,  # Claude 4 Opus
    "anthropic/claude-sonnet-4": 45.0,  # Claude 4 Sonnet
    "anthropic/claude-3-7-sonnet": 40.0,  # Claude 3.7 Sonnet
    "anthropic/claude-3-5-sonnet": 38.0,  # Claude 3.5 Sonnet
    "anthropic/claude-code": 45.0,  # Claude Code
    # Google Gemini models - variable speeds
    "google/gemini-3-pro": 75.0,  # Gemini 3 Pro - moderate
    "google/gemini-3-flash": 120.0,  # Gemini 3 Flash - fast
    "google/gemini-2-5-pro": 60.0,  # Gemini 2.5 Pro - moderate
    "google/gemini-2-5-flash": 100.0,  # Gemini 2.5 Flash - fast
    # DeepSeek - fast and cost-effective
    "deepseek/deepseek-chat": 90.0,  # DeepSeek V3.2 - fast
    # Grok - moderate speeds
    "grok/grok-4": 70.0,  # Grok 4
    "grok/grok-code-fast": 95.0,  # Grok Code Fast - optimized for speed
    # Other providers
    "mistral/devstral": 55.0,  # Devstral 2
    "cerebras/glm-4-7b": 45.0,  # GLM 4.7
    "qwen/qwen-plus": 60.0,  # Qwen 3
    "moonshot/kimi-k2-thinking": 50.0,  # Kimi K2 Thinking
    "opencode/grok-code": 70.0,  # OpenCode Grok Code
}


def get_litellm_id(name):
    if not name or not isinstance(name, str):
        return None

    name_lower = name.lower().strip()
    original_name = name

    # Check for existing provider prefix
    for prefix, provider in PROVIDER_PREFIXES.items():
        if prefix in name_lower:
            model_part = name_lower.replace(prefix, "").strip()
            return f"{provider}/{model_part}"

    # Try to match known patterns
    for pattern, (provider, model_name) in LITELLM_MAP.items():
        if pattern in name_lower:
            return f"{provider}/{model_name}"

    # Generic fallback - extract vendor and create reasonable ID
    cleaned = name_lower
    cleaned = re.sub(r"\(.*?\)", "", cleaned)
    cleaned = re.sub(r"\d{8}", "", cleaned)
    cleaned = re.sub(r"\d{4}-\d{2}-\d{4}", "", cleaned)
    cleaned = re.sub(r"[^a-z0-9]", "-", cleaned)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")

    # Detect vendor
    if "anthropic" in cleaned or "claude" in cleaned:
        return f"anthropic/{cleaned}"
    elif "openai" in cleaned or "gpt" in cleaned:
        return f"openai/{cleaned}"
    elif "google" in cleaned or "gemini" in cleaned:
        return f"google/{cleaned}"
    elif "groq" in cleaned or "llama" in cleaned:
        return f"groq/{cleaned}"
    elif "deepseek" in cleaned:
        return f"deepseek/{cleaned}"
    elif "mistral" in cleaned or "devstral" in cleaned:
        return f"mistral/{cleaned}"
    elif "cerebras" in cleaned or "glm" in cleaned:
        return f"cerebras/{cleaned}"
    elif "qwen" in cleaned:
        return f"qwen/{cleaned}"
    elif "grok" in cleaned:
        return f"grok/{cleaned}"
    elif "opencode" in cleaned:
        return f"opencode/{cleaned}"
    elif "kimi" in cleaned:
        return f"moonshot/{cleaned}"

    if len(cleaned) < 5:
        return None

    return cleaned


def sync_rankings():
    logger.info("Loading local benchmark data...")
    live_entries = []

    if os.path.exists(LEADERBOARD_CSV):
        df_local = pd.read_csv(LEADERBOARD_CSV)
        for _, row in df_local.iterrows():
            live_entries.append(
                {
                    "model": row["Model"],
                    "score": row["Score"],
                    "benchmark": row["Header"],
                }
            )
    else:
        logger.warning(f"Local leaderboard not found at {LEADERBOARD_CSV}")
        return

    logger.info(f"Loaded {len(live_entries)} entries from local data")

    model_stats = {}

    benchmark_map = {
        "LiveBench Coding": "livebench_coding",
        "Aider Score": "aider",
        "SWE-bench Bash": "swe_bench",
        "SWE-bench Verified": "swe_bench",  # SWE-bench Verified is the gold standard - prioritize this
        "SWE-rebench Resolved": "swe_rebench",
        "HumanEval": "humaneval",
        "BigCodeBench": "bigcodebench",
        "GSO Opt@1": "gso_bench",
        "TS Bench Success Rate": "ts_bench",
        "Vals SWE-bench": "vals_ai",
        "Agentic Coding": "agentic_coding",
        "Vals Terminal": "vals_ai_terminal",
        "Vals LCB": "vals_ai_lcb",
    }

    for entry in live_entries:
        litellm_id = get_litellm_id(entry["model"])
        if not litellm_id:
            continue

        if litellm_id not in model_stats:
            model_stats[litellm_id] = {
                "scores": {},
                "speed_tps": MODEL_SPEEDS.get(
                    litellm_id, 50.0
                ),  # Use speed lookup or default
                "original_name": entry["model"],
            }

        score = entry["score"]
        bench_name = entry["benchmark"]

        for bench_key, target_field in benchmark_map.items():
            if bench_key in bench_name:
                if target_field not in model_stats[litellm_id]["scores"]:
                    model_stats[litellm_id]["scores"][target_field] = []
                if score > 0:
                    model_stats[litellm_id]["scores"][target_field].append(score)
                break

    final_leaderboard = {}
    for litellm_id, stats in model_stats.items():
        bench_scores = {}

        # Take max score per benchmark category (don't average multiple scores from same benchmark)
        for k, v in stats["scores"].items():
            if v:
                bench_scores[k] = max(v)  # Max per benchmark category

        coding_benchmarks = [
            "livebench_coding",
            "aider",
            "swe_bench",
            "swe_rebench",
            "humaneval",
            "bigcodebench",
            "gso_bench",
            # "ts_bench",  # Removed - not a real agentic coding benchmark, inflates scores artificially
            "vals_ai",
        ]

        # For agentic coding score, prioritize the highest individual benchmark
        # This ensures that models like Claude Opus 4.5 with 80.9% SWE-bench are ranked #1
        available_scores = [
            bench_scores[f]
            for f in coding_benchmarks
            if f in bench_scores and bench_scores[f] > 0
        ]

        if available_scores:
            # Use the highest score for agentic coding (most representative of true capability)
            agentic_score = max(available_scores)
        else:
            agentic_score = 0.0

        # Calculate a combined quality/speed score for rankings
        # Normalize TPS: 1 to 3000 (Cerebras max) -> 0.0 to 1.0 using log scale
        import math

        tps = stats["speed_tps"]
        normalized_tps = min(1.0, math.log(max(1.0, tps)) / math.log(3000))

        # Combined Score = (Quality * 0.8) + (Speed * 0.2)
        # This prioritizes intelligence but gives a boost to ultra-fast models
        # Formula: Score = (Agentic_Coding * 0.8) + (Normalized_TPS * 20)
        composite_score = (agentic_score * 0.8) + (normalized_tps * 20.0)

        final_leaderboard[litellm_id] = {
            "agentic_coding": round(agentic_score, 2),
            "composite_score": round(composite_score, 2),
            "all_scores": bench_scores,
            "speed_tps": stats["speed_tps"],
            "name": stats["original_name"],
        }

    # Sort by composite score for the prioritized fallback order
    sorted_models = sorted(
        final_leaderboard.items(), key=lambda x: x[1]["composite_score"], reverse=True
    )

    updated_models = []
    for litellm_id, data in sorted_models:
        if data["agentic_coding"] < 30:
            continue

        scores = {"speed_tps": data["speed_tps"]}

        for benchmark in ALL_BENCHMARKS:
            if benchmark in data["all_scores"] and data["all_scores"][benchmark] > 0:
                scores[benchmark] = round(data["all_scores"][benchmark], 2)
            else:
                scores[benchmark] = -1

        scores["agentic_coding"] = data["agentic_coding"]
        scores["composite_score"] = data["composite_score"]

        if scores.get("humaneval", -1) == -1:
            for b in ALL_BENCHMARKS:
                if scores.get(b, -1) > 0:
                    scores["humaneval"] = scores[b]
                    break

        best_for = []
        if data["agentic_coding"] >= 70:
            best_for = ["coding-smart", "coding-elite"]
        elif data["agentic_coding"] >= 50:
            best_for = ["coding-fast"]
        elif data["agentic_coding"] >= 30:
            best_for = ["coding-budget"]

        updated_models.append(
            {
                "id": litellm_id,
                "name": data["name"],
                "scores": scores,
                "best_for": best_for,
            }
        )

    logger.info(f"Sync complete. {len(updated_models)} models ranked.")

    update_router_config(updated_models)

    with open(RANKINGS_YAML, "w") as f:
        yaml.dump({"models": updated_models}, f, sort_keys=False)

    rows = []
    for m in updated_models:
        s = m.get("scores", {})
        rows.append(
            {
                "ID": m.get("id"),
                "Name": m.get("name", ""),
                "Composite Score": s.get("composite_score", 0),
                "Agentic Coding": s.get("agentic_coding", 0),
                "HumanEval": s.get("humaneval", 0),
                "LiveBench": s.get("livebench_coding", 0),
                "Aider": s.get("aider", 0),
                "SWE-bench": s.get("swe_bench", 0),
                "SWE-rebench": s.get("swe_rebench", 0),
                "GSO-bench": s.get("gso_bench", 0),
                "TS-bench": s.get("ts_bench", 0),
                "Vals.ai": s.get("vals_ai", 0),
                "Speed (TPS)": s.get("speed_tps", 0),
            }
        )

    df = pd.DataFrame(rows)
    df.sort_values(by="Composite Score", ascending=False).to_csv(
        "/home/owens/CodingProjects/LLM-API-Key-Proxy/model_rankings_review.csv",
        index=False,
    )
    print("CSV regenerated with correct LiteLLM IDs and rankings")


def update_router_config(updated_models):
    config_path = (
        "/home/owens/CodingProjects/LLM-API-Key-Proxy/config/router_config.yaml"
    )

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load router_config.yaml: {e}")
        return

    def build_candidates(models_list, max_items=10):
        candidates = []
        for i, m in enumerate(models_list[:max_items]):
            parts = m["id"].split("/", 1)
            provider = parts[0]
            model = parts[1] if len(parts) > 1 else parts[0]

            candidate = {
                "provider": provider,
                "model": model,
                "priority": i + 1,
                "capabilities": ["tools", "function_calling"],
            }
            if i > 0:
                candidate["fallback_only"] = True

            if m["scores"].get("speed_tps", 0) > 100:
                candidate["capabilities"].append("fast")
            if "vision" in m["id"]:
                candidate["capabilities"].append("vision")
            if "reasoning" in m["id"] or "thinking" in m["id"]:
                candidate["capabilities"].append("reasoning")

            candidates.append(candidate)
        return candidates

    smart_models = [m for m in updated_models if m["scores"]["agentic_coding"] >= 60]
    fast_models = [
        m
        for m in updated_models
        if m["scores"]["speed_tps"] > 80 and m["scores"]["agentic_coding"] >= 40
    ]
    chat_smart_models = [
        m for m in updated_models if m["scores"]["agentic_coding"] >= 50
    ]
    chat_fast_models = [
        m
        for m in updated_models
        if m["scores"]["speed_tps"] > 100 and m["scores"]["agentic_coding"] >= 30
    ]

    if "router_models" not in config:
        config["router_models"] = {}

    def update_section(key, models, description):
        existing = config["router_models"].get(key, {})
        existing["description"] = description
        existing["candidates"] = build_candidates(models)
        config["router_models"][key] = existing

    update_section(
        "coding-smart",
        smart_models,
        "Best models for complex coding tasks (Auto-synced)",
    )
    update_section(
        "coding-fast",
        fast_models,
        "Fastest models for quick coding tasks (Auto-synced)",
    )
    update_section(
        "chat-smart", chat_smart_models, "High intelligence chat models (Auto-synced)"
    )
    update_section(
        "chat-fast", chat_fast_models, "Low latency chat models (Auto-synced)"
    )

    try:
        with open(config_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)
        logger.info(f"Updated {config_path}")
    except Exception as e:
        logger.error(f"Failed to write router_config.yaml: {e}")


if __name__ == "__main__":
    sync_rankings()
