import pandas as pd
import yaml
import os
import re

LEADERBOARD_CSV = (
    "/home/owens/CodingProjects/llm-leaderboard/llm_aggregated_leaderboard.csv"
)
RANKINGS_YAML = (
    "/home/owens/CodingProjects/LLM-API-Key-Proxy/config/model_rankings.yaml"
)


def normalize_model_name(name):
    if not name or not isinstance(name, str):
        return ""
    name = name.lower()
    name = (
        name.replace("anthropic/", "")
        .replace("google/", "")
        .replace("groq/", "")
        .replace("openai/", "")
    )
    name = (
        name.replace("anthropic-", "")
        .replace("google-", "")
        .replace("groq-", "")
        .replace("openai-", "")
    )
    name = re.sub(r"\d{8}", "", name)
    name = re.sub(r"\d{4}-\d{2}-\d{4}", "", name)
    name = re.sub(r"[^a-z0-9]", "-", name)
    name = re.sub(r"-+", "-", name).strip("-")
    return name


benchmark_map = {
    "LiveBench": "livebench_coding",
    "Aider": "aider",
    "SWE-bench": "SWE-bench",
    "SWE-rebench": "swe_rebench",
    "HumanEval": "humaneval",
    "BigCodeBench": "bigcodebench",
    "GSO-bench": "gso_bench",
    "TS-bench": "ts_bench",
    "Vals.ai": "vals_ai",
    "Agentic Coding": "agentic_coding",
    "Chatbot Arena": "intel_score",
    "LMArena": "intel_score",
}

df_local = pd.read_csv(LEADERBOARD_CSV)
print("Columns in local CSV:", df_local.columns.tolist())
print("Sample data:")
print(df_local.head(10))
print("\nUnique headers:")
print(df_local["Header"].unique())
