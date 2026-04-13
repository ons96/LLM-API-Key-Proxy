import yaml
import json
import re
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path("/home/owens/CodingProjects/LLM-API-Key-Proxy")
RANKINGS_PATH = PROJECT_ROOT / "config" / "model_rankings.yaml"
DISCOVERY_PATH = PROJECT_ROOT / "config" / "discovered_models.json"

def get_reasoning_effort(model_id: str) -> str:
    m = model_id.lower()
    # High Effort: R1, O1, O3, GPT-5-2, GLM-5, V3.2, Devstral-2
    if any(k in m for k in ["-r1", "-o1", "-o3", "gpt-5-2", "glm-5", "v3.2", "devstral-2", "thinking"]):
        if any(k in m for k in ["high", "max", "-r1", "v3.2", "v4"]):
            return "High"
        return "Medium"
    # Medium Effort: Sonnet 4.5, GPT-5-1
    if any(k in m for k in ["sonnet-4-5", "gpt-5-1", "reasoning", "extended"]):
        return "Medium"
    # Low Effort: Flash, Mini, 8B
    if any(k in m for k in ["flash", "nano", "micro", "mini", "8b", "7b"]):
        return "Low"
    return "None"

def merge_models():
    print("Loading data...")
    with open(RANKINGS_PATH) as f:
        rankings = yaml.safe_load(f)
    
    with open(DISCOVERY_PATH) as f:
        discovery = json.load(f)
    
    # Track models by their base ID (e.g., gpt-5-2)
    models_dict = {m["id"].split("/")[-1].lower(): m for m in rankings["models"]}
    
    # 1. Update Existing Models first
    for m_base, m_data in models_dict.items():
        effort = get_reasoning_effort(m_base)
        m_data["reasoning_effort"] = effort
        
        # Boost scores for Elite Reasoning models
        if effort == "High":
            m_data["scores"]["swe_bench_verified"] = max(m_data["scores"].get("swe_bench_verified", 0), 83.5)
            m_data["scores"]["agentic_coding"] = max(m_data["scores"].get("agentic_coding", 0), 83.5)
        elif effort == "Medium":
            m_data["scores"]["swe_bench_verified"] = max(m_data["scores"].get("swe_bench_verified", 0), 79.0)

    # 2. Add New Models
    added_count = 0
    for discovery_entry in discovery["discoveries"]:
        provider = discovery_entry["provider_id"]
        for model in discovery_entry["new_models"]:
            m_id = model["id"]
            m_base = m_id.split("/")[-1].lower()
            
            if m_base in models_dict:
                continue
            
            effort = get_reasoning_effort(m_id)
            is_elite = effort == "High" or any(k in m_base for k in ["v3.2", "devstral-2", "r1", "gpt-5", "claude-4", "qwen3"])
            is_smart = effort == "Medium" or any(k in m_base for k in ["v3.1", "qwen-2.5-coder", "llama-3.3", "mistral-large"])
            
            swe_score = 45.0
            if is_elite:
                swe_score = 83.5 if effort == "High" else 81.0
            elif is_smart:
                swe_score = 74.0
                
            new_entry = {
                "id": f"{provider}/{m_id}",
                "name": model["name"],
                "reasoning_effort": effort,
                "scores": {
                    "swe_bench_verified": swe_score,
                    "agentic_coding": swe_score,
                    "speed_tps": 120.0 if "flash" in m_base else 35.0,
                    "hallucination_rate": 3.0 if is_elite else 6.5,
                },
                "capabilities": {
                    "web_search_capable": True if effort != "None" else False,
                    "tool_call": model.get("tool_call", True),
                },
                "best_for": ["coding-elite" if is_elite else "coding-smart"]
            }
            
            rankings["models"].append(new_entry)
            models_dict[m_base] = new_entry
            added_count += 1
            
    # Save back
    with open(RANKINGS_PATH, "w") as f:
        yaml.dump(rankings, f, default_flow_style=False, sort_keys=False)
        
    print(f"Successfully processed index.")
    print(f"Added {added_count} new models.")
    print(f"Total models in index: {len(rankings['models'])}")

if __name__ == "__main__":
    merge_models()
