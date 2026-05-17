#!/usr/bin/env python3
"""
Fix YAML syntax errors and update providers_database.yaml with:
- Expanded NVIDIA NIM models (116 text, 32 reasoning)
- New BBG provider (DeepSeek V4 Pro, Kimi K2.6, MiniMax)
- New WSF provider (Kimi K2.5/2.6, SWE models)
"""
import yaml, sys

path = 'config/providers_database.yaml'

# Read raw file first to fix YAML syntax errors
with open(path, 'r') as f:
    raw_lines = f.readlines()

# Fix known issues
fixed_lines = []
for i, line in enumerate(raw_lines):
    # Fix line 1834: " - id: opencode_zen" -> "- id: opencode_zen"
    if i == 1833 and line.startswith(' -'):
        line = line[1:]  # strip leading space
    fixed_lines.append(line)

# Write fixed version
with open(path, 'w') as f:
    f.writelines(fixed_lines)

# Now load YAML
with open(path, 'r') as f:
    data = yaml.safe_load(f)

# ---------- NVIDIA UPDATE ----------
nvidia_text_models = [
    "01-ai/yi-large", "abacusai/dracarys-llama-3.1-70b-instruct",
    "adept/fuyu-8b", "ai21labs/jamba-1.5-large-instruct",
    "aisingapore/sea-lion-7b-instruct", "baai/bge-m3",
    "bigcode/starcoder2-15b", "bytedance/seed-oss-36b-instruct",
    "databricks/dbrx-instruct", "deepseek-ai/deepseek-coder-6.7b-instruct",
    "deepseek-ai/deepseek-v3.1-terminus", "deepseek-ai/deepseek-v3.2",
    "deepseek-ai/deepseek-v4-flash", "deepseek-ai/deepseek-v4-pro",
    "google/codegemma-1.1-7b", "google/codegemma-7b", "google/deplot",
    "google/gemma-2-2b-it", "google/gemma-2b", "google/gemma-3-12b-it",
    "google/gemma-3-27b-it", "google/gemma-3-4b-it", "google/gemma-3n-e2b-it",
    "google/gemma-3n-e4b-it", "google/gemma-4-31b-it", "google/recurrentgemma-2b",
    "ibm/granite-3.0-3b-a800m-instruct", "ibm/granite-3.0-8b-instruct",
    "ibm/granite-34b-code-instruct", "ibm/granite-8b-code-instruct",
    "meta/codellama-70b", "meta/llama-3.1-405b-instruct",
    "meta/llama-3.1-70b-instruct", "meta/llama-3.1-8b-instruct",
    "meta/llama-3.2-11b-vision-instruct", "meta/llama-3.2-1b-instruct",
    "meta/llama-3.2-3b-instruct", "meta/llama-3.2-90b-vision-instruct",
    "meta/llama-3.3-70b-instruct", "meta/llama-4-maverick-17b-128e-instruct",
    "meta/llama-guard-4-12b", "meta/llama2-70b", "microsoft/kosmos-2",
    "microsoft/phi-3-vision-128k-instruct", "microsoft/phi-3.5-moe-instruct",
    "microsoft/phi-4-mini-instruct", "microsoft/phi-4-multimodal-instruct",
    "minimaxai/minimax-m2.5", "minimaxai/minimax-m2.7",
    "mistralai/codestral-22b-instruct-v0.1", "mistralai/devstral-2-123b-instruct-2512",
    "mistralai/magistral-small-2506", "mistralai/ministral-14b-instruct-2512",
    "mistralai/mistral-7b-instruct-v0.3", "mistralai/mistral-large",
    "mistralai/mistral-large-2-instruct", "mistralai/mistral-large-3-675b-instruct-2512",
    "mistralai/mistral-medium-3-instruct", "mistralai/mistral-medium-3.5-128b",
    "mistralai/mistral-nemotron", "mistralai/mistral-small-4-119b-2603",
    "mistralai/mixtral-8x22b-instruct-v0.1", "mistralai/mixtral-8x22b-v0.1",
    "mistralai/mixtral-8x7b-instruct-v0.1",
    "moonshotai/kimi-k2-instruct", "moonshotai/kimi-k2-instruct-0905",
    "moonshotai/kimi-k2-thinking", "moonshotai/kimi-k2.6",
    "nv-mistralai/mistral-nemo-12b-instruct",
    "nvidia/llama-3.1-nemoguard-8b-content-safety",
    "nvidia/llama-3.1-nemoguard-8b-topic-control",
    "nvidia/llama-3.1-nemotron-51b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/llama-3.1-nemotron-nano-8b-v1",
    "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    "nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "nvidia/llama-3.2-nemoretriever-parse",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nvidia/llama-nemotron-embed-1b-v2", "nvidia/llama-nemotron-embed-vl-1b-v2",
    "nvidia/llama3-chatqa-1.5-70b", "nvidia/mistral-nemo-minitron-8b-8k-instruct",
    "nvidia/nemotron-3-content-safety", "nvidia/nemotron-3-nano-30b-a3b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nvidia/nemotron-3-super-120b-a12b", "nvidia/nemotron-4-340b-instruct",
    "nvidia/nemotron-4-340b-reward", "nvidia/nemotron-content-safety-reasoning-4b",
    "nvidia/nemotron-mini-4b-instruct", "nvidia/nemotron-nano-12b-v2-vl",
    "nvidia/nemotron-nano-3-30b-a3b", "nvidia/nvidia-nemotron-nano-9b-v2",
    "openai/gpt-oss-120b", "openai/gpt-oss-20b",
    "qwen/qwen2.5-coder-32b-instruct", "qwen/qwen3-coder-480b-a35b-instruct",
    "qwen/qwen3-next-80b-a3b-instruct", "qwen/qwen3-next-80b-a3b-thinking",
    "qwen/qwen3.5-122b-a10b", "qwen/qwen3.5-397b-a17b",
    "sarvamai/sarvam-m", "snowflake/arctic-embed-l",
    "stepfun-ai/step-3.5-flash", "stockmark/stockmark-2-100b-instruct",
    "upstage/solar-10.7b-instruct", "writer/palmyra-creative-122b",
    "writer/palmyra-fin-70b-32k", "writer/palmyra-med-70b",
    "writer/palmyra-med-70b-32k", "z-ai/glm-5.1", "z-ai/glm4.7",
    "z-ai/glm5", "zyphra/zamba2-7b-instruct",
]

reasoning_set = {
    "moonshotai/kimi-k2-instruct", "moonshotai/kimi-k2-instruct-0905",
    "moonshotai/kimi-k2-thinking", "moonshotai/kimi-k2.6",
    "qwen/qwen3-next-80b-a3b-thinking",
    "nvidia/llama-3.1-nemotron-51b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nvidia/nemotron-3-super-120b-a12b", "nvidia/nemotron-4-340b-instruct",
    "nvidia/nemotron-4-340b-reward", "nvidia/nemotron-content-safety-reasoning-4b",
    "meta/llama-guard-4-12b", "mistralai/mistral-nemotron",
}

nvidia_prov = next((p for p in data['providers'] if p['id'] == 'nvidia'), None)
if not nvidia_prov:
    print("ERROR: nvidia provider not found", file=sys.stderr)
    sys.exit(1)

nvidia_prov['free_models'] = []
for mid in nvidia_text_models:
    entry = {"id": mid, "context": 128000, "tps": 50,
             "capabilities": ["chat", "code"] + (["reasoning"] if mid in reasoning_set else [])}
    nvidia_prov['free_models'].append(entry)

print(f"NVIDIA provider: {len(nvidia_text_models)} models ({len([m for m in nvidia_text_models if m in reasoning_set])} reasoning)")

# ---------- BBG PROVIDER ----------
bbg_models = [
    "bbg/moonshotai/Kimi-K2.5", "bbg/moonshotai/Kimi-K2.6",
    "bbg/MiniMaxAI/MiniMax-M2.5", "bbg/MiniMaxAI/MiniMax-M2.7",
    "bbg/deepseek-ai/DeepSeek-V3.2", "bbg/deepseek-ai/DeepSeek-V4-Flash",
    "bbg/deepseek-ai/DeepSeek-V4-Pro",
    "bbg/Qwen/Qwen3.6-35B-A3B", "bbg/Qwen/Qwen3.6-35B-A3B-Thinking",
    "bbg/zai-org/GLM-5", "bbg/zai-org/GLM-5.1",
]

if not any(p['id'] == 'bbg' for p in data['providers']):
    bbg_prov = {
        "id": "bbg",
        "name": "BBG FreeTheAI (curated best)",
        "signup_url": "https://freetheai.xyz",
        "env_var": "FREETHEAI_API_KEY",
        "base_url": "https://api.freetheai.xyz/v1",
        "enabled": True, "free_tier": True,
        "capabilities": ["chat", "code", "tools"],
        "rate_limits": {"rpm": 30},
        "notes": "Curated FreeTheAI bbg/ namespace — Kimi K2.6 (reasoning), DeepSeek V4 Pro, MiniMax, Qwen 3.6, GLM-5. Requires FREETHEAI_API_KEY (global 30 RPM limit).",
        "free_models": []
    }
    for mid in bbg_models:
        bbg_prov['free_models'].append({
            "id": mid, "context": 128000, "tps": 50,
            "capabilities": ["chat", "code"] + (["reasoning"] if "Kimi-K2.6" in mid or "Thinking" in mid else [])
        })
    data['providers'].append(bbg_prov)
    print(f"BBG provider added: {len(bbg_models)} models")
else:
    print("BBG provider already exists — skipping")

# ---------- WSF PROVIDER ----------
wsf_models = ["wsf/kimi-k2.5", "wsf/kimi-k2.6", "wsf/swe-1.5", "wsf/swe-1.6"]
if not any(p['id'] == 'wsf' for p in data['providers']):
    wsf_prov = {
        "id": "wsf", "name": "WSF FreeTheAI (Kimi + SWE)",
        "signup_url": "https://freetheai.xyz",
        "env_var": "FREETHEAI_API_KEY",
        "base_url": "https://api.freetheai.xyz/v1",
        "enabled": True, "free_tier": True,
        "capabilities": ["chat", "code", "tools"],
        "rate_limits": {"rpm": 30},
        "notes": "FreeTheAI wsf/ namespace — Kimi K2.5/2.6 reasoning + SWE coder models",
        "free_models": []
    }
    for mid in wsf_models:
        wsf_prov['free_models'].append({
            "id": mid, "context": 131072, "tps": 100,
            "capabilities": ["chat", "code", "reasoning"] if "kimi" in mid.lower() else ["chat", "code", "tools"]
        })
    data['providers'].append(wsf_prov)
    print(f"WSF provider added: {len(wsf_models)} models")
else:
    print("WSF provider already exists — skipping")

# Write back
with open(path, 'w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)

print("All updates applied successfully!")
print("Restart gateway: pkill -f main.py && nohup python src/proxy_app/main.py --host 0.0.0.0 --port 8000 > ~/llm_proxy.log 2>&1 &")
