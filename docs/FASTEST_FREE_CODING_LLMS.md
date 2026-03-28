# Fastest Free LLMs for Quick/Fast Coding Tasks
Generated: 2026-03-28

## Summary

For quick, fast, accurate coding tasks in opencode/Omo (`coding-fast` category):
Best candidates are **Cerebras** and **Groq** — both free, both have API keys configured, and
their raw TPS dwarfs every other free provider by 3–10x.

---

## Top Tier: Cerebras (2600+ TPS, free, no daily limit)

| Model | TPS | RPM | Daily | Notes |
|---|---|---|---|---|
| `cerebras/llama3.1-8b` | 2600 | 30 | unlimited | Fastest free model anywhere. 8B so lower quality. |
| `cerebras/llama-3.3-70b` | 2100 | 30 | unlimited | Best balance: very fast + capable. **Primary coding-fast pick.** |
| `cerebras/qwen-3-32b` | 1200 | 30 | unlimited | Fast MoE, good at code. |
| `cerebras/qwen-3-235b-a22b-instruct-2507` | 500 | 30 | unlimited | Slower but higher quality reasoning. |
| `cerebras/gpt-oss-120b` | 800 | 30 | unlimited | Large model, still very fast. |
| `cerebras/zai-glm-4.7` | 1000 | 30 | unlimited | Fast, good for structured output. |

**Key constraint**: Cerebras RPM=30 across all models. With 30 RPM, you can sustain ~0.5 req/s
continuously. For burst coding tasks this is fine. For high-volume pipelines it's the bottleneck.

---

## Second Tier: Groq (750 TPS, free, 14,400 daily limit)

| Model | TPS | RPM | Daily | Notes |
|---|---|---|---|---|
| `groq/llama-3.1-8b-instant` | 750 | 30 | 14,400 | Fastest Groq model. Low quality for complex tasks. |
| `groq/meta-llama/llama-4-scout-17b-16e-instruct` | 400 | 30 | 14,400 | Good balance speed+quality. |
| `groq/meta-llama/llama-4-maverick-17b-128e-instruct` | 400 | 30 | 14,400 | Vision + tools capable. |
| `groq/openai/gpt-oss-20b` | 400 | 30 | 14,400 | Fast, good for simple edits. |
| `groq/llama-3.3-70b-versatile` | 275 | 30 | 14,400 | Higher quality, slower but still fast. |
| `groq/moonshotai/kimi-k2-instruct` | 200 | 30 | 14,400 | Best reasoning on Groq, slower. |

**Key constraint**: 14,400 req/day = ~600/hour. Resets at 00:00 UTC.

---

## Third Tier: g4f_groq (750 TPS, no daily limit, no auth key needed)

Same Groq models proxied through g4f.space. No API key = no daily tracking = potentially unlimited
but less reliable. Good as overflow fallback when Groq daily limit hit.

| Model | TPS |
|---|---|
| `g4f_groq/llama-3.1-8b-instant` | 750 |
| `g4f_groq/mixtral-8x7b-32768` | 500 |
| `g4f_groq/llama-3.3-70b-versatile` | 275 |

---

## Fourth Tier: SambaNova (1000 TPS, free, no daily limit)

Not currently in `virtual_models.yaml` or `router_config.yaml` but in `providers_database.yaml`.
Very fast, worth adding to `coding-fast` fallback chain.

| Model | TPS | Notes |
|---|---|---|
| `sambanova/Meta-Llama-3.1-8B-Instruct` | 1000 | Very fast, small |
| `sambanova/Qwen3-32B` | 500 | Good quality |
| `sambanova/Meta-Llama-3.3-70B-Instruct` | 400 | Best quality on SambaNova |

**Action needed**: Add SambaNova provider to `router_config.yaml` and add models to `coding-fast` chain.

---

## Recommended `coding-fast` Fallback Order

Based on TPS, reliability, and coding capability:

```
1. cerebras/llama-3.3-70b          (2100 TPS — best fast+capable combo)
2. cerebras/llama3.1-8b            (2600 TPS — fastest but 8B)
3. cerebras/qwen-3-32b             (1200 TPS — good MoE quality)
4. groq/llama-3.1-8b-instant       (750 TPS — Groq fastest)
5. groq/meta-llama/llama-4-scout-17b-16e-instruct  (400 TPS — better quality)
6. groq/openai/gpt-oss-20b         (400 TPS — reliable)
7. groq/llama-3.3-70b-versatile    (275 TPS — quality fallback)
8. g4f_groq/llama-3.1-8b-instant   (750 TPS — Groq overflow)
9. gemini/gemini-2.0-flash         (150 TPS — broader knowledge)
10. g4f/gemini-2.0-flash           (150 TPS — Gemini overflow)
```

This matches the current `coding-fast` virtual model in `virtual_models.yaml` closely,
with the addition of `cerebras/qwen-3-32b` and `groq/openai/gpt-oss-20b` as recommended additions.

---

## Models to AVOID for coding-fast

| Model | Why |
|---|---|
| Any Gemini Pro/2.5-Pro | TPS=50-75, slow for fast tasks |
| Any DeepSeek-R1 | TPS=50, reasoning overhead |
| `cloudflare/*` | TPS=300-500 but 10K daily cap and low quality |
| Any model via `together` | TPS=150-200, rate limits tight |

---

## For Omo `build`/`ultrawork`/`sisyphus` agents (NOT coding-fast)

These agents need accuracy over speed. Current config uses `google/antigravity-gemini-3.1-pro`
as primary, which is correct. The fallback chain in `omo-config-template.json` correctly uses
`supacoder/*` models (gpt-5.4, gpt-5.3-codex, etc.) which are high quality.

The `coding-fast` virtual model should NOT be used for complex autonomous coding tasks.
Use `coding-elite` or `coding-smart` for those.

