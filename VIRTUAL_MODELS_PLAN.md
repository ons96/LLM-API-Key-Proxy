# Virtual Models Integration Plan

**Last Updated:** 2026-02-24  
**Status:** Planning/Brainstorming

---

## Overview

The virtual models system routes requests to different LLM providers with automatic fallback chains. Each virtual model is optimized for a specific use case and uses data-driven model rankings based on benchmarks and performance metrics.

---

## Virtual Models Specification

| Virtual Model | Use Case | Primary Metric | Secondary Metric | Fallback Behavior |
|--------------|----------|----------------|------------------|-------------------|
| `coding-elite` | High-level planning, complex coding | Agentic coding (SWE-bench) | TPS (80/15/5 weighting) | Fail if below threshold |
| `coding-fast` | Greps, file searches, quick edits | TPS | Sufficient intelligence | Min context: 32k, min SWE: 50 |
| `chat-smart` | Reasoning, complex chat | Intelligence (MMLU/Arena) | TPS penalty | High intelligence, acceptable speed |
| `chat-fast` | Quick Q&A, simple responses | TPS | Minimum intelligence | Fast + can trigger web search |
| `chat-elite` | Best reasoning only | Intelligence | Ignores speed | Top 5 models by reasoning only |
| `chat-rp` | Uncensored roleplay | Writing quality | TPS | Refusal rate + UGI leaderboard |

---

## Data Sources

### Primary Sources (Free APIs)

| Source | URL | Data Provided | Use Case |
|--------|-----|---------------|-----------|
| **Artificial Analysis** | https://artificialanalysis.ai/leaderboards/providers | TPS, latency, pricing | TPS metrics |
| **UGI Leaderboard** | https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard | Roleplay rankings | chat-rp |
| **OpenRouter** | https://openrouter.ai/rankings | Benchmark scores | Agentic coding |
| **LiveCodeBench** | https://livecodebench.github.io/ | Coding performance | coding-elite |
| **Arena-Hard** | https://lmarena.ai/ | Chat intelligence | chat-smart |

### Model List Sources

- OpenRouter API (free): `GET https://openrouter.ai/api/v1/models`
- Provider APIs: Groq, Gemini, Together direct APIs

---

## Scoring Formulas

### coding-elite
```
Score = (Agentic_Score * 0.80) + (TPS_Normalized * 0.15) + (Availability * 0.05)
```
- Min threshold: 70.0 SWE-bench
- Agentic_Score: From LiveCodeBench or OpenRouter
- TPS_Normalized: 0-100 scale based on provider TPS

### coding-fast
```
Score = TPS_Normalized * 0.70 + Availability * 0.30
```
- Hard filters: Context < 32k → exclude, SWE < 50 → exclude
- Min intelligence: Can trigger web search (sufficient for grep/file search)

### chat-smart
```
Score = Intelligence_Score * 0.75 + (1 / Response_Time_Normalized) * 0.25
```
- Intelligence: Arena-Hard + MMLU combined
- Penalty: Models > 60s first token → -10% score

### chat-fast
```
Score = TPS_Normalized * 0.85 + Min_Intelligence * 0.15
```
- Min_Intelligence: Can determine when to use web search
- Exclude: Models < 10 TPS regardless of intelligence

### chat-elite
```
Score = Intelligence_Score * 1.0
```
- Ignore speed completely
- Top 20 models by Arena-Hard only

### chat-rp
```
Score = (UGI_Score * 0.50) + (TPS_Normalized * 0.30) + (Writing_Quality * 0.20)
```
- UGI_Score: From HuggingFace leaderboard
- Writing_Quality: Based on output length capability + creativity scores
- Filter: Refusal rate > 5% → exclude

---

## Data Normalization

### Missing Data Handling

When a model has no score on a specific benchmark:

1. **Same family fallback**: Use score from related model (e.g., `gpt-4` → `gpt-4-turbo`)
2. **Provider median**: Use median score from same provider family
3. **Last resort**: Place at end of rankings with `score: null` + warning log

### Multi-Benchmark Aggregation

For models with multiple benchmark scores:

```
Normalized_Score = Σ(Benchmark_Score × Benchmark_Weight) / Σ(Applied_Weights)
```

Where weights are:
- SWE-bench: 0.40
- LiveCodeBench: 0.30
- HumanEval: 0.20
- MBPP: 0.10

---

## Implementation Plan

### Phase 1: Data Collection (Priority: High)

- [ ] Create `src/proxy_app/data_collector.py`
  - OpenRouter API client (free, no key needed for rankings)
  - Artificial Analysis scraper
  - UGI Leaderboard scraper
- [ ] Create `src/proxy_app/benchmark_normalizer.py`
  - Normalize scores from different benchmarks
  - Handle missing data
  - Calculate composite scores

### Phase 2: Dynamic Rankings (Priority: High)

- [ ] Update `virtual_models.yaml` to use dynamic endpoints
- [ ] Create `/api/v1/models/rankings` endpoint
- [ ] Implement 6-hour refresh interval
- [ ] Add provider model list refresh

### Phase 3: Fallback Chains (Priority: Medium)

- [ ] Refactor `config/virtual_models.yaml` to use weighted scoring
- [ ] Add threshold filtering
- [ ] Implement "soft" vs "hard" fallback rules

### Phase 4: New chat-rp Model (Priority: Medium)

- [ ] Add `chat-rp` virtual model
- [ ] Integrate UGI Leaderboard data
- [ ] Test with uncensored models (when available)

---

## File Locations

| Component | File | Notes |
|-----------|------|-------|
| Data collector | `src/proxy_app/data_collector.py` | New file |
| Benchmark normalizer | `src/proxy_app/benchmark_normalizer.py` | New file |
| Virtual models config | `config/virtual_models.yaml` | Update existing |
| Router logic | `src/proxy_app/router_core.py` | May need updates |
| Rankings endpoint | `src/proxy_app/main.py` | Add new endpoint |

---

## Conflicting Specs (RESOLVED)

| Aspect | DESIGN.md (old) | INTEGRATION_ROADMAP.md (new) | Resolution |
|--------|-----------------|------------------------------|------------|
| Weights | 60/30/10 | 80/15/5 | Use 80/15/5 (newer) |
| Thresholds | None | 70.0 / 65.0 min | Add thresholds |
| Zen Provider | Not mentioned | Add as baseline | Add later |

---

## Open Questions

1. **chat-elite vs chat-smart**: Should chat-elite be removed in favor of just chat-smart with higher intelligence weight?
2. **Refresh interval**: 6 hours too frequent? Consider 24 hours + manual refresh endpoint
3. **Provider rate limits**: How to track dynamic rate limits in real-time?
4. **chat-rp models**: Most leaderboards don't include uncensored models. Sources?

---

## References

- INTEGRATION_ROADMAP.md (primary spec)
- DESIGN.md (legacy spec, partially obsolete)
- AGENTS.md (project overview)
