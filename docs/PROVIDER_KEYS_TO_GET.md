# Provider API keys still needed (free providers)

Generated 2026-08-13. Where to put keys: laptop opencode -> `~/.env` (UPPER_SNAKE) or `~/.secrets/<name>` (mode 600); VPS-40 gateway -> `~/LLM-API-Key-Proxy/.env` + `pm2 restart gateway`. Restore archived blocks from `~/CodingProjects/scripts/archived-models/archived-models-providers-20260813T024231Z.json`.

| Provider | Endpoint | Env var | What you get free | Effort |
|---|---|---|---|---|
| meganova | https://api.meganova.ai/v1 | MEGANOVA_API_KEY | Tier 1 daily quota ~550 req/day (resets 19:00 EST); Mistral-Small-3.2-24B-Instruct-2506 = 50/day; Manta Mini/Flash/Pro, GLM-4.7-Flash, DeepSeek-V3-0324-Free | 2 min signup, no CC |
| tencent-tokenhub | https://tokenhub-intl.tencentcloudmaas.com/v1 (Singapore intl) | TENCENT_TOKENHUB_API_KEY | 1M tokens PER MODEL x ~16 models (deepseek-v4-pro/flash, glm-5.2/5.1, kimi-k2.6, minimax-m3/2.7/2.5, hy-mt2-plus...), 90-day validity, promo to 2026-12-31, auto-claim on first API call | 10-15 min: intl.cloud.tencent.com signup (Canada OK, email/phone verify) |
| cohere (VPS-40 gateway) | https://api.cohere.com/v2 | COHERE_API_KEY (placeholder added to gateway .env) | north-mini-code 262K context, code-focused | 2 min (cohere.com free tier) |
| cloudflare-workers-ai | https://api.cloudflare.com/client/v4/accounts/{id}/ai/v1 | CLOUDFLARE_AI_TOKEN + CLOUDFLARE_ACCOUNT_ID | 10k neurons/day, 33 models | ~30 min signup |
| furry.vg | https://ai.furry.vg/v1 | FURRYVG_API_KEY | 136 models catalog: gpt-5.6-luna/terra/sol, claude-opus-4-8 (dash form), grok-4.6, gemini-3.6-flash, qwen3.7-max, deepseek-v4-flash-0731, kimi-k3 | CURRENT KEY CHAT-403 (catalog-only). Make a new key, then restore archived block (fix model IDs to catalog form) |
| qoder | qoder.com PAT -> Docker luka7620/qoder-proxy :3000/v1 | QODER_PERSONAL_ACCESS_TOKEN (+PROXY_API_KEY, DASHBOARD_PASSWORD) | Lite free tier; 14-day Pro trial 300 credits; 800 free Qwen3.8-Max calls promo (claim in-app, expires 2026-09-30) | ~10 min signup + Docker container |
| siliconflow | use https://api.siliconflow.com (global), NOT .cn | SILICONFLOW_API_KEY | Free models (Qwen3-8B, R1-Distill, Nex-N2-Pro promo); .cn requires Chinese real-name verification | 5 min on .com; skip .cn from Canada |

Notes:
- antigravity = OAuth-only (opencode plugin), no API key needed.
- nianhua/tokenlb = one-time finite credits (treat as non-renewable). paxsenix = 500k tok/day free, 5 RPM, no tool-calling.
- VPS-40 gateway .env currently HAS: P0_API_KEY, AGENTROUTER_API_KEY, ATESSA_API_KEY, FREETOKENFAUCET_API_KEY, GRATISFY_NEW_KEY, TOKENROUTER_API_KEY, VIALERA_API_KEY, SEVENTEEN_NAS_API_KEY, NIANHUA_API_KEY, DEXT_API_KEY, NOVITA_API_KEY, QWEN_API_KEY, etc.
- cohere-north-mini-code already referenced in gateway virtual_models.yaml (L59/L245) + router_config.yaml chain; was blocked by forbidden_providers_under_free_mode=[openai,anthropic,cohere] -> cohere removed from that list 2026-08-13.

## 2026-08-13 04:10 UTC updates
- REMOVED from archive: lpgpt.us (no free credits/models), ai.yanproxy.link (no new signups), llama (api.llama.com RETIRED 2026-07-06; replaced by PAID Meta Model API https://api.meta.ai/v1, muse-spark-1.1/1.2, pay-as-you-go -- NOT free).
- MEGANOVA TIERS (for 'normal free user' = TIER 1): free registration (no CC). Tier 1: 550 msgs/day total, RPM 60, TPM 200K, free models = <100B ONLY (Manta-Mini 1.0), free-model RPD 50-500. 'Creator Tier+' = TIER 2 = $1 deposit unlock. Tier 2 free RPD examples: Manta-Mini 500, L3-8B-Stheno 500, MN-Violet-Lotus 500, L3-70B-Euryale 300, L3.3-MS-Nevoria 300, Sapphira-L3.3-70B 300, Mistral-Small-3.2-24B 300, GLM-4.7-Flash 50, Manta-Pro 50. So as Tier 1 free user: 8 non-Creator models available (Manta-Mini, Manta-Flash, Sapphira, MN-Violet-Lotus, L3.3-MS-Nevoria, Mistral-Small-3.2-24B-Instruct-2506, L3-70B-Euryale, L3-8B-Stheno); GLM-4.7-Flash/Manta-Pro/Gemini-3.6-Flash/Gemini-3.5-Flash-Lite need $1 Tier 2. MEGANOVA_API_KEY still needed + restore meganova block from archive.

## 2026-08-13 03:41 UTC updates
- MEGANOVA: KEY NOW PROVIDED + RESTORED in opencode.stock.json (backup .bak.pre-meganova-restore-20260813T034049Z). Key file ~/.secrets/meganova-key. Verified chat 200 on mistralai/Mistral-Small-3.2-24B-Instruct-2506. Catalog 116 models incl deepseek-ai/DeepSeek-V4-Flash-0731, zai-org/GLM-5.2/5.1, deepseek-ai/DeepSeek-V4-Pro/Flash, gemini-3.6-flash (unverified free-tier; do not add until smoke-tested).
- furry.vg new key (sk-zdtl9...): catalog 200/136 models, chat 403 new_api_error insufficient_user_quota 剩余额度 0 — account has ZERO quota. Restore pending account with quota or top-up.
- Qoder promo-cycling: NOT viable (device fingerprint, explicit suspension for extra trial accounts, reactivate=forfeit credits, VM-blocked). Cursor 7-day trial removed 2026-01-13; cycling detected via machine fingerprint + SheerID, device-level lock; bypass tooling exists (go-cursor-help) but ToS violation + account termination risk.
- Tencent TokenHub: user verdict = defer (only ~16 models, some weak, 90-day expiry) unless nothing better.
