# Four-way provider/model diff

Generated: 2026-10-20T22:31:25Z

Columns: db = llm_providers.db, cfg = gateway YAML, live = provider /v1/models today,
ans = models that answered a 1-token chat request.

| provider | type | enabled | db | cfg | live | ans | db-only | live-only | cfg-only | live status |
|---|---|---|---|---|---|---|---|---|---|---|
| openrouter | fully_free | yes | 365 | 3 | 468 | 1 | 124 | 227 | 0 | 200 |
| nous-inference | free_tier_limited | yes | 0 | 8 | 426 | 0 | 0 | 426 | 8 | 200 |
| koyeb | free_one_time | yes | 200 | 0 | 396 | 0 | 43 | 239 | 0 | 200 |
| koyeb-new | free_tier_limited | yes | 0 | 0 | 396 | 0 | 0 | 396 | 0 | 200 |
| g4f | fully_free | yes | 178 | 0 | 223 | 0 | 166 | 211 | 0 | 200 |
| g4f_nvidia | fully_free | yes | 20 | 20 | 81 | 0 | 18 | 79 | 0 | 200 |
| nvidia | fully_free | yes | 126 | 34 | 81 | 0 | 60 | 15 | 7 | 200 |
| g4f_gemini | fully_free | yes | 54 | 3 | 61 | 0 | 16 | 23 | 0 | 200 |
| mistral | fully_free | yes | 65 | 3 | 53 | 1 | 28 | 16 | 0 | 200 |
| atessa | free_daily | yes | 0 | 0 | 33 | 0 | 0 | 33 | 0 | 200 |
| pooled | free_daily | yes | 25 | 0 | 25 | 0 | 0 | 0 | 0 | 200 |
| blaze | free_one_time | yes | 7 | 0 | 20 | 0 | 0 | 13 | 0 | 200 |
| cliproxyapi | internal | yes | 13 | 4 | 17 | 0 | 0 | 4 | 0 | 200 |
| iamhc | free_one_time | yes | 25 | 0 | 16 | 0 | 22 | 13 | 0 | 200 |
| blaze-free | free_one_time | yes | 17 | 1 | 15 | 0 | 17 | 15 | 0 | 200 |
| bluesminds | free_one_time | yes | 159 | 19 | 13 | 0 | 155 | 9 | 0 | 200 |
| g4f_ollama | fully_free | yes | 39 | 10 | 12 | 0 | 31 | 4 | 0 | 200 |
| g4f_groq | fully_free | yes | 5 | 5 | 11 | 0 | 5 | 11 | 0 | 200 |
| freemodel-anthropic | free_daily | yes | 0 | 4 | 10 | 0 | 0 | 10 | 4 | 200 |
| agnes_ai | free_tier_limited | yes | 0 | 2 | 10 | 0 | 0 | 10 | 2 | 200 |
| dext | fully_free | yes | 0 | 0 | 9 | 0 | 0 | 9 | 0 | 200 |
| g4f_pollinations | fully_free | yes | 18 | 10 | 4 | 0 | 18 | 4 | 0 | 200 |
| freemodel | free_daily | yes | 0 | 4 | 3 | 0 | 0 | 3 | 4 | 200 |
| groq | fully_free | yes | 22 | 16 | 0 | 0 | 0 | 0 | 0 | 403 http 403 |
| gemini | fully_free | yes | 53 | 7 | 0 | 0 | 0 | 0 | 0 | 401 http 401 |
| cerebras | fully_free | yes | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 403 http 403 |
| together | fully_free | yes | 245 | 4 | 0 | 0 | 0 | 0 | 0 | 403 http 403 |
| openai | paid | yes | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| sambanova | fully_free | no | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| github-models | fully_free | no | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| cloudflare | fully_free | no | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| opencode_zen | fully_free | yes | 46 | 0 | 0 | 0 | 0 | 0 | 0 | 403 http 403 |
| iflow | free_tier_limited | yes | 11 | 11 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| aihubmix | fully_free | yes | 226 | 3 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| noobrouter | free_daily | yes | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| llmgateway | internal | no | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| zhipuai-coding-plan | internal | no | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| zai-coding-plan | internal | no | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| llama | internal | no | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| modelscope | internal | no | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| siliconflow-cn | internal | no | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| tencent-coding-plan | internal | no | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| iflowcn | internal | no | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| alibaba-coding-plan | internal | no | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| alibaba-coding-plan-cn | internal | no | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| cortecs | internal | no | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| poe | internal | no | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| zenmux | internal | no | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| alibaba-cn | internal | no | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| minimax-cn-coding-plan | internal | no | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| minimax-coding-plan | internal | no | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| nova | internal | no | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| xiaomi-token-plan-ams | internal | no | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| xiaomi-token-plan-cn | internal | no | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| xiaomi-token-plan-sgp | internal | no | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| zai | internal | no | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| zhipuai | internal | no | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| fireworks-ai | internal | no | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| huggingface | internal | no | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| jiekou | internal | no | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| kimi-for-coding | internal | no | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| meganova | internal | no | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| nano-gpt | internal | no | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| novita-ai | internal | no | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| siliconflow | internal | no | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| brave_search | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| tavily | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| duckduckgo | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| exa | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| jina | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| router/best-coding | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| router/best-reasoning | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| router/best-research | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| router/best-chat | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| router/best-coding-moe | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| coding-smart | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| coding-fast | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| chat-smart | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| chat-fast | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| antigravity | fully_free | no | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 403 http 403 |
| xinjianya | free_daily | no | 212 | 10 | 0 | 0 | 0 | 0 | 0 | 403 http 403 |
| vps-gateway | internal | yes | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| hapuppy | free_daily | no | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 403 http 403 |
| supacoder | free_one_time | yes | 81 | 80 | 0 | 0 | 0 | 0 | 0 | 403 http 403 |
| blazeai | free_daily | yes | 124 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| wiwi | internal | no | 24 | 22 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| cursor-proxy | internal | yes | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| kilo | fully_free | no | 26 | 16 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| kilocloud | paid | yes | 96 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| zenllm | paid | yes | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| claude-carter | paid | yes | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| swiftrouter | free_one_time | yes | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 TimeoutError |
| logfare | fully_free | yes | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| custom | internal | yes | 207 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| aitools | paid | yes | 42 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| ktai | paid | yes | 47 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| ktai-paid | paid | yes | 86 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| ollama-cloud | free_daily | yes | 61 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| huashang | free_daily | yes | 104 | 0 | 0 | 0 | 0 | 0 | 0 | 0 URLError |
| paxsenix | free_one_time | yes | 150 | 0 | 0 | 0 | 0 | 0 | 0 | 403 http 403 |
| freetheai | free_daily | yes | 138 | 12 | 0 | 0 | 0 | 0 | 0 | 530 http 530 |
| chat-api4-087654-xyz | free_daily | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| api-kr777-top | free_daily | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| agentrouter | free_one_time | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| unity2 | free_one_time | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| sparkcode | free_one_time | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| vectorengine | free_one_time | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| ai-17nas | free_one_time | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| megallm | free_one_time | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| claw-oooai | free_daily | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| chunxueapi | free_daily | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| moleapi | free_one_time | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| zanity | free_daily | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| electronhub | fully_free | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| naxios | fully_free | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| crowllm | free_one_time | yes | 0 | 66 | 0 | 0 | 0 | 0 | 66 | 403 http 403 |
| futureppo | free_daily | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 401 http 401 |
| firepass | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| kiro | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| orcarouter | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| tencent-token-plan | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| tencent-tokenhub | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| umans-ai-coding-plan | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| wafer.ai | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| agentic-gateway | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| anthropic | free_daily | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| cohere | fully_free | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| deepseek | fully_free | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| google | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| kuae-cloud-coding-plan | paid | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| llm7 | fully_free | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| lotte-library | fully_free | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| mydamoxing | paid | yes | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 401 http 401 |
| opencode | fully_free | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| aerolink | free_one_time | yes | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 403 http 403 |
| aibz | free_one_time | yes | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| anmix | free_one_time | yes | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| blaze-org | free_one_time | yes | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 401 http 401 |
| buddybackend | free_one_time | yes | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| choosec | free_daily | yes | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| dcapi-gbox | free_one_time | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| evolvex | free_one_time | yes | 160 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| free-gemini | fully_free | yes | 44 | 0 | 0 | 0 | 0 | 0 | 0 | 404 http 404 |
| freellmpool | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| gbox | free_one_time | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 URLError |
| gratisfy | fully_free | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| kilocode | fully_free | yes | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 URLError |
| kiloproxy-lite | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| logfare-alt | fully_free | yes | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| lpgpt | free_one_time | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 URLError |
| navy | free_one_time | yes | 178 | 0 | 0 | 0 | 0 | 0 | 0 | 403 http 403 |
| nianhua | free_one_time | yes | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| ollama-alt | free_one_time | yes | 35 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| opencode-zen | paid | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| railway-newapi | free_one_time | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| setbug | free_one_time | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 URLError |
| tokenlb | free_tier_limited | yes | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 URLError |
| tokenrouter | paid | yes | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 TimeoutError |
| voidai | free_one_time | yes | 92 | 0 | 0 | 0 | 0 | 0 | 0 | 403 http 403 |
| xinjianya-alt | free_one_time | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| yzgpt-alt | free_one_time | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 404 http 404 |
| yzgpt-main | free_one_time | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 404 http 404 |
| zenllm-local | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| zenllm-responses | paid | yes | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| zhangyuapi | free_one_time | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 401 http 401 |
| freetokenfaucet | fully_free | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 403 http 403 |
| sh00t | fully_free | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 401 http 401 |
| seekai | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| airforce | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| 17nas | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| jasper | internal | yes | 0 | 2 | 0 | 0 | 0 | 0 | 2 | 0 no remote base_url |
| bosco | internal | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| p0 | free_tier_limited | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 403 http 403 |
| poolside | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| alibaba-token-plan | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| llmtr | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| alibaba-token-plan-cn | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| zeldoc | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| subconscious | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| kenari | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| unorouter | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| empiriolabs | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| inferx | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| redwakeai | free_tier_limited | no | 0 | 7 | 0 | 0 | 0 | 0 | 7 | 404 http 404 |
| gcli | free_tier_limited | yes | 0 | 5 | 0 | 0 | 0 | 0 | 5 | 401 http 401 |
| yungnet | free_tier_limited | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| euromodels | free_tier_limited | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no api key |
| aiand | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| hetzner | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| infomaniak | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| regolo-ai | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| merge-gateway | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| edenai | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| larprouter | free_tier_limited | yes | 0 | 7 | 0 | 0 | 0 | 0 | 7 | 403 http 403 |
| scnet-token-plan | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| septorlabs | free_tier_limited | yes | 0 | 11 | 0 | 0 | 0 | 0 | 11 | 403 http 403 |
| requesty | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| opencode-go | internal | no | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 no remote base_url |
| dlab-community | free_tier_limited | yes | 0 | 8 | 0 | 0 | 0 | 0 | 8 | 403 http 403 |
| kiosapi | free_tier_limited | yes | 0 | 26 | 0 | 0 | 0 | 0 | 26 | 403 http 403 |
| privaite | internal | yes | 0 | 7 | 0 | 0 | 0 | 0 | 7 | 0 no remote base_url |
| zenllm-free | free_tier_limited | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 401 http 401 |
| zyf | free_tier_limited | yes | 0 | 6 | 0 | 0 | 0 | 0 | 6 | 403 http 403 |

Totals: providers=203, db_models=4592, config_models=488, live_models=2383, answered_models=2, providers_live_ok=23, providers_with_any_gap=33
