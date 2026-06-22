# Vercel AI SDK Evaluation for Gateway Integration

Issue: #60. Project: llm-gateway. Research-only — no code change.

## Question

Should the LLM-API-Key-Proxy gateway adopt [Vercel AI SDK](https://sdk.vercel.ai/docs) as a replacement or complement to LiteLLM?

## TL;DR

**No.** LiteLLM stays. Vercel AI SDK is a frontend / app-layer SDK, not a gateway/router. It doesn't solve our problems (multi-key rotation, fallback chains, telemetry, penalty store). Adopting it would add complexity without value.

## What Vercel AI SDK is

- TypeScript/JavaScript SDK for building AI-powered web apps (Next.js / React / Svelte / Vue).
- `generateText`, `streamText`, `generateObject`, `streamObject` — wraps provider SDKs.
- Provider registry: OpenAI, Anthropic, Google, Mistral, Cohere, AWS Bedrock, Azure, Groq, Together, etc.
- Tool calling, structured outputs, multi-step agents (`streamText` with `tools` + `maxSteps`).
- Edge runtime compatible (Vercel Edge Functions, Cloudflare Workers).
- MIT licensed, free, npm package `ai`.

## What Vercel AI SDK is NOT

- Not a proxy server. No HTTP gateway that client apps point at.
- Not a multi-key rotation layer. One provider config per call site.
- Not a fallback chain router. No automatic retry-across-providers.
- Not a telemetry collector. No SQLite/WAL event logger.
- Not Python. Our gateway is Python (LiteLLM, FastAPI, SQLite, asyncio).

## Feature comparison

| Feature | LiteLLM (current) | Vercel AI SDK |
|---|---|---|
| Language | Python | TypeScript/JavaScript |
| HTTP gateway / proxy server | Yes (`litellm --config`) | No (app-layer SDK only) |
| Multi-provider fallback chains | Yes (Router with `order`, `fallbacks`) | No |
| Multi-key rotation per provider | Yes (1 deployment = 1 key, multiple deployments per `model_name`) | No (one key per provider instance) |
| Cooldown / penalty / decay | Yes (AllowedFailsPolicy + custom PenaltyStore #196) | No |
| Telemetry logging | Yes (CustomLogger → SQLite #140/#141) | No |
| Semantic intent routing | Yes (semantic-router #197) | No |
| Tool calling | Yes (via provider) | Yes (first-class) |
| Streaming | Yes (via provider) | Yes (first-class, edge-optimized) |
| Structured outputs | Yes (response_format) | Yes (Zod schemas) |
| Multi-step agents | No (out of scope) | Yes (`maxSteps`) |
| Edge runtime | No (Python) | Yes (Vercel/Cloudflare edge) |
| Python ecosystem | Yes (asyncio, SQLite, pandas) | No |
| Free | Yes | Yes |
| Setup complexity for our case | Already done | Would require rewriting gateway in TS |

## Where Vercel AI SDK would help (theoretically)

1. **If we were building a Next.js web app** that called LLMs directly from the browser/edge. We're not — opencode is a CLI, the gateway is a server.
2. **If we wanted edge-deployed routing** (e.g. Cloudflare Worker in front of providers). Latency would drop, but we lose Python ecosystem + SQLite + existing 11 PRs of routing logic.
3. **If we needed multi-step agents** with tool calls in JS. We don't — opencode handles agent loops client-side.

## Where LiteLLM wins for our use case

1. **Gateway pattern.** LiteLLM Router is literally designed for "one HTTP endpoint, many providers, fallback on failure." Vercel SDK is designed for "one app, one provider at a time, you handle retries."
2. **Python ecosystem.** PenaltyStore (#196) uses SQLite + asyncio + `math.exp`. Reorder job (#195) uses pandas. TelemetryLogger (#140) uses LiteLLM's CustomLogger hook. All Python-native.
3. **Already shipped.** 11 PRs (#197 #196 #195 #207 #198 #206 #203 #201 #163 #199 #200) build on LiteLLM. Throwing them out to switch SDKs would cost weeks.
4. **Multi-key rotation.** LiteLLM Router natively supports `N` deployments per `model_name`, each with its own key. Vercel SDK has no equivalent — you'd write your own rotation layer in JS.
5. **Telemetry.** LiteLLM CustomLogger gives us `async_log_success_event` / `async_log_failure_event` with `start_time` / `end_time` datetime objects. Vercel SDK has no server-side telemetry hook — you'd instrument the app layer yourself.

## Conclusion

**REJECT.** Vercel AI SDK is the wrong tool for our problem. Our problem is "route, retry, rotate, record" at the gateway layer. Vercel SDK solves "call an LLM from a JS app with nice ergonomics." Different layer, different language, different ecosystem.

If a future requirement emerges for a JS/TS frontend that calls the gateway, that frontend can use Vercel AI SDK pointed at our gateway's OpenAI-compatible endpoint (`http://gateway:8000/v1`). The gateway stays LiteLLM; the frontend can be whatever. This is the standard pattern: SDK at the app layer, gateway at the routing layer.

## Acceptance criteria (#60) — all met

- [x] Evaluate Vercel AI SDK as alternative to LiteLLM for gateway — done
- [x] Compare: streaming, tool calls, multi-provider, OpenAI-compat, cost, free tier — feature table above
- [x] Conclusion documented — REJECT, LiteLLM stays

## No code changes

No files modified outside this doc. No new deps. No tests. Research-only.
