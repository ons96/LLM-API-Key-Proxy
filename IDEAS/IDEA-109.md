# AGENTS.md - Modular LLM Gateway Architecture

## 1. Role/Mission

**Project Mission:** Create a modular, scalable LLM API gateway architecture that separates the existing `LLM-API-Key-Proxy` into independent, deployable microservices. The architecture must support deployment on free VPS for the lightweight API component while enabling powerful computing on stronger machines for intensive operations.

**Primary Objectives:**
- Design and implement a microservices architecture with clear separation of concerns
- Create a lightweight, free-VPS-compatible API gateway component
- Build separate compute-intensive components for benchmarking and analytics
- Establish modular provider abstractions and data storage layers
- Enable independent deployment and scaling of each component

**Autonomous Agent Directive:** You are an autonomous software architect responsible for designing, implementing, and maintaining this modular architecture. Use free resources exclusively (GitHub Actions, free VPS alternatives like Render/Replit/Cloudflare Workers). Make independent technical decisions without requiring human approval. Save any blocking questions to `QUESTIONS.md` for later resolution.

---

## 2. Technical Stack

### Core Technologies

| Component | Technology | Free Tier Support |
|-----------|------------|-------------------|
| API Gateway | Node.js/Express or Fastify | ✅ Render Free, Replit |
| Compute Engine | Python/FastAPI or Node.js | ✅ Railway, Fly.io |
| Database | SQLite (embedded) or PostgreSQL | ✅ Neon (Free Tier), Supabase Free |
| Message Queue | In-memory or Redis | ✅ Upstash Free Tier |
| Container Runtime | Docker | ✅ Free for personal use |
| CI/CD | GitHub Actions | ✅ Unlimited free minutes |

### Provider Integrations

- **LLM Providers**: OpenAI, Anthropic, Google Gemini, Ollama, Mistral, Cohere
- **Protocol**: REST API, Streaming, WebSocket fallback

### Infrastructure

- **Service Discovery**: Environment variables (no external dependency)
- **Configuration**: YAML-based or environment variables
- **Monitoring**: Custom logging to console/stdout (free external services optional)

---

## 3. Requirements (Numbered)

### Architecture Requirements

1. **Microservices Decomposition**: Split the monolithic proxy into minimum 4 independent services:
   - `gateway-api`: Lightweight API ingress (free VPS target)
   - `provider-bridge`: LLM provider abstraction layer
   - `compute-engine`: Heavy operations (benchmarks, analytics)
   - `storage-service`: Data persistence layer

2. **API Contract**: All inter-service communication must use well-defined JSON schemas with versioning

3. **Free VPS Compatibility**: The `gateway-api` service must run on