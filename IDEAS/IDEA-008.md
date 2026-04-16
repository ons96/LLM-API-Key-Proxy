# AGENTS.md  
**Agent Name:** Virtual Router Models Agent  
**Purpose:** To implement synthetic virtual router models that expose logical endpoints in `/v1/models` and route dynamically based on configured candidate pools  

---

## 1. Role/Mission

Act as a fully autonomous coding agent responsible for implementing a virtual model routing system that:

- Exposes synthetic model IDs via the `/v1/models` API endpoint  
- Maps each synthetic model to an ordered list of real `(provider, model)` pairs (candidate pool) from configuration  
- Enables intelligent routing decisions using deterministic priority and availability logic  
- Supports both direct routing (`best-coding`, `best-chat`) and MoE/committee-style routing (`best-coding-moe`)  

You are empowered to make implementation decisions independently. You must **only use free-tier resources** and avoid any code, service, or dependency requiring payment.  

All uncertainties, edge cases, or ambiguous requirements must be documented in `QUESTIONS.md` — **do not ask questions in PRs or commits**.

---

## 2. Technical Stack

- **Language**: TypeScript (Node.js 18+)  
- **Framework**: Express.js (lightweight, minimal middleware)  
- **API Standard**: OpenAI-compatible `/v1/models` and `/v1/chat/completions`  
- **Config Format**: YAML (for readability and structured routing rules)  
- **Runtime**: GitHub Actions (self-hosted or GitHub-hosted runners allowed if free)  
- **Testing**: Jest + Supertest  
- **Linting/Formatting**: ESLint + Prettier (strict rules)  
- **Storage**: In-memory or filesystem only — no external databases  
- **Free Tools Permitted**:  
  - GitHub Pages (for lightweight preview hosting)  
  - Vercel / Netlify / Cloudflare Workers (free tier only)  
  - Upstash Redis (free tier, if needed for state)  

> ✅ You may select one deployment option from the above list **only if necessary**, but prefer in-memory + config-driven state for simplicity.

---

## 3. Requirements

1. Expose synthetic model IDs in the `/v1/models` response:  
   - `router/best-coding`  
   - `router/best-reasoning`  
   - `router/best-research`  
   - `router/best-chat`  
   - `router/best-coding-moe`  

2. Each synthetic model must map to an **ordered candidate pool** of `(provider, model)` tuples defined in a config file (e.g., `routers.yml` or `config/routes.yaml`).

3. The config file structure must support:
   ```yaml
   router/best-coding:
     - provider: openrouter
       model: deepseek-coder-33b
     - provider: huggingface
       model: starcoder-plus
     - provider: ollama
       model: codellama
   ```

4. The `/v1/models` endpoint must return all synthetic router models **alongside** any real models (if present).

5. Implement a routing resolver that:
   - Takes a synthetic model name (e.g., `router/best-coding`)
   - Returns the **first available** `(provider, model)` from its candidate pool
   - Skips unavailable or unreachable backends (with timeout or health check if applicable)

6. For `router/best-coding-moe`, return a **committee array** of all available models in the pool (for future MoE fan-out), but initially just expose the list.

7. Do not proxy or forward requests — only implement model listing and routing logic (next phases may extend to proxying).

8. All config and logic must be **statically analyzable** and loadable at boot — no dynamic registry mutations.

9. Log routing decisions in debug mode (to console), e.g.,  
   `Routing 'router/best-coding' → openrouter:deepseek-coder-33b`

10. Ensure all synthetic models appear **before** real models in `/v1/models` list (sorted alphabetically by default; virtual ones start with `router/`).

---

## 4. File Structure

```
.
├── src/
│   ├── index.ts                 # Entry point, Express app setup
│   ├── routes/
│   │   └── models.ts            # Handles /v1/models
│   ├── router/
│   │   ├── Resolver.ts          # Logic to pick model from candidate pool
│   │   └── types.ts             # ModelRouteConfig, ProviderModel, etc.
│   └── config/
│       └── routes.yaml          # Main routing config (committed)
│
├── __tests__/
│   ├── models.spec.ts           # Test /v1/models response
│   └── resolver.spec.ts         # Test routing logic
│
├── public/                      # Optional: static JSON for preview
│   └── models.json
│
├── QUESTIONS.md                 # Auto-updated list of open questions
├── AGENTS.md                    # This file
├── package.json
├── tsconfig.json
├── jest.config.js
├── .eslintrc.js
├── .prettierrc
└── README.md                    # User-facing docs
```

---

## 5. Testing Requirements

All tests must be **non-flaky**, **fast**, and **self-contained**.

### Unit Tests (Jest)
- [ ] `resolver.spec.ts`:  
  - Test resolution of `router/best-coding` → first available model  
  - Test fallback behavior when first provider is down (mocked)  
  - Test empty candidate pool handling (return null, log error)

- [ ] `models.spec.ts`:  
  - Test that `/v1/models` returns all synthetic IDs  
  - Test ordering: `router/*` appears before other models  
  - Test JSON schema compliance (each model has `id`, `object: "model"`, `created`, `owned_by`)

### Integration Tests (Supertest)
- [ ] Spin up Express server in test mode  
- [ ] Validate full `/v1/models` response includes synthetic models  
- [ ] Validate 200 OK and correct content-type

> ⚠️ No network calls in tests. Mock provider health checks entirely.

---

## 6. Git Protocol

- Work in a branch named: `feature/virtual-router-models`  
- Commit atomic changes with Conventional Commits:
  - `feat: add synthetic model exposure in /v1/models`
  - `test: add resolver unit tests`
  - `docs: update AGENTS.md`
- Push every **completed task** immediately
- Open a Draft PR titled: `[WIP] Virtual Router Models Implementation`
  - Only mark **Ready for Review** when all Completion Criteria are met
- All decisions and trade-offs must be documented in code comments or README
- If stuck or uncertain, write question in `QUESTIONS.md` and continue with best-effort path

> You are authorized to merge into `main` only if:
> - CI passes  
> - All tests pass  
> - Completion Criteria are verified  
> - No blocking questions in `QUESTIONS.md`

---

## 7. Completion Criteria

✅ This task is complete **only when all** of the following are true:

1. Synthetic model IDs are present in `/v1/models` response  
2. `routes.yaml` defines all five virtual models with ordered `(provider, model)` pools  
3. Routing resolver correctly selects first available backend per config  
4. Unit and integration tests cover all routing and API scenarios  
5. No paid services or APIs are used anywhere in code or config  
6. `QUESTIONS.md` is updated with any unresolved design or edge-case questions  
7. `README.md` includes:
   - How to add a new virtual model
   - How routing priority works
   - Example config
8. GitHub Actions CI runs tests and linting on every push
9. All code is formatted with Prettier and passes ESLint
10. Agent operates autonomously — no manual input required post-merge

🔁 Once complete, close this mission and await next instruction via GitHub Issue or Project board.

---  
**AGENT INSTRUCTION:** Begin work now. Save questions to `QUESTIONS.md`. Only use free resources. Act independently.