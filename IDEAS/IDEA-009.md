# AGENTS.md - Free LLM API Gateway/Endpoint

## 1. Role/Mission

**Mission:** Build a self-hosted, free LLM API gateway that provides a unified interface for accessing multiple Large Language Model (LLM) providers, supports Bring Your Own Key (BYOK) authentication, and operates as a reverse proxy—all while remaining free to deploy and use.

**Agent Role:** You are an autonomous coding agent responsible for designing, implementing, and delivering a complete, production-ready API gateway solution. You must make independent technical decisions, use only free resources, and ensure the system is extensible for future provider additions.

---

## 2. Technical Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Language** | Go (Golang) | High performance, low memory footprint, built-in HTTP server, excellent for CLI tools and APIs |
| **API Framework** | Chi Router (go-chi/chi) | Lightweight, idiomatic Go HTTP router; no external dependencies beyond stdlib |
| **Reverse Proxy** | Built-in Go `httputil.ReverseProxy` | No additional infrastructure required; full control over request/response transformation |
| **Configuration** | YAML + Viper | Human-readable config files, environment variable substitution, widely used |
| **Key Management** | Custom AES-256-GCM encryption | Self-contained, no external services required; keys stored encrypted at rest |
| **Authentication** | JWT tokens + API key hashing | Standard industry practice; supports external tool integration (BYOK) |
| **Client HTTP** | Native `net/http` with retry logic | No external HTTP client library needed; keeps dependencies minimal |
| **Testing** | Go testing package + httptest | Native Go testing; no additional test frameworks required |
| **Containerization** | Docker (multi-stage build) | Minimal final image size; Alpine-based for security |
| **Logging** | Zerolog (or built-in log) | Structured logging, JSON output for observability |

**Free Resource Constraints:**
- No paid services or subscriptions
- All dependencies must be open-source with permissive licenses (MIT, Apache 2.0, BSD)
- Self-hostable on any Linux environment (VPS, Raspberry Pi, home server)

---

## 3. Requirements

### 3.1 Core Functionality

1. **Unified LLM Provider Interface**
   - Expose a single API endpoint (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`) that proxies to backend LLM providers
   - Translate provider-specific request/response formats to OpenAI-compatible format
   - Initially support: OpenAI, Anthropic Claude, Ollama (local), xAI Grok, Perplexity

2. **Reverse Proxy Architecture**
   - Act as a transparent proxy: client sends request to gateway → gateway transforms → forwards to provider → transforms response → returns to client
   - Preserve all original request metadata (tokens, model parameters, temperature, etc.)
   - Handle streaming (`text/event-stream`) and non-streaming responses

3. **BYOK (Bring Your Own Key) Support**
   - Allow users to configure their own API keys for each provider
   - Keys encrypted at rest using AES-256-GCM with a master key
   - Support per-request key selection via header (`X-Provider-Key`) or configured provider mapping
   - Allow external tools to inject keys without exposing them to the gateway operator

4. **Self-Hosting Capability**
   - Single binary deployment (no complex runtime dependencies)
   - Docker Compose file for one-command startup
   - Resource limits: should run on 512MB RAM, single CPU core
   - Support ARM64 ( Raspberry Pi, Apple Silicon) and AMD64 architectures

5. **Rate Limiting & Quotas**
   - Implement per-API-key rate limiting (configurable limits)
   - Default: 60 requests/minute per key (adjustable)
   - Quota tracking stored in-memory or SQLite (for simplicity)

### 3.2 Security Requirements

6. **TLS/SSL Termination**
   - Support TLS with self-signed certificates for local testing
   - Production: redirect HTTP to HTTPS or terminate at load balancer
   - Certificate auto-renewal preparation (config-only for v1)

7. **Request Validation**
   - Validate all incoming JSON payloads against schema
   - Reject requests exceeding max token limits per provider
   - Sanitize logs to prevent key leakage

8. **Audit Logging**
   - Log all requests with: timestamp, provider, model, status code, latency
   - Exclude sensitive data from logs (API keys, request body contents)
   - Support JSONstructured logging for log aggregation tools

### 3.3 Extensibility

9. **Plugin/Provider Architecture**
   - Provider implementations as Go interfaces in separate packages
   - New providers can be added by implementing the `Provider` interface
   - Configuration-driven provider enablement (no code changes needed to add new providers)

10. **Metrics & Observability**
    - Prometheus-compatible `/metrics` endpoint
    - Track: request latency histogram, total requests, error rates by provider, active connections

### 3.4 Developer Experience

11. **SDK/Client Libraries**
    - Provide a Python client library (optional, for external tool integration)
    - OpenAI-compatible client shim: set `OPENAI_API_BASE` to gateway URL

12. **Documentation**
    - `README.md` with quickstart guide
    - API reference (OpenAPI/Swagger spec at `/docs`)
    - Configuration schema with examples for each provider

---

## 4. File Structure

```
llm-gateway/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
├── cmd/
│   └── gateway/
│       └── main.go             # Application entrypoint
├── internal/
│   ├── config/
│   │   ├── config.go           # Configuration loader
│   │   └── config_test.go
│   ├── server/
│   │   ├── server.go           # HTTP server setup
│   │   ├── middleware.go       # Custom middleware (logging, rate limit)
│   │   └── routes.go           # Route definitions
│   ├── proxy/
│   │   ├── proxy.go            # Reverse proxy logic
│   │   └── transformer.go     # Request/response transformations
│   ├── providers/
│   │   ├── provider.go        # Provider interface definition
│   │   ├── openai.go          # OpenAI provider
│   │   ├── anthropic.go      # Anthropic Claude provider
│   │   ├── ollama.go         # Ollama (local) provider
│   │   └── xai.go            # xAI Grok provider
│   ├── auth/
│   │   ├── keys.go            # Key management & encryption
│   │   └── jwt.go             # JWT validation (optional)
│   ├── storage/
│   │   └── sqlite.go         # SQLite quota tracking
│   └── metrics/
│       └── metrics.go        # Prometheus metrics
├── configs/
│   ├── config.yaml           # Example configuration
│   └── nginx.conf           # Optional nginx reverse proxy config
├── docker/
│   ├── Dockerfile           # Multi-stage Docker build
│   └── docker-compose.yml   # Local development stack
├── pkg/
│   └── client/
│       └── python/          # Python client library (optional)
├── SPEC.md                  # Technical specification document
├── AGENTS.md                # This file
├── README.md                # Project documentation
├── go.mod                  # Go module definition
├── go.sum                  # Go dependencies
└── Makefile               # Build automation
```

---

## 5. Testing Requirements

### 5.1 Unit Tests

- **Coverage Target:** Minimum 70% code coverage for `internal/` packages
- **Test Naming:** Describe the scenario being tested (e.g., `TestProxy_TransformsOpenAIRequest_Success`)
- **Mocking:** Use interfaces for external dependencies (providers, storage) to enable mocking

### 5.2 Integration Tests

- **Test Providers:** Mock HTTP responses for each provider to avoid external API calls during CI
- **Test Auth:** Verify BYOK key injection works correctly via headers
- **Test Proxy:** Ensure request/response transformations preserve all fields

### 5.3 End-to-End Tests

- **Docker Compose Test:** Verify the gateway starts and responds to requests in Docker
- **Request Flow:** Start gateway → Send request to `/v1/chat/completions` → Verify response format matches OpenAI spec

### 5.4 Performance Tests

- **Basic Load:** 100 concurrent requests to verify gateway doesn't bottleneck (use Go's stdlib `net/http/httptest`)
- **Memory Usage:** Verify memory stays under 200MB under moderate load

### 5.5 CI Pipeline

- Run tests on every push and pull request
- Lint with `golangci-lint` (free, open-source)
- Build Docker image on tag releases

---

## 6. Git Protocol

### 6.1 Branch Strategy

- **Main Branch:** `main` - Always deployable, contains stable releases
- **Development Branch:** `develop` - Integration branch for features
- **Feature Branches:** `feature