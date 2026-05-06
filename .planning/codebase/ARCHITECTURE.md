# LLM-API-Key-Proxy Architecture

This document describes the high-level architecture and data flow of the `LLM-API-Key-Proxy` codebase.

## High-Level Components

*   **FastAPI Application (`proxy_app`)**: The HTTP gateway handling incoming requests, API key authentication, and endpoint routing. Includes terminal UI for local management.
*   **Router Core (`router_core.py`)**: The central routing intelligence. Handles virtual models (e.g., `coding-elite`), constructs fallback chains across multiple free and paid LLM providers, and manages rate-limit/timeout logic.
*   **Rotator Library (`rotator_library`)**: The core library abstracting provider interactions.
    *   **Client (`client.py`)**: The `RotatingClient` orchestrates the execution of LLM requests across a managed pool of API keys and provider connections.
    *   **Providers (`providers/`)**: Adapter layer implementing standard interfaces for diverse providers (e.g., Groq, Gemini CLI, G4F, OpenAI).
*   **Settings & Credentials**: 
    *   `settings_tool.py`: Manages configuration application state.
    *   `credential_tool.py` / `credential_manager.py`: Manages OAuth flows, secrets, and API keys outside of code logic.
*   **Configuration Files (`config/*.yaml`)**: Declarative definitions for the router (`router_config.yaml`), virtual models (`virtual_models.yaml`), model aliases (`aliases.yaml`), and provider details (`providers_database.yaml`).

## Data and Request Flow

1.  **Ingress (`main.py`)**: A client sends an OpenAI-compatible request to an endpoint (e.g., `POST /v1/chat/completions`, `POST /v1/responses`, `POST /v1/embeddings`).
2.  **Authentication & Preparation**: FastAPI validates the incoming `Authorization: Bearer <key>`, extracts the requested model, and prepares the payload.
3.  **Routing Strategy (`router_core.py`)**: 
    *   Checks if the requested model is a **Virtual Model** (defined in `config/virtual_models.yaml`).
    *   If it is, the router resolves it into an ordered **Fallback Chain** of specific provider+model combinations (e.g., Groq Llama -> Gemini Pro -> G4F GPT-4).
4.  **Client Execution (`client.py`)**:
    *   The `RotatingClient` attempts the request against the first provider in the fallback chain.
    *   It uses the specific Provider Adapter (`src/rotator_library/providers/*`) to translate the OpenAI payload into the target provider's native format.
5.  **Fallback & Error Handling**:
    *   If the provider adapter succeeds, the response is translated back into OpenAI format and returned to the router.
    *   If the provider encounters an error (rate limit, timeout, provider error), `error_handler.py` classifies the error. The router/client transparently moves to the next provider in the fallback chain.
6.  **Egress**: The translated OpenAI-compatible JSON or Server-Sent Events (SSE) stream is sent back to the original client.

## Core Systems Interaction

*   **Endpoint Layer**: `src/proxy_app/main.py`
*   **Routing Layer**: `src/proxy_app/router_core.py`
*   **Execution Layer**: `src/rotator_library/client.py`
*   **Provider Adapters**: `src/rotator_library/providers/*`
*   **State & Limits**: Tracks cooldowns (`cooldown_manager.py`) and usage (`usage_manager.py`) to prevent spamming exhausted/banned keys.
