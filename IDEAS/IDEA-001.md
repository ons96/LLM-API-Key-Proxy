AGENTS.md
================

### 1. Role/Mission
The autonomous coding agent's role is to modify the existing public OpenAI-compatible LLM gateway repository to implement a $0-only multi-provider router with virtual models, rate-limit-aware fallback, optional MoE (Mixture of Experts), and web-search augmentation. The agent must ensure compatibility with BYOK (Bring Your Own Key) OpenAI-format tools, such as Kilo Code and Gemini CLI, while maintaining the existing functionality of the gateway, including support for `/v1/models`, `/v1/chat/completions`, and SSE streaming (`stream:true`).

### 2. Technical Stack
The agent will utilize the following technical stack to accomplish its mission:
- Python as the primary programming language
- FastAPI for building the API gateway
- Uvicorn as the ASGI server
- Render for deployment
- OpenAI-compatible LLM models
- Gemini CLI and Kilo Code for BYOK compatibility

### 3. Requirements
1. Implement a $0-only multi-provider router to manage requests across different LLM models.
2. Develop virtual models to abstract the underlying LLM models and simplify model management.
3. Integrate rate-limit-aware fallback to handle cases where the primary model is at capacity or unreachable.
4. Optionally implement MoE (Mixture of Experts) for improved model performance and efficiency.
5. Integrate web-search augmentation to enhance the capabilities of the LLM gateway.
6. Ensure the modified gateway remains compatible with BYOK OpenAI-format tools (Kilo Code, Gemini CLI, etc.).
7. Maintain existing endpoint support for `/v1/models`, `/v1/chat/completions`, and SSE streaming (`stream:true`).
8. Optimize the solution for deployment on Render, leveraging free resources where possible.
9. Automate testing and deployment using GitHub Actions, ensuring the agent can operate independently.

### 4. File Structure
The agent will maintain a structured approach to code organization, with the following directories and files:
- `app/`: FastAPI application code
    - `models/`: Virtual model definitions
    - `routers/`: Multi-provider router implementation
    - `utils/`: Utility functions for rate limiting, MoE, and web search augmentation
- `config/`: Configuration files for Render deployment and model settings
- `tests/`: Automated tests for the gateway, including endpoint tests and integration tests
- `QUESTIONS.md`: For documenting any questions or uncertainties encountered during development
- `requirements.txt`: Dependencies required for the project

### 5. Testing Requirements
The agent must implement comprehensive automated testing to ensure the modified gateway functions as expected. This includes:
- Unit tests for individual components (virtual models, rate-limit-aware fallback, MoE, web-search augmentation)
- Integration tests for endpoint functionality (`/v1/models`, `/v1/chat/completions`, SSE streaming)
- End-to-end tests simulating real-world usage scenarios
- Performance tests to evaluate the gateway's efficiency under various loads

### 6. Git Protocol
The agent will follow standard Git guidelines for commit messages, branching, and merging:
- Commit messages will be descriptive, following the GitHub guidelines.
- Feature developments will occur on separate branches, merged into `main` upon completion and successful testing.
- The agent will resolve any merge conflicts independently, using best judgment to preserve functionality.

### 7. Completion Criteria
The mission is considered complete when:
- All requirements (1-9) outlined in section 3 are fulfilled.
- The modified gateway is successfully deployed on Render.
- Automated tests (unit, integration, end-to-end, performance) pass without errors.
- The gateway is compatible with BYOK OpenAI-format tools and supports existing endpoints.
- Any questions or uncertainties are documented in `QUESTIONS.md` for future reference.

Upon completion, the agent will ensure the repository is in a deployable state, with clear instructions for any manual steps required for deployment or configuration.