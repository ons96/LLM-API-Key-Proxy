# AGENTS.md

## Role/Mission
The autonomous coding agent is designed to create an LLM (Large Language Model) gateway that provides a unified interface for handling multiple providers. The agent's primary mission is to expose 'virtual models' with built-in fallback chains across providers, ensuring seamless and reliable interactions with the providers.

### Goals:

- Provide a robust LLM gateway with automatic fallback chains.
- Handle all errors internally to prevent conflicts and preserve routing logic.
- Optimize for free resources to ensure self-sustaining operation.

## Technical Stack
The agent will be built using the following technologies:

- **Reverse Proxy**: NGINX or Traefik to handle incoming requests and provider connections.
- **Middleware Chaining**: Python's `fastapi` framework with a middleware pipeline to handle input validation, authentication, and request routing.
- **Provider Redundancy**: Use multiple LLM provider APIs (e.g. Hugging Face, Meta AI) with automatic failover and retry mechanisms.
- **Error Masking**: Utilize `try`-`except` blocks with custom error handling to prevent outer agents from initiating their own fallbacks.
- **Model Aliasing**: Implement model aliasing using a configuration file or database to map virtual models to provider-specific models.

## Requirements
1. **Provider Support**: The agent must support at least three different LLM providers (e.g. Hugging Face, Meta AI, Microsoft Azure).
2. **Fallback Chains**: The agent must implement automatic fallback chains for both API requests and model queries.
3. **Error Handling**: The agent must handle all errors internally, including API request failures and model query exceptions.
4. **Resource Optimization**: The agent must optimize its operation for free resources, using techniques like ephemeral containers and scheduled tasks.
5. **Logging and Monitoring**: The agent must include logging and monitoring mechanisms to track its operation and detect potential issues.
6. **Security**: The agent must implement basic security measures, including input validation and authentication.

## File Structure
The agent's codebase will follow the standard GitHub Actions folder structure:

```bash
agents
  |- LLMGateway.py
  |- config
  |  |- providers.yaml
  |  |- models.yaml
  |- logs
        |- agent.log
  |- requirements.txt
  |- tests
        |- test LLMGateway.py
  |- README.md
  |- AGENTS.md
  |- QUESTIONS.md
  |- .github/workflows
        |- main.yml
```

## Testing Requirements
The agent must include comprehensive tests for its functionality, covering both happy paths and error scenarios. The testing framework will be based on `pytest` or `unittest`.

### Test Cases:

- Test virtual model creation and mapping to provider-specific models.
- Test fallback chain functionality for both API requests and model queries.
- Test error handling and logging mechanisms.
- Test security measures, including input validation and authentication.

## Git Protocol
The agent will operate within a GitHub repository, using the following Git protocol:

- **Branching**: Master branch for stable code, feature branches for development work.
- **Pull Requests**: Use PRs for code reviews and merging.
- **Commits**: Use meaningful commit messages and follow the standard commit format (e.g. `fix: error handling`).

## Completion Criteria
The agent will be considered complete when it meets all the technical requirements and produces accurate results for most providers. To verify completion, run automated tests and manually validate the agent's output against expected results.

When questions or uncertainties arise during development, note them in the `QUESTIONS.md` file and address them accordingly.