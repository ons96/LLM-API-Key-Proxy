# AGENTS.md
## Role/Mission

As an autonomous coding agent, the mission is to investigate and resolve the issue of conversation context preservation when switching between different LLM (Large Language Model) API providers. The agent will analyze the scenario, develop strategies to maintain coherent context across heterogeneous backends, and potentially implement these strategies through prompt engineering or state tracking.

Given the autonomous nature of the agent, it will operate independently, make decisions based on the information available, and utilize only free resources on GitHub Actions. In the course of its work, the agent may encounter questions or ambiguities, which will be saved to the QUESTIONS.md file for further investigation by human maintainers.

## Technical Stack

* Programming language: Python 3.9+
* Package manager: pip
* LLM API providers: Investigate and utilize multiple providers (e.g., Hugging Face Transformers, Google Cloud AI Platform, Microsoft Azure Cognitive Services)
* Session storage: Utilize a free, in-memory session storage solution (e.g., Redis, Memcached)
* Context tracking: Leverage techniques such as prompt templating, context window analysis, LLM memory techniques, and conversation history serialization
* Testing framework: Utilize Python's built-in unittest module

## Requirements

1. **Switching between LLM providers**: The agent must be able to seamlessly switch between different LLM API providers without loss of conversation context.
2. **Context preservation**: The agent must preserve conversation context when switching between different LLM providers, ensuring that the context is available for subsequent interactions.
3. **Prompt engineering**: The agent must employ strategies such as prompt templating to create context-aware prompts that can be interpreted by the chosen LLM provider.
4. **Session management**: The agent must manage sessions effectively, using a free, in-memory session storage solution to store and retrieve context information.
5. **Testing and validation**: The agent must include comprehensive testing and validation to ensure that conversation context is preserved across LLM providers.
6. **Resource efficiency**: The agent must utilize only free resources on GitHub Actions, optimizing its performance to minimize costs.

## File Structure

The project will follow the standard structure for GitHub Actions, with the following main directories:

* `src/`: Source code for the agent
* `tests/`: Unit tests and integration tests for the agent
* `docs/`: Documentation for the agent
* `utils/`: Utility functions for the agent
* `requirements.txt`: Dependencies required by the agent
* `QUESTIONS.md`: File for logging questions or ambiguities encountered by the agent

## Testing Requirements

The agent must include comprehensive testing and validation to ensure that conversation context is preserved across LLM providers. The testing framework should include:

* Unit tests for individual components of the agent
* Integration tests to ensure seamless switching between LLM providers
* System tests to validate the overall functionality of the agent

## Git Protocol

The agent will operate on GitHub Actions and utilize the standard Git protocol for committing changes. It will follow standard Git best practices, including:

* Commit frequency: Commit changes regularly to ensure that the codebase is up-to-date.
* Commit messages: Include descriptive commit messages that summarize the changes made.
* Branching strategy: Utilize a feature branch strategy, with separate branches for individual features or fixes.

## Completion Criteria

The agent will be considered complete when it meets the requirements specified above, including:

* Successful switching between LLM providers without loss of conversation context
* Preserved conversation context across LLM providers
* Effective prompt engineering strategies
* Efficient resource usage on GitHub Actions
* Comprehensive testing and validation

Once the agent meets these criteria, it can be considered for deployment and further development.