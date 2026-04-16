# AGENTS.md

## 1. Role/Mission

The Web Search Augmentation agent is responsible for enhancing the OpenAI LLM gateway with web search capabilities. This will involve integrating AI-powered web search APIs and crawling tools to provide context-based search results. The agent will make decisions independently to optimize functionality while adhering to free resource usage.

**Primary Function:** Develop and integrate web search functionality into the OpenAI LLM gateway for enhanced user experience.

**Secondary Function:** Continuously monitor and improve the integrated web search capabilities.

## 2. Technical Stack

* Programming Language: Python
* Framework: FastAPI
* ASGI Server: Uvicorn
* Cloud Platform: Render (for free resources)
* BYOK (Bring Your Own Key) for secure authentication
* Kilo Code for efficient and maintainable code
* Gemini CLI for easy deployment and management

## 3. Requirements

1. The agent must utilize free resources to build and operate the web search augmentation service.
2. The integrated web search API must be authenticated using BYOK for secure access.
3. The OpenAI LLM gateway interface must be enhanced to support web search queries.
4. The web search integration must display relevant, high-quality search results to users.
5. The agent must use available caching mechanisms to improve performance and reduce latency.
6. The agent must save any questions or concerns in the QUESTIONS.md file.

## 4. File Structure

The repository will follow a standard structure for Python projects with the main files and directories organized as follows:

* `agents/` - containing the autonomous agent code
* `requirements.txt` - listing dependencies for the agent
* `README.md` - main documentation for the project
* `AGENTS.md` - specific instructions and guidelines for the agent
* `QUESTIONS.md` - for collecting and addressing any questions or concerns
* `tests/` - containing test cases for the integrated web search capabilities

## 5. Testing Requirements

1. The agent must include comprehensive test cases for the integrated web search API.
2. Test cases must cover different scenarios, including edge cases and error handling.
3. Tests must be written using Python's built-in `unittest` module or equivalent.

## 6. Git Protocol

The agent will follow the following Git guidelines:

* Use git branches to segregate development and testing environments.
* Regularly push changes to the main repository.
* Use meaningful commit messages for each update.

## 7. Completion Criteria

The agent will be considered complete when it has successfully integrated the web search API into the OpenAI LLM gateway and:

1. Displayed relevant, high-quality search results to users.
2. Utilized BYOK authentication for secure access.
3. Provided improved performance through caching mechanisms.
4. Saved any questions or concerns in the QUESTIONS.md file.

Once the completion criteria have been met, the agent will automatically notify relevant stakeholders through GitHub comments or direct messages.