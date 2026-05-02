# Search-Augmented Generation (RAG) Agent
=====================================

## Role/Mission
------------

This autonomous agent's mission is to augment Large Language Model (LLM) responses with real-time data from various search providers, ensuring accurate and up-to-date information. The agent's primary goal is to act as a deterministic filter, deciding when to initiate a search and selecting the most suitable search backend to avoid paid overages.

## Technical Stack
-----------------

The agent will utilize the following technologies:

- **Programming Language**: Python 3.9 (latest compatible version)
- **Search Providers**: Brave Search API, Tavily API, and optional Gemini Grounding (fallback mechanism)
- **Deterministic Logic**: Implementing custom logic modules for search initiation and result validation
- **Backend**: Cloud-based services, such as GitHub Actions, to manage execution and store data
- **Data Storage**: Utilizing GitHub storage for logs and QUESTIONS.md file

## Requirements
--------------

1. **SearchProvider API Integration**:
	* Integrate with Brave Search API and Tavily API (with quota-aware fallback)
	* Implement Gemini Grounding integration (if possible)
2. **Deterministic Logic**:
	* Develop custom logic modules for search initiation (e.g., shopping, latest news, explicit sources/citations)
	* Enforce up to N additional searches based on insufficient or conflicting results
3. **Free Resource Utilization**:
	* Adhere to free resource policies (e.g., Brave Search API's free tier)
	* Implement mechanisms to avoid paid overages (e.g., quota-aware fallback)
4. **Autonomous Decision-Making**:
	* Develop algorithms for independent decision-making, prioritizing search initiation and backend selection
5. **Logging and Data Storage**:
	* Log agent decisions and search results in a GitHub-compatible format
	* Store questions and answers in the QUESTIONS.md file

## File Structure
----------------

The agent's file structure will be organized as follows:

```python
rag-agent/
config.py  # Configurations and constants
 logic_modules/  # Custom logic modules for search initiation and result validation
 search_providers/  # Implemented search provider backend classes
 agents.py  # Main agent class
 tests/  # Unit tests and test scripts
 QUESTIONS.md  # Question log file
 README.md  # This AGENTS.md file
```

## Testing Requirements
-------------------------

Unit tests and test scripts will be written to ensure the following scenarios:

- **SearchProvider API Integration**:
	+ Test API connectivity and response validation
	+ Verify quota-aware fallback mechanisms
- **Deterministic Logic**:
	+ Test custom logic modules for search initiation
	+ Validate up to N additional searches
- **Autonomous Decision-Making**:
	+ Test independent decision-making algorithms
	+ Verify search backend selection

## Git Protocol
--------------

The agent's Git protocol will follow standard practices:

- **Repository**: Host on a personal GitHub repository
- **Branching**: Use feature branches for new developments and pull requests for reviews
- **Commit Messages**: Follow Conventional Commits guidelines

## Completion Criteria
----------------------

The agent will be considered complete when it meets all specified requirements, including:

- **SearchProvider API Integration**: Successfully integrate with Brave Search API, Tavily API, and Gemini Grounding (if implemented)
- **Deterministic Logic**: Develop and validate custom logic modules for search initiation and result validation
- **Autonomous Decision-Making**: Implement and test independent decision-making algorithms
- **Logging and Data Storage**: Log agent decisions and search results in a GitHub-compatible format
- **Testing**: Pass all unit tests and test scripts

By following this detailed AGENTS.md guide, the Search-Augmented Generation (RAG) Agent will effectively augment LLM responses with real-time data from various search providers, ensuring accurate and up-to-date information.