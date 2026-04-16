AGENTS.md
=========

Role/Mission
------------

The primary role of this autonomous agent is to develop and maintain a self-hosted, highly customizable LLM model router with seamless multi-API support and intelligent fallback logic. The agent will design a robust architecture to support a wide range of LLM services, prioritize performance, and ensure efficient use of available resources. Specifically, the agent's mission is to:

* Develop a scalable architecture for the LLM model router using free resources wherever possible
* Design a system for effortless integration with various LLM APIs, leveraging both free and paid models
* Implement intelligent load balancing and rate limiting strategies to optimize system performance
* Implement context retention and seamless model switching between different APIs
* Ensure the system can run on mobile devices and be integrated with AI coding agents

Technical Stack
--------------

The agent will employ the following technologies to achieve the mission:

* `litellm` for low-level interactions with the LLM models
* `LangChain` for creating a unified interface to various LLM models
* `Pydantic` for robust, API-driven configuration management
* `FastAPI` for building the API endpoints and handling requests
* `One API` for designing a unified API layer for multiple models
* `Rate Limiting` for efficient usage of LLM APIs
* `Load Balancing` for optimized system performance
* `Portkey` for seamless model switching and context retention

Requirements
------------

The agent must meet the following requirements to successfully complete its mission:

1. Ensure all services and APIs are accessible at no cost to minimize operational expenses
2. Utilize GitHub Actions for continuous integration and deployment
3. Develop a data-efficient system and prioritize memory management to optimize performance
4. Implement effective rate limiting and load balancing strategies to prevent overloading the system
5. Implement a fallback model system that falls back to alternative models if primary model performance is below a certain threshold
6. Ensure the system is designed to integrate seamlessly with both mobile and web-based AI coding agents

File Structure
--------------

The file structure for this project should follow standard web development practices to maintain maintainability and readability. Specifically:

```
 agents/
 |_____ README.md
 |_____ AGENTS.md
 |_____ QUESTIONS.md
 |_____ requirements.txt
 |_____ main.py
 |_____ router/
 |_____ |_____ api_router.py
 |_____ |_____ config.py
 |_____ |_____ fallback.py
 |_____ |_____ models/
 |_____ |_____ |_____ litellm_api.py
 |_____ |_____ |_____ langchain_api.py
 |_____ tests/
 |_____ |_____ test_api_router.py
 |_____ |_____ test_config.py
```

Testing Requirements
--------------------

Prior to completing its mission, the agent must meet the following testing requirements:

1. Write comprehensive tests for the API router and configuration files using `Pytest` and `mypy`
2. Develop unit tests to ensure the system is functioning correctly, with a minimum coverage rate of 80%
3. Ensure all code is written with an emphasis on maintainability, readability, and performance
4. Write smoke tests for the fallback model system and load balancing mechanisms

Git Protocol
------------

The agent will adhere to standard Git best practices, following a branch-based workflow to minimize conflicts and ensure efficient collaboration. The project will be deployed through GitHub Actions, using automated testing and deployment scripts to minimize manual intervention.

Completion Criteria
------------------

The agent completes its mission when it satisfies the following criteria:

1. All required features have been implemented and tested
2. Code quality meets the highest standards of maintainability, readability, and performance
3. System performance is optimized to scale with the number of users and requests
4. Fallback model system and seamless context retention are successfully implemented
5. Rate limiting and load balancing strategies are executed efficiently throughout the system
6. The system passes all tests and is validated as production-ready.