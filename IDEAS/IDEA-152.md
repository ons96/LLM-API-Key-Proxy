**Agents: Testing and Reliability**
=====================================

**Role/Mission**
---------------

This autonomous agent's primary mission is to ensure the testing and reliability of the codebase. It will add unit tests for HTTP caching, parsing, normalization, and deduplication, and implement structured JSON logging and error boundaries with friendly UI messages. The agent will make decisions independently, leveraging its technical stack to manage resources efficiently and effectively.

**Technical Stack**
-----------------

* Programming Language: Python 3.9+
* Unit Testing Framework: Pytest
* HTTP Caching Library: `docker://http-cache` (for caching HTTP responses in Docker)
* JSON Logging Library: `loguru` (for structured JSON logging)
* Error Boundary Library: `brukiin` (for friendly UI messages and error boundary features)

**Requirements**
--------------

1. The agent must run on GitHub Actions, utilizing free resources.
2. The agent must add unit tests for HTTP caching, including tests for caching different types of HTTP responses (success, error, not found).
3. The agent must add unit tests for parsing, including tests for parsing fixtures from different data sources.
4. The agent must add unit tests for normalization, including tests for mathematical operations and aggregation with missing categories.
5. The agent must add unit tests for deduplication, including tests for removing duplicates based on different criteria.
6. The agent must implement structured JSON logging, capturing log messages with timestamps, levels, and relevant details.
7. The agent must implement error boundaries with friendly UI messages, providing a seamless user experience.

**File Structure**
-----------------

```bash
agents/
│
├── scripts/
│   ├── cache_test.py (unit tests for HTTP caching)
│   ├── parser_test.py (unit tests for parsing)
│   ├── normalizer_test.py (unit tests for normalization)
│   └── deduper_test.py (unit tests for deduplication)
│
├── loggers/
│   ├── json_logger.py (structured JSON logging implementation)
│   └── error_broker.py (error boundary implementation)
│
├── config/
│   └── github_actions.yml (GitHub Actions configuration)
│
├── QUESTIONS.md (for storing questions and feedback)
│
└── README.md (this file)
```

**Testing Requirements**
------------------------

1. The agent must run all unit tests for HTTP caching, parsing, normalization, and deduplication whenever the codebase changes.
2. The agent must validate the correctness of the tests and provide feedback on any failures or errors.
3. The agent must notify the development team of any test failures or errors.

**Git Protocol**
----------------

1. The agent must use Git as the source control system.
2. The agent must track changes to the codebase using the `HEAD` commit hash.
3. The agent must update the `QUESTIONS.md` file with any questions or feedback.

**Completion Criteria**
-----------------------

1. The agent must add all required unit tests for HTTP caching, parsing, normalization, and deduplication.
2. The agent must implement structured JSON logging and error boundaries with friendly UI messages.
3. The agent must successfully run all unit tests and report any failures or errors to the development team.
4. The agent must update the `QUESTIONS.md` file with any questions or feedback.

**Important Notes**

* The agent must only use free resources on GitHub Actions.
* The agent must make decisions independently, leveraging its technical stack to manage resources efficiently and effectively.
* The agent must save any questions or feedback to the `QUESTIONS.md` file.
* The agent must follow the GitHub Actions configuration specified in `config/github_actions.yml`.