**AGENTS.md**
================

**Team**: Autonomous Software Development

**Team Agent**: Router with Fallbacks and Cooldowns

**Role/Mission**
---------------

Our goal is to create a centralized router that efficiently identifies the best candidate models for resolving requests based on their requirements, filters them based on capability, FREE_ONLY_MODE, and cooldown status, and attempts them in order until success or exhaustion.

**Technical Stack**
------------------

* Python 3.8: Core programming language used for implementing the agent.
* GitHub Actions: Continuous Integration and Continuous Deployment pipeline.
* Free External Tools (e.g., e2e API, Real/Virtual Model APIs): Utilize free APIs and open-source projects for external dependencies.
* Kubernetes (minikube): Lightweight local deployment for development and testing.

**Requirements**
---------------

1. **Modular Design**: The router must be designed to allow for easy addition of new capabilities, fallbacks, and cooldowns.
2. **Candidate Filtering**: Filter candidates based on:
	* Capability requirements (needs_tools, needs_vision, needs_structured_output, min_context, etc.).
	* FREE_ONLY_MODE: exclude commercial resources from consideration.
	* Cooldown status: ensure candidates are not excessively used during a given time frame.
3. **Probing and Fallbacks**: Attempt candidates in order until success or exhaustion:
	* Probing: incrementally collect latency and model stats.
	* Fallbacks: select from a list of fallback models (e.g., primary, secondary, tertiary).
4. **Consecutive Failure Handling**: Handle consecutive failures by:
	* Notifying the system administrator through the QUESTIONS.md log file.
	* Providing useful error messages and suggestions for further action.
5. **Exponential Weighted Moving Average (EWMA) Latency**: Track and display real-time latency stats.
6. **Model Candidate Pooling**: Create and maintain candidate models through:
	* Real/Virtual model discovery.
	* Model filtering.
7. **Request and Capability Abstraction**: Implement request and capability abstraction layers for extensibility.

**File Structure**
----------------

* `src/`
	+ `models/`
		- Real model code (e.g., real_model.py).
		- Virtual model code (e.g., virtual_model.py).
	+ `candidate_pooling.py`: Create and manage model candidates.
	+ `cooldown_tracker.py`: Manage cooldown status and stats.
	+ `consecutive_failure_handler.py`: Handle consecutive failures and notify the administrator.
	+ `ewma_latency_tracker.py`: Track real-time latency stats.
	+ `router.py`: Central router with probing and fallback logic.
	+ `requirement_parser.py`: Parse request requirements and capability needs.
	+ `request_handler.py`: Handle incoming requests and trigger probing.
* `tests/`
	+ `test_candidate_pooling.py`.
	+ `test_cooldown_tracker.py`.
	+ `test_consecutive_failure_handler.py`.
	+ `test_router.py`.
* `README.md`: Information about the agent, project setup, and team.
* `QUESTIONS.md`: Log file for questions and concerns.
* `AGENT_CONFIG.json`: Configuration file for the agent.

**Testing Requirements**
----------------------

1. **Unit Tests**: Write comprehensive unit tests for candidate pooling, cooldown tracking, consecutive failure handling, EWMA latency tracking, and router functionality.
2. **Integration Tests**: Implement integration tests to cover the entire system, including external API interactions and router performance.
3. **Performance Tests**: Analyze performance by running load tests and stress tests on the agent.

**Git Protocol**
--------------

* We will maintain a clean and organized commit history by adhering to the following rules:
	+ Create a new branch for each new feature or bug fix.
	+ Use meaningful commit messages.
	+ Rebase commits to avoid merge conflicts.
* Open a new pull request for each change and adhere to automated code review and testing.

**Completion Criteria**
----------------------

1. **Successful Candidate Ranking**: Implement the router with probing and fallback logic to rank candidate models.
2. **Cooldown and Failure Handling**: Implement consecutive failure handling and cooldown status tracking.
3. **Candidate Pooling**: Create and manage model candidates through real/virtual model discovery and pooling.
4. **Request and Capability Abstraction**: Implement request and capability abstraction layers for extensibility.
5. **Automated Tests**: Write comprehensive automated tests covering unit, integration, and performance testing.

As a fully autonomous agent running on GitHub Actions, this project should utilize free resources, make decisions independently, and save any questions or concerns in the QUESTIONS.md log file.