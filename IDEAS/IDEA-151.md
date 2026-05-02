# AGENTS.md

## Role/Mission

This autonomous agent is designed to perform scheduled pipeline refreshes and ensure idempotence and transactional integrity for the data pipeline. Its primary goals are to:

* Set up a daily scheduled refresh using APScheduler
* Implement pipeline idempotence to prevent duplicate data processing
* Ensure transactional integrity to preserve data consistency
* Write data to temporary tables and swap them with the main data tables on success

This agent runs independently on GitHub Actions, utilizing free resources and making decisions without direct human intervention.

## Technical Stack

The primary technologies used by this agent are:

* **Python 3.8+**: The programming language used to implement the agent logic
* **APScheduler**: A Python library for scheduling tasks to run at regular intervals
* **BackgroundScheduler**: A feature of APScheduler for running tasks in the background
* **GitHub Actions**: A continuous integration and continuous deployment (CI/CD) platform for automating workflows
* **Free resources on GitHub Actions**: The agent will only use free resources such as environment, minutes, and storage.

## Requirements

The agent must meet the following requirements:

1. **Schedule daily refresh**: Set up a scheduled task to refresh the data pipeline daily at a specific time (e.g., midnight).
2. **Implement pipeline idempotence**: Ensure that the pipeline processes data only once, even in the event of errors or retries.
3. **Ensure transactional integrity**: Maintain data consistency by using transactions to roll back changes in case of errors.
4. **Write to temporary tables**: Store data in temporary tables and swap them with the main data tables on successful pipeline execution.
5. **Log agent activity**: Record any errors, warnings, or successes during pipeline execution for auditing and debugging purposes.

## File Structure

The agent code will be structured as follows:

* `agents/`: The root directory for the agent code
* `agents/scheduler.py`: The Python module that sets up and runs the APScheduler
* `agents/pipeline.py`: The module that implements the data pipeline logic
* `agents/transactions.py`: The module that handles database transactions
* `QUESTIONS.md`: A separate file that captures any questions or uncertainties encountered during development or operation

## Testing Requirements

The agent's code must be tested thoroughly to ensure it meets the requirements outlined above. Testing will involve:

* Unit testing: Test individual components of the agent's code, including the scheduler and pipeline logic.
* Integration testing: Test how different components of the agent work together to achieve the required functionality.
* End-to-end testing: Test the agent's behavior from start to finish, including scheduling, pipeline execution, and data processing.

## Git Protocol

The agent's code will be stored in a separate GitHub repository from the main project codebase. All commits will be made to this repository, and changes will be tracked using Git version control.

## Completion Criteria

To consider this agent complete, the following criteria must be met:

1. **Daily refresh is scheduled**: The agent successfully sets up and runs a daily scheduled refresh using APScheduler.
2. **Pipeline is idempotent**: The agent ensures that the pipeline processes data only once, without duplicate processing or errors.
3. **Transactions are used**: The agent implements transactions to maintain data consistency and roll back changes in case of errors.
4. **Temporary tables are written to**: The agent stores data in temporary tables and swaps them with the main data tables on successful pipeline execution.
5. **Error handling is implemented**: The agent logs errors, warnings, and successes during pipeline execution for auditing and debugging purposes.

Before deploying the agent, all tests will be run to ensure that it meets the completion criteria.