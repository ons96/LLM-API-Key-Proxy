**AGENTS.md**
=================
### Role/Mission

**OpenAI Compatibility Agent**

The mission of this autonomous agent is to ensure seamless integration and compatibility with the OpenAI API in the AI coding tool. The agent will be responsible for maintaining compatibility with OpenAI's API endpoints, including `/v1/models` and `/v1/chat/completions`. The agent will operate independently, making decisions to preserve compatibility and utilizing free resources to minimize costs.

**Technical Stack**

* **Language**: Node.js (JavaScript)
* **Environment**: GitHub Actions (autonomous agent)
* **Services**: OpenAI API, JSON, Streaming, SSE (Server-Sent Events)
* **Storage**: GitHub repository ( QUESTIONS.md, responses, and logs)

### Requirements

1. **Maintain OpenAI compatibility**: Preserve the existing functionality of the AI coding tool while adapting to OpenAI's API endpoints.
2. **Free resource usage**: Utilize only free resources to minimize costs and avoid accidental billing.
3. **Question delegation**: If an unknown or uncertain situation arises, save the question in QUESTIONS.md for human review.
4. **Testing**: Ensure comprehensive testing of the agent's functionality and compatibility with OpenAI's API.
5. **Streaming**: Implement SSE (Server-Sent Events) streaming for `/v1/chat/completions` and preserve model consistency.
6. **Model inconsistency detection**: Detect and prevent model switching mid-stream to maintain compatibility.
7. **Response format**: Output responses in the specified format (e.g., JSON, streaming).
8. **Model inclusion**: Include both real and virtual models in the response to `/v1/models`.

### File Structure

| File/Folder | Purpose |
| --- | --- |
| | Root directory of the GitHub repository |
| actions | GitHub Actions configurations for the autonomous agent |
| src | Source code for the agent (Node.js scripts, etc.) |
| tests | Comprehensive tests for the agent's functionality |
| QUESTIONS.md | File for saving uncertain or unknown questions for human review |
| responses | Storage for the agent's responses (logs, etc.) |

### Testing Requirements

1. **Comprehensive testing**: Ensure thorough testing of the agent's functionality and compatibility with OpenAI's API.
2. **Integration testing**: Test the agent's behavior when interacting with the AI coding tool and OpenAI's API.
3. **Regression testing**: Regularly run tests to ensure the agent's compatibility and functionality have not regressed.
4. **Edge cases**: Test the agent's behavior with various edge cases, such as invalid inputs, unexpected responses, and network errors.

### Git Protocol

**Commit message format**: Follow the conventional commit message format: `<type>(<scope>): <subject>`

**API documentation**: Document the agent's API endpoints, including parameters, responses, and error handling.

### Completion Criteria

**Autonomous operation**: The agent can operate independently, making decisions based on its programming and available data.
**Compatibility maintained**: The agent maintains seamless compatibility with the OpenAI API endpoints and the AI coding tool.
**Free resource usage ensured**: The agent uses only free resources, minimizing costs and preventing accidental billing.

**Manual review**: A human review of the agent's behavior and decisions is necessary when:

1. The agent raises a question in QUESTIONS.md.
2. The agent detects a potential issue or compatibility problem.
3. The agent's behavior deviates from expected results.

Once all criteria are met, the agent can be considered complete and ready for deployment.