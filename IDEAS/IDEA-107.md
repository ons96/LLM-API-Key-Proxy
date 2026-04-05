# AGENTS.md - Context Management System with Pruning

---

## 1. Role/Mission

### Purpose

Design and implement a **Context Management System with Pruning** — an intelligent, autonomous system that manages conversation context across sessions, enabling efficient handling of long-running conversations while staying within token budget constraints.

### Mission Statement

Build a modular, extensible context management framework that:
- Maintains conversation history across sessions with persistent storage
- Prunes context intelligently using summarization and token budgeting
- Enables seamless context transfer between different conversation sessions
- Operates efficiently within LLM context window limitations
- Self-manages with minimal human intervention

### Core Philosophy

The system should treat context as a finite resource that requires careful allocation, pruning, and optimization—similar to memory management in operating systems.

---

## 2. Technical Stack

### Language & Runtime
- **Language:** Python 3.10+
- **Runtime:** Standard CPython (no specialized runtime required)

### Core Dependencies (Free/Open Source)

| Package | Purpose | License |
|---------|---------|---------|
| `pydantic` | Data validation schemas | MIT |
| `redis` | Session persistence (optional, with in-memory fallback) | MIT |
| `tiktoken` | Token counting | MIT |
| `sqlalchemy` | ORM for session persistence | Apache 2.0 |
| `pytest` | Testing framework | MIT |
| `pytest-asyncio` | Async test support | MIT |

### Storage Options (Choose One)
- **Primary:** In-memory dict (for zero-setup usage)
- **Optional:** SQLite via SQLAlchemy (for persistence)
- **Optional:** Redis (for distributed setups)

### LLM Integration (Abstracted)
- Design as an interface/protocol
- Support pluggable summarization backends
- Allow injection of any LLM client

### No External Services Required
All infrastructure must be self-hosted or free-tier compatible. No paid APIs are required for core functionality.

---

## 3. Requirements (Numbered)

### 3.1 Core Context Management

1. **Context Storage**
   - Store conversation messages with metadata (role, content, timestamp, token count)
   - Include support for system prompts and custom metadata fields
   - Implement unique session identifiers (UUID v4)

2. **Token Budget Management**
   - Implement configurable token budgets per session (default: 80% of context window)
   - Track token usage in real-time
   - Enforce hard limits that trigger pruning before overflow

3. **Context Window Optimization**
   - Calculate exact token counts using `tiktoken` with configurable encoding (default: `cl100k_base`)
   - Support both counting full messages and selective retention
   - Provide warnings when approaching budget limits (at 75%, 90% thresholds)

### 3.2 Context Pruning & Summarization

4. **Summarization-Based Pruning**
   - Summarize oldest messages when token budget is exceeded
   - Use configurable summarization prompt templates
   - Preserve key facts, decisions, and user preferences in summaries
   - Mark summarized content distinctly in metadata

5. **Multiple Pruning Strategies**
   - **Strategy A (Summarize):** Compress old messages via LLM summarization
   - **Strategy B (Truncate):** Keep only last N messages
   - **Strategy C (Priority):** Retain messages by role (always keep system prompt, user favorites)
   - Allow custom strategy injection via interface

6. **Pruning Triggers**
   - Automatic trigger when token budget exceeded
   - Manual trigger via API call
   - Configurable trigger thresholds (e.g., prune at 85% capacity)

### 3.3 Session Persistence

7. **Session Lifecycle**
   - Create new sessions with optional initial system prompt
   - Resume sessions from persistent storage
   - Archive inactive sessions after configurable timeout (default: 24 hours)
   - Delete sessions explicitly or via cleanup

8. **Persistence Backends**
   - In-memory (default, zero-setup)
   - SQLite file-based (no external service)
   - Provide abstract backend interface for extensibility

9. **Session Metadata**
   - Track session creation time, last activity, message count
   - Store pruning history (what was pruned, when)
   - Include optional user-defined tags for filtering

### 3.4 Context Transfer

10. **Export Context**
    - Export session context to JSON format
    - Support selective export (last N messages, summary only, full history)
    - Include all metadata in export

11. **Import Context**
    - Import context from JSON into new or existing session
    - Handle conflicts (merge vs. replace options)
    - Validate imported data schema

12. **Cross-Session Transfer Protocol**
    - Transfer specific messages between sessions
    - Copy selected context to start new session
    - Support partial transfer (e.g., system prompt + last 5 messages)

### 3.5 API & Interface Design

13. **Programmatic API**
    - Provide Python class-based API for integration
    - Methods: `create_session()`, `add_message()`, `get_context()`, `prune()`, `export()`, `import()`
    - Support both sync and async interfaces

14. **CLI Interface**
    - Command-line tool for manual operations
    - Commands: `new`, `list`, `show`, `prune`, `export`, `import`, `stats`
    - Human-readable output with color support

15. **Configuration**
    - YAML-based configuration file
    - Support environment variable overrides
    - Default configuration bundled with sensible defaults

### 3.6 Observability

16. **Logging**
    - Structured logging with configurable levels
    - Log all pruning events with before/after token counts
    - Log session state changes (create, resume, archive, delete)

17. **Statistics**
    - Track total sessions, active sessions, messages processed
    - Track pruning frequency and effectiveness
    - Expose stats via API method

---

## 4. File Structure

```
context_manager/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── core/
│   ├── __init__.py
│   ├── session.py
│   ├── message.py
│   ├── context.py
│   └── token_tracker.py
├── pruning/
│   ├── __init__.py
│   ├── base.py
│   ├── summarizer.py
│   ├── truncator.py
│   └── priority.py
├── persistence/
│   ├── __init__.py
│   ├── base.py
│   ├── memory.py
│   └── sqlite.py
├── api/
│   ├── __init__.py
│   ├── client.py
│   └── models.py
├── cli/
│   ├── __init__.py
│   └── main.py
└── utils/
    ├── __init__.py
    └── helpers.py

tests/
├── __init__.py
├── test_core/
│   ├── __init__.py
│   ├── test_session.py
│   ├── test_message.py
│   └── test_token_tracker.py
├── test_pruning/
│   ├── __init__.py
│   └── test_strategies.py
├── test_persistence/
│   ├── __init__.py
│   └── test_backends.py
├── test_api/
│   ├── __init__.py
│   └── test_client.py
└── test_cli/
    ├── __init__.py
    └── test_commands.py

docs/
├── architecture.md
├── api_reference.md
└── examples.md

config.yaml
requirements.txt
pyproject.toml
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `core/session.py` | Session class managing conversation context |
| `core/message.py` | Message data model with token counting |
| `core/token_tracker.py` | Token budget tracking and limits |
| `pruning/base.py` | Abstract base for pruning strategies |
| `pruning/summarizer.py` | Summarization-based pruning implementation |
| `persistence/memory.py` | In-memory session storage |
| `api/client.py` | Programmatic API for integration |
| `cli/main.py` | CLI entry point |

---

## 5. Testing Requirements

### 5.1 Test Coverage Goals

- **Minimum:** 80% line coverage
- **Core modules:** 95% coverage mandatory
- **API layer:** Full integration testing required

### 5.2 Test Categories

1. **Unit Tests**
   - Test individual components in isolation
   - Mock external dependencies (LLM clients)
   - Use `pytest` with parametrized fixtures

2. **Integration Tests**
   - Test session lifecycle (create → add → prune → export)
   - Test persistence backends
   - Test transfer protocol

3. **Property-Based Tests**
   - Verify token counts don't exceed budget
   - Verify pruning preserves message order
   - Verify state consistency after operations

### 5.3 Test Fixtures

- Mock LLM client that returns deterministic summaries
- In-memory persistence backend for fast tests
- Sample conversation data with known token counts

### 5.4 Test Execution

```bash
# Run all tests with coverage
pytest --cov=context_manager --cov-report=html

# Run fast tests only (skip integration)
pytest -m "not slow"

# Run with verbose output
pytest -v --tb=long
```

### 5.5 CI Integration

- Run tests on every push
- Fail build if coverage drops below threshold
- Generate coverage report as artifact

---

## 6. Git Protocol

### 6.1 Branching Strategy

```
main
├── develop
│   ├── feature/*       (new features)
│   ├── fix/*           (bug fixes)
│   └── refactor/*      (refactoring)
```

- `main` — Production-ready code only
- `develop` — Integration branch
- Feature branches from `develop`, merge back to `develop`

### 6.2 Commit Messages

Follow conventional commits:

```
