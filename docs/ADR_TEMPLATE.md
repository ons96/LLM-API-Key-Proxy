# ADR Template

> Architecture Decision Record (ADR) template and guide for LLM API Proxy project

An Architecture Decision Record (ADR) is a document that captures an important architectural decision made along with its context and consequences.

## ADR Format

Use this template when creating new ADRs:

```markdown
# ADR-XXX: [Short title describing the decision]

## Status
[Proposed | Accepted | Deprecated | Superseded]

## Context
[Describe the issue motivating this decision]

## Decision
[Describe the change being proposed/decided]

## Consequences
[Describe the resulting context after applying the decision]

### Positive
- [List positive outcomes]

### Negative
- [List negative outcomes or trade-offs]

### Neutral
- [List neutral items or things that may change in the future]

## Related ADRs
- [Link to related ADRs by number]

## Notes
[Any additional notes, links, or references]
```

## ADR Naming Convention

- File name: `ADR-XXX-short-title.md`
- Example: `ADR-001-use-fastapi-for-gateway.md`
- Number sequentially starting from 001

## When to Create an ADR

Create an ADR when:

1. **Adding a new provider** - Document the reasoning behind the implementation
2. **Changing the routing logic** - Record the decision and its impact
3. **Modifying authentication** - Document security considerations
4. **Changing data storage** - Record the reasoning and trade-offs
5. **Introducing new dependencies** - Document why the dependency was chosen

## ADR List

| ADR | Title | Status |
|-----|-------|--------|
| ADR-001 | Use FastAPI for the gateway | Accepted |
| ADR-002 | Use YAML for configuration | Accepted |
| ADR-003 | Implement virtual models | Accepted |
| ADR-004 | Use credential rotation | Accepted |

## Examples

### Example 1: Provider Implementation

```markdown
# ADR-005: Add Gemini Provider Support

## Status
Accepted

## Context
We need to add Google Gemini as a provider to improve model diversity and add a free-tier option.

## Decision
Implement a new `GeminiProvider` adapter class that:
- Uses Google AI API for completions
- Supports both API key and OAuth authentication
- Falls back to OAuth for Gemini CLI users

## Consequences

### Positive
- Adds high-quality free tier provider
- OAuth support enables CLI users to use their existing credentials
- Consistent with existing provider architecture

### Negative
- Requires OAuth credential management
- More complex than simple API key providers

### Neutral
- May need updates as Gemini API evolves

## Related ADRs
- ADR-001: Use FastAPI for the gateway
- ADR-004: Use credential rotation

## Notes
- See docs/PROVIDER_ARCHITECTURE.md for implementation details
- OAuth implementation similar to Gemini CLI provider
```

### Example 2: Configuration Change

```markdown
# ADR-006: Use YAML for Configuration

## Status
Accepted

## Context
We need a human-readable configuration format that's easy to edit and version control.

## Decision
Use YAML for all configuration files:
- `router_config.yaml` for provider settings
- `virtual_models.yaml` for virtual model definitions

## Consequences

### Positive
- Human-readable and editable
- Supports comments for documentation
- Native Python support via PyYAML
- Easy to version control

### Negative
- YAML parsing can be slow for large files (not a concern here)
- Indentation-sensitive

### Neutral
- JSON also works but is less readable for humans

## Related ADRs
None

## Notes
- Consider JSON for machine-generated configs in the future
```

## Workflow

1. **Create ADR**: Copy the template and fill in the sections
2. **Discuss**: Share with team for feedback
3. **Update status**: Mark as Accepted/Deprecated as decisions are made
4. **Link to PR**: Reference ADR in pull requests

## Tools

To list all ADRs:

```bash
ls -1 docs/adrs/
```

To create a new ADR number:

```bash
# Get next ADR number
ls docs/adrs/ | sort -V | tail -1
```

## Resources

- [Markdown ADR](https://adr.github.io/madr/)
- [Michael Nygard's ADR format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [joelparkerhenderson/architecture_decision_record](https://github.com/joelparkerhenderson/architecture_decision_record)
