# Autonomous GitHub Actions

This repository includes fully autonomous GitHub Actions that use free LLMs to process and fix issues.

## Workflows

### 1. AI Batch Issue Fixer (`ai-batch-issue-fixer.yml`)

**Purpose**: Automatically processes open issues and creates PRs with fixes.

**Triggers**:
- Every 6 hours (schedule)
- Manual dispatch

**Features**:
- Processes up to 5 issues per run
- Parallel processing (max 3 concurrent)
- Uses free LLMs via proxy (Groq, Gemini)
- Auto-creates branches and PRs
- Adds `in-progress` label to issues being worked on
- Handles failures gracefully

**Usage**:
```yaml
jobs:
  fix-issues:
    uses: ./.github/workflows/ai-batch-issue-fixer.yml
    secrets: inherit
```

### 2. AI Issue Generator (`ai-issue-generator.yml`)

**Purpose**: Scans codebase for potential issues and creates GitHub issues.

**Triggers**:
- Weekly (Sunday 2 AM)
- Manual dispatch

**Scan Types**:
- `full`: All categories
- `security`: Security vulnerabilities
- `performance`: Performance issues
- `documentation`: Missing docs
- `code-quality`: Code smells

### 3. Multi-Repo Orchestrator (`multi-repo-orchestrator.yml`)

**Purpose**: Coordinates issue processing across multiple repositories.

**Triggers**:
- Every 12 hours
- Manual dispatch

**Features**:
- Discovers all repos in your account
- Processes up to 5 repos in parallel
- Triggers individual repo processors

### 4. Reusable Issue Processor (`reusable-issue-processor.yml`)

**Purpose**: Called by other workflows to process issues.

**Usage**:
```yaml
jobs:
  process:
    uses: OWNER/REPO/.github/workflows/reusable-issue-processor.yml@main
    with:
      proxy_url: ${{ secrets.PROXY_URL }}
      max_issues: 5
      max_parallel: 3
    secrets:
      BOT_APP_ID: ${{ secrets.BOT_APP_ID }}
      BOT_PRIVATE_KEY: ${{ secrets.BOT_PRIVATE_KEY }}
      PROXY_API_KEY: ${{ secrets.PROXY_API_KEY }}
```

## Setup Requirements

### Required Secrets

| Secret | Description |
|--------|-------------|
| `BOT_APP_ID` | GitHub App ID for the bot |
| `BOT_PRIVATE_KEY` | GitHub App private key |
| `PROXY_API_KEY` | API key for the LLM proxy |
| `PROXY_URL` | (Optional) Custom proxy URL |

### GitHub App Permissions

The bot needs these permissions:
- `contents: write` - Create branches, push changes
- `issues: write` - Add labels, comments
- `pull_requests: write` - Create PRs

### LLM Proxy

The workflows use the LLM proxy at `http://40.233.101.233:8000/v1` which provides:
- **Groq**: llama-3.3-70b, llama-3.1-8b-instant
- **Gemini**: gemini-2.5-pro, gemini-2.5-flash
- **G4F**: Various free models

Virtual models:
- `coding-elite` - Best for complex fixes
- `coding-fast` - Quick for simple tasks

## How It Works

### Issue Processing Flow

```
1. Discover eligible issues
   ↓
2. Filter out: wontfix, invalid, duplicate, in-progress
   ↓
3. Process in parallel (max 3)
   ↓
4. For each issue:
   a. Add 'in-progress' label
   b. Create feature branch
   c. Run OpenCode with issue context
   d. OpenCode analyzes and implements fix
   e. Create PR with fix
   f. Comment on issue with PR link
```

### Parallel Execution

GitHub Actions free tier allows:
- Up to 5 concurrent jobs per workflow
- Up to 3 parallel matrix jobs recommended
- 6-hour job timeout (we use 30 min per issue)

### Free LLM Usage

The proxy provides free LLM access:
- No API costs
- Rate-limited but sufficient for CI
- Automatic fallback between providers

## Monitoring

Check workflow runs:
- Go to Actions tab
- Filter by workflow name
- View individual job logs

Issue status indicators:
- `in-progress`: Being processed
- `ai-generated`: PR was auto-created
- `needs-discussion`: AI couldn't fix automatically

## Limitations

1. **Rate limits**: Free LLMs have rate limits
2. **Complex issues**: Some issues need human review
3. **Context size**: Large codebases may exceed context
4. **Determinism**: AI fixes may vary between runs

## Best Practices

1. **Label important issues**: Use `priority-high` for urgent fixes
2. **Exclude with labels**: Add `wontfix`, `invalid`, or `duplicate` to skip
3. **Review AI PRs**: Always review auto-generated PRs before merging
4. **Set limits**: Use `max_issues` to control processing load
