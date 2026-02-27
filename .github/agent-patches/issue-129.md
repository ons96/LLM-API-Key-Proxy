## Root Cause

The repository lacks **pre-commit hooks** configuration, which means:
- Code style violations (trailing whitespace, missing EOF newlines) can be committed unnoticed
- Large files (>1MB) might accidentally enter the git history
- Merge conflict markers or broken YAML/JSON syntax can reach the main branch
- Developers must manually remember to run linters/formatters, leading to inconsistent code quality

## Solution

Add a `.pre-commit-config.yaml` file with essential hooks for general file hygiene, plus a CI workflow to enforce these checks on pull requests.

### 1. Create `.pre-commit-config.yaml`

```yaml
# .pre-commit-config.yaml
# See https://pre-commit.com for more information
# See https://pre-commit.com/hooks.html for more hooks

default_install_hook_types: [pre-commit, commit-msg]
default_stages: [pre-commit]

repos:
  # General file hygiene (language-agnostic)
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
        args: [--markdown-linebreak-ext=md]
        exclude: \.(svg|patch)$
      - id: end-of-file-fixer
        exclude: \.(svg|patch)$
      - id: check-yaml
        args: [--allow-multiple-documents]
      - id: check-json
      - id: check-toml
      - id: check-xml
      - id: check-added-large-files
        args: ['--maxkb=500']  # Prevent accidental commit of large binaries
      - id: check-merge-conflict
      - id: check-case-conflict
      - id: detect-private-key  # Prevent committing AWS keys, etc.
      - id: mixed-line-ending
        args: ['--fix=lf']  # Force LF line endings
      - id: check-executables-have-shebangs
      - id: check-shebang-scripts-are-executable
      - id: check-symlinks
      - id: destroyed-symlinks
      - id: forbid-new-submodules
      - id: fix-byte-order-marker

  # Commit message formatting (optional but recommended)
  - repo: https://github.com/commitizen-tools/commitizen
    rev: v3.13.0
    hooks:
      - id: commitizen
        stages: [commit-msg]

  # Python specific (uncomment if this is a Python project)
  # - repo: https://github.com/astral-sh/ruff-pre-commit
  #   rev: v0.1.9
  #   hooks:
  #     - id: ruff
  #       args: [--fix, --exit-non-zero-on-fix]
  #     - id: ruff-format
  #
  # - repo: https://github.com/pre-commit/mirrors-mypy
  #   rev: v1.7.1
  #   hooks:
  #     - id: mypy
  #       additional_dependencies: [types-all]

  # JavaScript/TypeScript (uncomment if applicable)
  # - repo: https://github.com/pre-commit/mirrors-eslint
  #   rev: v8.56.0
  #   hooks:
  #     - id: eslint
```

### 2. Create GitHub Actions workflow (CI enforcement)

Create `.github/workflows/pre-commit.yml`:

```yaml
name: Pre-commit Checks

on:
  pull_request:
  push:
    branches: [main, master, develop]

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          
      - name: Install pre-commit
        run: pip install pre-commit
        
      - name: Cache pre-commit hooks
        uses: actions/cache@v3
        with:
          path: ~/.cache/pre-commit
          key: pre-commit-${{ hashFiles('.pre-commit-config.yaml') }}
          
      - name: Run pre-commit
        run: pre-commit run --all-files --show-diff-on-failure
```

### 3. Update documentation

Add to `CONTRIBUTING.md` (or create a section in `README.md`):

```markdown
## Development Setup

### Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to ensure code quality. Install it:

```bash
# Install pre-commit
pip install pre-commit
# or
brew install pre-commit

# Install hooks in the repository
pre-commit install
pre-commit install --hook-type commit-msg
```

Hooks will now run automatically on `git commit`. To run manually:

```bash
# Check all files
pre-commit run --all-files

# Check specific file
pre-commit run --files src/main.py

# Skip hooks temporarily (not recommended)
git commit -m "message" --no-verify
```
```

## Explanation of Changes

**1. File Hygiene Hooks (`pre-commit-hooks`):**
- `trailing-whitespace` & `end-of-file-fixer`: Ensure consistent formatting and smaller diffs
- `check-yaml`/`json`/`toml`: Prevents syntax errors in config files
- `check-added-large-files`: Blocks files >500KB from being committed (prevents git bloat)
- `detect-private-key`: Security measure to prevent credential leakage
- `check-merge-conflict`: Rejects commits containing unresolved conflict markers (`<<<<<<< HEAD`)
- `mixed-line-ending`: Enforces Unix-style line endings (LF) for cross-platform consistency

**2. CI Integration:**
- The GitHub Actions workflow ensures hooks run on all PRs, catching issues even if developers haven't installed pre-commit locally
- Caching is configured to speed up subsequent runs

**3. Commit Message Hook (Optional):**
- `commitizen` enforces [Conventional Commits](https://www.conventionalcommits.org/) format (e.g., `feat:`, `fix:`, `docs:`), enabling automatic changelog generation

**4. Extensibility:**
- Python and JavaScript hooks are commented out but ready to activate based on the repository's tech stack
- The configuration uses `rev` tags (specific versions) rather than `main` to ensure reproducible builds

**To activate:** Developers simply run `pre-commit install` once after cloning. The hooks are non-intrusive and auto-fix most issues (like trailing whitespace) in-place before the commit completes.
