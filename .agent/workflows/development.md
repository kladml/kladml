---
description: Development workflow and coding standards for KladML
---

# KladML Development Workflow

## Core Principles (NON-NEGOTIABLE)

### 0. Architecture First
Before implementing any new component, consult `.agent/workflows/architecture.md` for design patterns and constraints.

### 1. Test-Driven Development (TDD)
**ALWAYS follow the Red-Green-Refactor cycle:**

1. **RED**: Write a failing test FIRST that describes the expected behavior
2. **GREEN**: Write the minimum code to make the test pass
3. **REFACTOR**: Clean up the code while keeping tests green

**NO CODE WITHOUT TESTS.** If you write code without a corresponding test, you have failed.

### 2. Coverage Policy ("Ratchet Policy")
- **New code**: Must have 100% test coverage
- **Modified code**: Any file you touch must be brought to 100% coverage
- **Never decrease coverage**: Each commit should maintain or improve overall coverage

### 3. Before Implementing Any Feature
1. Read this workflow file
2. Check current test coverage: `pytest tests --cov=src/kladml --cov-report=term-missing`
3. Identify which tests need to be written
4. Write tests FIRST
5. Then implement the feature

## Testing Commands

```bash
# Run all tests with coverage
pytest tests --cov=src/kladml --cov-report=term-missing

# Run specific test file
pytest tests/test_<module>.py -v

# Run tests matching a pattern
pytest tests -k "test_export" -v

# Generate HTML coverage report
pytest tests --cov=src/kladml --cov-report=html
```

## CLI Testing Pattern

Use Typer's CliRunner for testing CLI commands:

```python
from typer.testing import CliRunner
from kladml.cli.main import app

runner = CliRunner()

def test_command_success():
    result = runner.invoke(app, ["command", "--option", "value"])
    assert result.exit_code == 0
    assert "expected output" in result.stdout

def test_command_error():
    result = runner.invoke(app, ["command", "--invalid"])
    assert result.exit_code != 0
```

## Git Commit Standards

- Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `chore:`
- Commits are made as: **KladML** <dev@kladml.com>
- Push frequently, commit granularly

## File Structure for Tests

```
tests/
├── __init__.py
├── conftest.py          # Shared fixtures
├── test_<module>.py     # Unit tests for src/kladml/<module>.py
├── cli/
│   └── test_<command>.py  # CLI command tests
└── functional/
    └── test_<scenario>.py # End-to-end tests
```

## Before Pushing

1. Run full test suite: `pytest tests`
2. Check coverage hasn't decreased
3. Run linter: `ruff check .`
4. Commit with descriptive message

## Current Coverage Baseline (2026-01-23)

- Total: 36%
- Critical gaps: CLI (0%), TUI (0%), Backends (<50%)
- Strong: Training (>80%), Validator (90%)

**Goal: 100% coverage before v1.0.0**
