# Development Setup

## Prerequisites

- Python 3.9+
- Git
- (Optional) Pre-commit hooks

## Setup

```bash
# Clone repository
git clone https://github.com/kohr2/mindprint-model
cd mindprint-model

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/unit/core/test_losses.py

# Run benchmarks
pytest tests/benchmarks --benchmark-only
```

## Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type check
mypy src/
```

## Pre-commit Hooks

Pre-commit hooks automatically run:
- Ruff (formatting and linting)
- MyPy (type checking)
- Fast unit tests

```bash
# Run manually
pre-commit run --all-files
```

## Project Structure

- `src/core/` - Domain logic (no external deps)
- `src/adapters/` - External integrations
- `src/pipelines/` - High-level orchestration
- `tests/` - Test suite
- `docs/` - Documentation

## Adding New Features

1. Write tests first (TDD)
2. Implement feature
3. Ensure tests pass
4. Update documentation
5. Submit PR

## Code Style

- Use type hints everywhere
- Follow Google-style docstrings
- Keep functions focused and small
- Write tests for all new code
