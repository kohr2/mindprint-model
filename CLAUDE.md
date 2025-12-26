# Claude Code Instructions

## Project Overview

Mindprint RLHF LoRA fine-tuning project for Bob Loukas Bitcoin cycle trading persona. Python/ML codebase using PyTorch, Transformers, and PEFT libraries.

## Development Philosophy: Test-First Always

**TDD and BDD are mandatory for all code changes.**

### Test-Driven Development (TDD)

Before writing ANY implementation code:

1. **Red**: Write a failing test first that defines the expected behavior
2. **Green**: Write the minimum code to make the test pass
3. **Refactor**: Clean up while keeping tests green

```
# Example workflow
1. User requests: "Add a function to parse preference pairs"
2. Claude FIRST writes: test_parse_preference_pairs.py
3. Claude runs test (expects FAIL)
4. Claude writes: preference_parser.py
5. Claude runs test (expects PASS)
6. Claude refactors if needed
```

### Behavior-Driven Development (BDD)

For features, start with acceptance criteria in Gherkin-style:

```gherkin
Feature: Voice Fidelity Evaluation
  Scenario: Evaluate response against reference
    Given a reference answer from Bob's textbook
    And a model-generated response
    When I calculate voice fidelity score
    Then the score should be between 0.0 and 1.0
    And responses matching Bob's style score above 0.75
```

Use `pytest-bdd` for feature files when appropriate.

## Mandatory Testing Practices

### Before Implementation

- [ ] Write test file BEFORE implementation file
- [ ] Define expected inputs and outputs
- [ ] Cover edge cases in tests
- [ ] Include type hints in test signatures

### Test Structure

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Component interaction tests
├── fixtures/       # Shared test data
└── features/       # BDD feature files
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_data_preparation.py

# Run BDD features
pytest tests/features/
```

### Test Naming Convention

- Test files: `test_<module_name>.py`
- Test functions: `test_<function_name>_<scenario>`
- Example: `test_parse_preference_pairs_handles_empty_input`

## Git Branching Strategy

This project uses a multi-branch workflow to separate shared infrastructure from approach-specific implementations.

### Branch Structure

- **`shared`** - Shared foundation code (data prep, evaluation, model config)
- **`dpo`** - DPO (Direct Preference Optimization) approach implementation
- **`sft`** - SFT (Supervised Fine-Tuning) approach implementation (if applicable)
- Other approach-specific branches as needed

### What Goes Where

**Push to `shared` branch:**
- Data preparation pipeline (`src/data_prep/`)
- Evaluation infrastructure (`src/evaluation/`)
- Model configuration (`src/models/config.py`)
- Shared utilities (`src/utils/`)
- Common tests for shared components
- Documentation for shared architecture

**Push to approach-specific branches (e.g., `dpo`, `sft`):**
- Training loops specific to the approach
- Approach-specific hyperparameters
- Reward modeling (if approach-specific)
- Approach-specific tests

### Workflow

1. **Before starting work**, identify if the change is shared or approach-specific
2. **For shared changes**: Checkout `shared` branch, implement, push to `shared`
3. **For approach changes**: Checkout the approach branch, implement, push there
4. **Merge shared into approach branches** regularly to keep them up to date

```bash
# Working on shared infrastructure
git checkout shared
# ... make changes ...
git add . && git commit -m "Add feature to shared infrastructure"
git push origin shared

# Update approach branch with shared changes
git checkout dpo
git merge shared
```

### Decision Guide

Ask: "Would this code be used by multiple training approaches?"
- **Yes** → Push to `shared`
- **No** → Push to the specific approach branch

## Code Quality Requirements

### Before Committing

1. All tests pass (`pytest`)
2. Type checking passes (`mypy src/`)
3. Linting passes (`ruff check .`)
4. Coverage maintained above 80%

### Type Hints

All functions must have type annotations:

```python
def calculate_voice_fidelity(
    reference: str,
    response: str,
    model: SentenceTransformer
) -> float:
    ...
```

## ML-Specific Testing

### Data Pipeline Tests

- Validate data shapes and types
- Test with synthetic fixtures (not real data)
- Assert on tensor dimensions

### Model Tests

- Use small model variants for tests
- Mock expensive operations
- Test training step outputs, not convergence

### Evaluation Tests

- Test metric calculations with known values
- Validate score ranges
- Test edge cases (empty inputs, max length)

## File Organization

```
mindprint-model/
├── src/
│   ├── data/           # Data loading and processing
│   ├── models/         # Model definitions
│   ├── training/       # Training loops
│   ├── evaluation/     # Metrics and evaluation
│   └── utils/          # Shared utilities
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   └── features/
├── configs/            # Training configurations
└── scripts/            # CLI entry points
```

## Claude Workflow Checklist

When asked to implement a feature:

1. **Clarify** requirements and acceptance criteria
2. **Write tests** that define the expected behavior
3. **Run tests** to confirm they fail (red)
4. **Implement** minimum code to pass tests
5. **Run tests** to confirm they pass (green)
6. **Refactor** while keeping tests green
7. **Document** with docstrings and type hints

When asked to fix a bug:

1. **Write a test** that reproduces the bug
2. **Run test** to confirm it fails
3. **Fix** the bug
4. **Run test** to confirm it passes
5. **Check** no other tests broke

## Dependencies

Core libraries:
- `torch` - Deep learning
- `transformers` - HuggingFace models
- `peft` - LoRA fine-tuning
- `datasets` - Data loading
- `sentence-transformers` - Embeddings

Testing:
- `pytest` - Test framework
- `pytest-cov` - Coverage
- `pytest-bdd` - BDD support
- `pytest-mock` - Mocking

## ML Engineering Best Practices

### Reproducibility

- **Set seeds everywhere**: `torch.manual_seed()`, `random.seed()`, `np.random.seed()`
- **Log all hyperparameters**: Use config files, never hardcode
- **Version control data**: Track data versions with DVC or hashes
- **Pin dependencies**: Use exact versions in `requirements.txt`

```python
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

### Configuration Management

- Use `dataclasses` or Pydantic for configs
- Store configs as YAML/JSON, not Python dicts
- Validate configs at load time

```python
@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5
    batch_size: int = 8
    num_epochs: int = 3

    def __post_init__(self):
        assert self.learning_rate > 0
        assert self.batch_size > 0
```

### Experiment Tracking

- Log metrics to Weights & Biases or MLflow
- Save model checkpoints with meaningful names
- Track: loss curves, eval metrics, hyperparameters, git commit

### Memory Management

- Use `torch.cuda.empty_cache()` between experiments
- Prefer gradient checkpointing for large models
- Use mixed precision (fp16/bf16) when possible
- Profile memory with `torch.cuda.memory_summary()`

### Data Pipeline Best Practices

- Stream large datasets, don't load to memory
- Use `datasets.Dataset.map()` with `batched=True`
- Cache preprocessed data to disk
- Validate data shapes at pipeline boundaries

```python
def validate_batch(batch: dict) -> None:
    assert "input_ids" in batch
    assert batch["input_ids"].dim() == 2
    assert batch["input_ids"].dtype == torch.long
```

### LoRA-Specific Practices

- Start with small rank (r=8), increase if needed
- Target attention layers first (`q_proj`, `v_proj`)
- Use alpha = 2 * rank as starting point
- Merge adapters for inference speed

### Evaluation Best Practices

- Evaluate on held-out set, never training data
- Use multiple metrics (not just loss)
- Compare against baseline (base model without LoRA)
- Track both quantitative metrics AND qualitative samples

### Error Handling

- Fail fast with clear error messages
- Validate inputs at function boundaries
- Use custom exceptions for domain errors

```python
class DataValidationError(Exception):
    """Raised when training data fails validation."""
    pass

def load_preference_pairs(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not data:
        raise DataValidationError(f"Empty preference pairs at {path}")
    return data
```

### Logging

- Use structured logging (`structlog` or `logging` with JSON)
- Log at appropriate levels (DEBUG for shapes, INFO for progress)
- Include context: batch number, epoch, learning rate

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Training started", extra={
    "epochs": config.num_epochs,
    "batch_size": config.batch_size,
    "model": config.model_name
})
```

## Code Review Checklist

Before considering any feature complete:

- [ ] Tests written and passing
- [ ] Type hints on all functions
- [ ] Docstrings on public functions
- [ ] No hardcoded paths or magic numbers
- [ ] Config values externalized
- [ ] Error handling for edge cases
- [ ] Logging at key points
- [ ] Memory-efficient for large data

## Reminders for Claude

- **Never skip tests** - If asked to "just implement it quickly", still write tests
- **Test edge cases** - Empty inputs, max lengths, invalid types
- **Use fixtures** - Share test data via pytest fixtures
- **Mock external calls** - Don't hit APIs or load large models in unit tests
- **Keep tests fast** - Unit tests should complete in milliseconds
- **Document test purpose** - Each test should have a clear docstring
- **Config over code** - Externalize all hyperparameters
- **Validate early** - Check data shapes and types at boundaries
- **Log progress** - User should always know what's happening
- **Fail loudly** - Clear error messages beat silent failures
