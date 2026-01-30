# Testing Guide

## Test Structure

```
tests/
├── unit/          # Fast, isolated tests
├── integration/   # Component interaction tests
├── property/      # Property-based tests (Hypothesis)
├── benchmarks/    # Performance benchmarks
└── fixtures/      # Shared test data
```

## Running Tests

### All Tests

```bash
pytest
```

### With Coverage

```bash
pytest --cov=src --cov-report=term-missing
```

### Specific Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Benchmarks
pytest tests/benchmarks --benchmark-only
```

## Writing Tests

### Unit Test Example

```python
def test_simpo_loss():
    """Test SimPO loss computation."""
    loss_fn = SimPOLoss(SimPOConfig(beta=2.0))
    
    result = loss_fn.compute(
        policy_chosen_logps=...,
        policy_rejected_logps=...,
        chosen_lengths=...,
        rejected_lengths=...,
    )
    
    assert result.loss is not None
    assert "simpo_loss" in result.metrics
```

### Property-Based Test Example

```python
from hypothesis import given, strategies as st

@given(
    chosen_logps=st.lists(st.floats(-10, 10), min_size=1),
    rejected_logps=st.lists(st.floats(-10, 10), min_size=1),
)
def test_loss_always_positive(chosen_logps, rejected_logps):
    """Loss should always be non-negative."""
    loss_fn = SimPOLoss(SimPOConfig())
    result = loss_fn.compute(...)
    assert result.loss >= 0
```

## Test Coverage

Target: **90%+ coverage**

Check coverage:
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## Benchmarks

Benchmarks track performance over time:

```bash
pytest tests/benchmarks --benchmark-json=benchmark.json
```

Compare benchmarks:
```bash
pytest tests/benchmarks --benchmark-compare
```
