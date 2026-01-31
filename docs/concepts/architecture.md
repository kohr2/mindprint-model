# Architecture

## Overview

Mindprint Model follows **clean architecture** principles with clear separation of concerns.

## Directory Structure

```
src/
├── core/              # Domain logic (no external dependencies)
│   ├── losses/       # Loss functions (DPO, SimPO, ORPO)
│   ├── schedulers/   # LR schedulers
│   ├── data/         # Data types and validation
│   └── config/       # Configuration schemas
│
├── adapters/          # External integrations
│   ├── mlx/          # MLX backend
│   ├── pytorch/      # PyTorch backend
│   ├── llm/          # LLM API clients
│   └── tracking/     # Experiment tracking
│
├── pipelines/         # High-level orchestration
│   ├── training.py   # Training pipeline
│   └── evaluation.py # Evaluation pipeline
│
└── infrastructure/    # Cross-cutting concerns
    ├── logging.py    # Structured logging
    └── reproducibility.py  # Seed management
```

## Design Principles

### 1. Dependency Rule

**Core** has no external dependencies. It can be tested without ML frameworks.

**Adapters** depend on Core, not vice versa.

**Pipelines** orchestrate Core and Adapters.

### 2. Interface Segregation

Each module exposes minimal, focused interfaces:

- `BaseLoss` for loss functions
- `ModelInterface` for models
- `TrainerInterface` for trainers

### 3. Dependency Injection

Dependencies are injected, not imported:

```python
# Good: Inject loss function
trainer = Trainer(model, loss_fn=SimPOLoss(config))

# Bad: Hard-coded dependency
trainer = Trainer(model)  # Uses DPO internally
```

## Data Flow

```
CLI → Pipeline → Adapter → Core → Adapter → Pipeline → Output
```

Example:
```
run_orpo_training.py
  → TrainingPipeline
    → MLXTrainer (adapter)
      → SimPOLoss (core)
        → MLXModel (adapter)
          → TrainingPipeline
            → WandBTracker (adapter)
```

## Testing Strategy

- **Unit tests**: Test Core modules in isolation
- **Integration tests**: Test Adapter + Core interactions
- **E2E tests**: Test full Pipeline

## Benefits

1. **Testability**: Core logic tested without ML frameworks
2. **Flexibility**: Swap backends/losses without changing pipelines
3. **Maintainability**: Clear boundaries, easy to understand
4. **Extensibility**: Add new losses/backends without touching existing code
