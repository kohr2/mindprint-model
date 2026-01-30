# Architecture Overview

See [Concepts: Architecture](concepts/architecture.md) for the complete architecture documentation.

## Quick Links

- [Architecture Guide](concepts/architecture.md) - Clean architecture structure
- [Loss Functions](concepts/loss-functions.md) - DPO, SimPO, ORPO comparison
- [User Guide: Training](user-guide/training.md) - How to train models
- [API Reference](api-reference/core/losses.md) - Core modules API

## Key Components

- **Core** (`src/core/`) - Domain logic (losses, schedulers, data)
- **Adapters** (`src/adapters/`) - External integrations (MLX, PyTorch, LLM APIs)
- **Pipelines** (`src/pipelines/`) - High-level orchestration
- **Infrastructure** (`src/infrastructure/`) - Logging, reproducibility

For detailed architecture documentation, see [concepts/architecture.md](concepts/architecture.md).
