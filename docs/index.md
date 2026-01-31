# Mindprint Model

Production-grade RLHF fine-tuning system for creating personalized language models.

## Features

- **State-of-the-art training**: ORPO (Odds Ratio Preference Optimization) single-stage training
- **Multiple backends**: MLX (Apple Silicon) and PyTorch (CUDA)
- **Production ready**: CI/CD, comprehensive testing, experiment tracking
- **Clean architecture**: Modular, testable, maintainable codebase

## Quick Start

```bash
# Install
pip install -e .

# Train a model
python scripts/run_orpo_training.py \
    --config configs/training_pipeline.yaml \
    --backend mlx
```

## Documentation

- [Getting Started](getting-started/installation.md)
- [User Guide](user-guide/training.md)
- [API Reference](api-reference/core/losses.md)
- [Contributing](contributing/development-setup.md)

## Architecture

The codebase follows clean architecture principles:

- **Core**: Domain logic (losses, schedulers, metrics)
- **Adapters**: External integrations (MLX, PyTorch, LLM APIs)
- **Pipelines**: High-level orchestration
- **Infrastructure**: Cross-cutting concerns (logging, reproducibility)

## License

MIT
