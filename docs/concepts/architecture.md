# Architecture

## Directory Structure

```
src/
├── training/          # ORPO training pipeline
│   ├── orpo_pipeline.py    # ORPOPipeline - main orchestration
│   ├── merge.py            # LoRA adapter merging
│   ├── mps_utils.py        # Apple Silicon MPS utilities
│   ├── reward_model.py     # Reward model
│   ├── adapter_utils.py    # Adapter path management
│   └── data_quality.py     # Data quality metrics
│
├── backends/          # ML framework backends
│   ├── mlx/          # MLX backend (Apple Silicon)
│   └── pytorch/      # PyTorch backend (CUDA)
│
├── core/              # Domain logic (no framework deps)
│   ├── losses/       # Loss functions (ORPO, SimPO, DPO)
│   ├── schedulers/   # Learning rate schedulers
│   └── data/         # Data types and validation
│
├── evaluation/        # Quiz evaluation pipeline
├── data_prep/         # Data preparation scripts
├── export/            # Model export utilities
├── post_training/     # Post-training merge + eval
├── models/            # Model configs (YAML)
├── adapters/          # External integrations (tracking, LLM APIs)
└── infrastructure/    # Logging, reproducibility
```

## Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `ORPOPipeline` | `src.training.orpo_pipeline` | Main training orchestration |
| `PipelineConfig` | `src.training.orpo_pipeline` | Training hyperparameters |
| `TopicProgress` | `src.training.orpo_pipeline` | Per-topic training state |
| `LoRAMerger` | `src.training.merge` | Adapter merging |
| `QuizEvaluator` | `src.evaluation` | Post-training evaluation |

## Design Principles

**Dependency rule**: Core has no external dependencies. Backends depend on Core, not vice versa. Training pipeline orchestrates both.

**Interface segregation**: Each module exposes minimal interfaces (`BaseLoss`, `ModelInterface`, `TrainerInterface`).

**Dependency injection**: Losses and backends are injected, not hard-coded:

```python
trainer = Trainer(model, loss_fn=ORPOLoss(config))
```

## Data Flow

```
CLI (run_orpo_training.py)
  → ORPOPipeline
    → Backend (MLX/PyTorch)
      → ORPOLoss (core)
      → LoRA adapters
    → QuizEvaluator (evaluation)
    → Checkpoint save/resume
```

## Testing

- **Unit tests** (`tests/unit/`): Core modules in isolation, no ML frameworks
- **Integration tests** (`tests/integration/`): Backend + training interactions
- **Data quality tests**: Validate preference data source attribution
