# Backends System

Multi-framework ML backend abstraction for ORPO training.

## Overview

Unified interface for training with different ML frameworks:

- **MLX** (mlx-lm) — Apple Silicon M-series (recommended for Mac Studio)
- **PyTorch** (transformers, PEFT) — Cloud CUDA GPUs

Solves PyTorch MPS corruption bugs by providing a stable MLX alternative.

## Quick Start

```python
from src.backends import create_backend

# Mac Studio (MLX)
backend = create_backend("mlx", device="auto", dtype="float16")

# Cloud GPU (PyTorch)
backend = create_backend("pytorch", device="cuda", dtype="float16")

# Load model and train
model = backend.load_model("Qwen/Qwen2.5-7B-Instruct")
trainer = backend.create_sft_trainer(model, config)
result = trainer.train(train_data)
```

## Configuration

```yaml
# configs/training_pipeline.yaml
backend:
  type: mlx       # "pytorch", "mlx", or null for legacy
  device: auto     # "auto", "mps", "cuda", "cpu"
  dtype: float16
```

## Architecture

```
src/backends/
├── protocol.py           # BackendProtocol interface
├── model_interface.py    # ModelInterface abstraction
├── trainer_interface.py  # TrainerInterface, TrainingResult
├── adapter_interface.py  # AdapterManager, AdapterConfig
├── factory.py            # BackendRegistry, create_backend()
├── mlx/                  # MLX backend implementation
└── pytorch/              # PyTorch backend implementation
```

## Pipeline Integration

```python
from src.training.orpo_pipeline import ORPOPipeline, PipelineConfig
from src.backends import create_backend

backend = create_backend("mlx", device="auto")

config = PipelineConfig(orpo_steps_per_topic=100, orpo_learning_rate=3e-4)
pipeline = ORPOPipeline(model=None, tokenizer=None, config=config, backend=backend)

result = pipeline.train_curriculum()
```

## PyTorch vs MLX

| Feature | PyTorch | MLX |
|---------|---------|-----|
| Target | Cloud CUDA GPU | Mac Studio M-series |
| Device Management | Explicit (MPS/CUDA/CPU) | Unified memory (auto) |
| LoRA Library | PEFT | mlx-lm |
| Adapter Stability | Corruption on MPS | Stable |

## LoRA Adapter Implementation (MLX)

Uses `mlx_lm.tuner.lora` for LoRA layers:

1. Converts `nn.Linear` → `LoRALinear` for target modules (q_proj, v_proj, o_proj, up_proj, down_proj)
2. Only LoRA parameters receive gradients during training
3. ~8M trainable params for Qwen2.5-7B (vs 7B total)

See `docs/mlx/MLX_LORA_TRAINING_ISSUE.md` for investigation details.

## Testing

```bash
pytest tests/unit/backends/ -v
pytest tests/integration/test_backend_pipeline.py -v
```

## API Reference

### BackendProtocol

```python
class BackendProtocol(Protocol):
    def load_model(self, model_path, ...) -> ModelInterface: ...
    def create_sft_trainer(self, model, config) -> TrainerInterface: ...
```

### ModelInterface

```python
class ModelInterface(ABC):
    def generate(self, input_ids, ...) -> Any: ...
    def forward(self, input_ids, ...) -> Dict: ...
    def save_adapter(self, path) -> None: ...
    def load_adapter(self, path) -> None: ...
```

### TrainerInterface

```python
class TrainerInterface(ABC):
    def train(self, train_data) -> TrainingResult: ...
    def train_on_topic(self, topic_data, topic_id) -> TrainingResult: ...
    def save_adapter(self, path) -> Path: ...
```

## Troubleshooting

**PyTorch MPS corruption**: Switch to MLX backend. Known bug — cannot be fixed in application code.

**"Unknown backend"**: Import the backend package: `import src.backends.mlx`

**"mlx-lm not installed"**: `pip install mlx mlx-lm`
