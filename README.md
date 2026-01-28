# Mindprint RLHF Fine-Tuning

RLHF (Reinforcement Learning from Human Feedback) fine-tuning system for creating personalized language models using DPO (Direct Preference Optimization) and SFT (Supervised Fine-Tuning) with LoRA adapters.

## Features

- **Multi-Backend Support**: Train with PyTorch (CUDA/CPU) or MLX (Apple Silicon)
- **DPO Training**: Direct Preference Optimization for aligning models with human preferences
- **SFT Training**: Supervised Fine-Tuning for teaching specific knowledge
- **LoRA Adapters**: Parameter-efficient fine-tuning with Low-Rank Adaptation
- **Curriculum Learning**: Progressive training across multiple topics
- **Voice Fidelity Evaluation**: Assess how well the model maintains target voice/style

### Backend Status

**PyTorch Backend** (Cloud GPU):
- Status: ✅ Production Ready
- Platform: CUDA GPUs
- Features: Full TRL support, PEFT adapters
- Known Issues: MPS backend has adapter corruption bugs (use MLX instead)

**MLX Backend** (Mac Studio):
- Status: ✅ Production Ready (LoRA support fixed Jan 28, 2026)
- Platform: Apple Silicon (M1/M2/M3)
- Features: Native optimization, proper LoRA implementation, no corruption issues
- Recent Fixes: Implemented proper LoRA layer conversion (was stub before)

See `docs/mlx/MLX_LORA_TRAINING_ISSUE.md` for details on the MLX LoRA investigation and fix.

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd mindprint-model

# Install dependencies
pip install -r requirements.txt

# For PyTorch backend (CUDA GPU)
pip install torch transformers peft trl

# For MLX backend (Apple Silicon)
pip install mlx mlx-lm
```

### Basic Training

```bash
# Train with PyTorch backend (CUDA GPU)
python scripts/run_dpo_training.py \
  --config configs/training_pipeline.yaml \
  --backend pytorch \
  --device cuda

# Train with MLX backend (Mac Studio)
python scripts/run_dpo_training.py \
  --config configs/training_pipeline.yaml \
  --backend mlx \
  --device auto
```

### Configuration

Edit `configs/training_pipeline.yaml`:

```yaml
# Backend configuration
backend:
  type: pytorch  # "pytorch" or "mlx"
  device: auto   # "auto", "cuda", "mps", "cpu"
  dtype: float16 # "float16", "float32", "bfloat16"

# Model configuration
model:
  name: Qwen/Qwen2.5-7B-Instruct

# SFT configuration
sft:
  epochs_per_topic: 3
  learning_rate: 3e-4
  batch_size: 4
  lora_r: 8
  lora_alpha: 16

# DPO configuration
dpo:
  steps_per_topic: 100
  learning_rate: 5e-7
  beta: 0.1
  lora_r: 1
  lora_alpha: 2
```

## Supported Models

| Model | Parameters | Layers | VRAM (INT4) | VRAM (FP16) | Context | Platform |
|-------|------------|--------|-------------|-------------|---------|----------|
| Gemma-3-12B | 12B | 48 | 12GB | 24GB | 128K | Mac Studio (fp16), Cloud GPU |
| Qwen2.5-7B | 7B | 28 | 8GB | 14GB | 131K | Mac Studio (fp16), Cloud GPU |
| **Qwen2.5-72B** | **72.7B** | **80** | **36GB** | **145GB** | **131K** | **Mac Studio (int4), Cloud GPU** |

**Notes:**
- **Mac Studio M2 Ultra (64GB)**: Can run Gemma-3-12B and Qwen2.5-7B in fp16, or Qwen2.5-72B in int4 quantization
- **Cloud GPU (8x H100)**: Required for Qwen2.5-72B in fp16/bf16
- **INT4 quantization**: Enables large models on limited memory with minimal quality loss

For detailed model specifications and training recommendations, see [docs/models/](docs/models/).

## Architecture

### Backend Abstraction Layer

The project uses a backend abstraction layer to support multiple ML frameworks:

```
Application Layer (DPOPipeline, CLI)
         ↓
Backend Interface (BackendProtocol, ModelInterface, TrainerInterface)
         ↓
    ┌────┴────┐
    ↓         ↓
PyTorch    MLX Backend
Backend    (Mac Studio)
(Cloud GPU)
```

**Key Components**:
- **BackendProtocol**: Defines the interface all backends must implement
- **ModelInterface**: Unified model wrapper for both PyTorch and MLX
- **TrainerInterface**: Unified trainer interface for SFT and DPO
- **AdapterManager**: Handles LoRA adapter operations

### Directory Structure

```
mindprint-model/
├── src/
│   ├── backends/              # Backend abstraction layer
│   │   ├── protocol.py        # Core interfaces
│   │   ├── factory.py         # Backend registry
│   │   ├── model_interface.py # Model abstraction
│   │   ├── trainer_interface.py # Trainer abstraction
│   │   ├── pytorch/           # PyTorch implementation
│   │   │   ├── backend.py
│   │   │   ├── sft_trainer.py
│   │   │   └── dpo_trainer.py
│   │   └── mlx/               # MLX implementation
│   │       ├── backend.py
│   │       ├── sft_trainer.py
│   │       └── dpo_trainer.py
│   ├── training/              # Training pipelines
│   │   └── dpo_pipeline.py    # DPO training pipeline
│   ├── data/                  # Data loading and processing
│   ├── evaluation/            # Evaluation and metrics
│   └── utils/                 # Shared utilities
├── configs/                   # Configuration files
├── data/                      # Training data
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── docs/                      # Documentation
└── scripts/                   # CLI scripts
```

## Why Multiple Backends?

### PyTorch MPS Corruption Bug

PyTorch's MPS (Metal Performance Shaders) backend has a fundamental bug with non-contiguous tensors that causes `merge_and_unload()` operations to silently corrupt LoRA adapters. This affects:
- Merging adapters after training
- Saving/reloading models
- Adapter stacking across training stages

**References**:
- [PyTorch Issue #78043](https://github.com/pytorch/pytorch/issues/78043)
- [PEFT Issue #2502](https://github.com/huggingface/peft/issues/2502)
- [PEFT Issue #2764](https://github.com/huggingface/peft/issues/2764)

### MLX Solution

MLX (Apple's ML framework) provides a stable alternative for Mac Studio training:
- ✅ No adapter corruption issues
- ✅ Native Apple Silicon optimization
- ✅ Unified memory management
- ✅ Reliable LoRA operations

### Deployment Strategy

- **Mac Studio M2 Ultra**: Use MLX backend for local development and training
- **Cloud GPU (CUDA)**: Use PyTorch backend for production training at scale
- **Switch backends**: Change one line in config file

## Training Pipeline

### DPO Pipeline

The DPO (Direct Preference Optimization) pipeline consists of:

1. **Data Preparation**: Load preference pairs (chosen vs rejected responses)
2. **SFT Phase**: Supervised fine-tuning on chosen responses
3. **DPO Phase**: Optimize model to prefer chosen over rejected responses
4. **Evaluation**: Assess voice fidelity and preference alignment

```python
from src.training.dpo_pipeline import DPOPipeline, PipelineConfig
from src.backends import create_backend

# Create backend
backend = create_backend("mlx", device="auto")

# Configure pipeline
config = PipelineConfig(
    backend_type="mlx",
    sft_epochs_per_topic=3,
    dpo_steps_per_topic=100,
)

# Initialize pipeline
pipeline = DPOPipeline(
    model=None,  # Loaded via backend
    tokenizer=None,
    config=config,
    backend=backend,
)

# Train
result = pipeline.train_curriculum()
```

### Curriculum Learning

Train across multiple topics progressively:

```python
topics = [
    "bitcoin_basics",
    "market_cycles",
    "trading_strategy",
]

for topic_id in topics:
    # SFT: Learn topic knowledge
    sft_result = pipeline.train_sft_on_topic(topic_id)

    # DPO: Align preferences
    dpo_result = pipeline.train_dpo_on_topic(topic_id)

    # Evaluate
    score = pipeline.evaluate_voice_fidelity(topic_id)
    print(f"Topic {topic_id} voice score: {score:.2f}")
```

## Evaluation

### Voice Fidelity Score

Measures how well the model maintains the target voice/style:

```python
from src.evaluation.voice_fidelity import calculate_voice_fidelity

score = calculate_voice_fidelity(
    reference_answer="Market cycles follow 4-year patterns...",
    model_response="Bitcoin cycles are driven by halving events...",
    model=sentence_transformer
)

# Score ranges from 0.0 (no similarity) to 1.0 (perfect match)
# Typical good scores: 0.75+
```

### Metrics

- **Training Loss**: Cross-entropy (SFT) or DPO loss
- **Voice Fidelity**: Semantic similarity to reference style
- **Preference Accuracy**: Model's ability to prefer chosen responses
- **Perplexity**: Model confidence on held-out data

## Testing

### Unit and Integration Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test suite
pytest tests/unit/backends/ -v

# Run integration tests
pytest tests/integration/ -v

# Run slow tests (requires --run-slow flag)
pytest tests/integration/test_backend_equivalence.py --run-slow
```

### Real-World Testing on Mac Studio

For testing the MLX backend on Mac Studio:

```bash
# Verify configuration (dry-run)
./scripts/test_dry_run.sh

# Test single topic
./scripts/test_single_topic_mlx.sh unit-01 chapter-01 topic-01

# Monitor training
./scripts/monitor_training.sh

# Deploy from local machine
./scripts/train_on_mac_studio.sh mac-studio.local user
```

See [MLX Real-World Testing Guide](docs/MLX_REAL_WORLD_TESTING.md) for detailed instructions.

## Documentation

- [Backend System README](src/backends/README.md) - API documentation for backend abstraction
- [Migration Guide](docs/MIGRATION.md) - Migrating existing code to use backends
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment instructions
- [MLX Real-World Testing Guide](docs/MLX_REAL_WORLD_TESTING.md) - Testing MLX backend on Mac Studio
- [Adapter Stacking Debug](ADAPTER_STACKING_DEBUG.md) - Debugging adapter issues
- [Testing Instructions](TESTING_INSTRUCTIONS.md) - Testing procedures

## Development

### Adding a New Backend

1. Create backend directory: `src/backends/my_backend/`
2. Implement `BackendProtocol`:
   ```python
   from src.backends.protocol import BackendProtocol
   from src.backends.factory import register_backend

   @register_backend("my_backend")
   class MyBackend(BackendProtocol):
       def load_model(self, model_path): ...
       def create_sft_trainer(self, model, config): ...
       def create_dpo_trainer(self, model, ref_model, config): ...
   ```
3. Implement trainers and model wrapper
4. Add tests
5. Update documentation

### Code Quality

- **Test-First**: Write tests before implementation (TDD)
- **Type Hints**: All functions must have type annotations
- **Linting**: Use `ruff check .` before committing
- **Coverage**: Maintain >80% test coverage

## Performance

### Benchmarks

**Qwen2.5-7B**:

| Backend | Device | SFT Time/Epoch | DPO Time/100 steps | Memory Usage |
|---------|--------|----------------|-------------------|--------------|
| PyTorch | CUDA (A100) | 45 min | 12 min | 24 GB |
| PyTorch | CPU | 8 hours | 2 hours | 16 GB |
| MLX | M2 Ultra | 2 hours | 30 min | 32 GB unified |

**Qwen2.5-72B** (with INT4 quantization):

| Backend | Device | SFT Time/Topic | DPO Time/Topic | Memory Usage |
|---------|--------|----------------|----------------|--------------|
| MLX | M2 Ultra (64GB) | 6-8 hours | 8-10 hours | 44-48 GB unified |
| PyTorch | 8x H100 80GB | 2-3 hours | 3-4 hours | ~500 GB (distributed) |

*Note: Times are approximate and depend on dataset size, batch size, and sequence length.*

### Optimization Tips

**PyTorch**:
- Use gradient checkpointing for large models
- Enable mixed precision (fp16/bf16)
- Increase batch size on larger GPUs
- Use gradient accumulation for effective larger batches

**MLX**:
- MLX automatically optimizes for Apple Silicon
- Unified memory allows larger batch sizes
- Use bf16 for better numerical stability

## Troubleshooting

### Common Issues

**PyTorch MPS Corruption**:
```yaml
# Switch to MLX backend
backend:
  type: mlx
  device: auto
```

**Out of Memory**:
- Reduce batch size
- Enable gradient checkpointing
- Use smaller LoRA rank
- Reduce max sequence length

**Slow Training**:
- Use mixed precision (fp16/bf16)
- Increase batch size (if memory allows)
- Use gradient accumulation
- Profile with `torch.profiler` or MLX profiling tools

**Adapter Not Loading**:
- Ensure adapter was saved with same backend
- PyTorch and MLX adapters are not compatible
- Check adapter path and file permissions

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed troubleshooting.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Write tests for new functionality
4. Ensure all tests pass: `pytest`
5. Run linting: `ruff check .`
6. Commit with clear messages
7. Open a pull request

## License

[Add license information]

## Citation

```bibtex
[Add citation information if applicable]
```

## Acknowledgments

- Based on [Transformers](https://github.com/huggingface/transformers)
- Uses [PEFT](https://github.com/huggingface/peft) for LoRA
- Uses [TRL](https://github.com/huggingface/trl) for DPO (PyTorch)
- Uses [MLX](https://github.com/ml-explore/mlx) for Apple Silicon

## Contact

[Add contact information]
