# Mindprint RLHF Fine-Tuning

RLHF (Reinforcement Learning from Human Feedback) fine-tuning system for creating personalized language models using ORPO (Odds Ratio Preference Optimization) with LoRA adapters.

## Features

- **State-of-the-Art Training**: ORPO (Odds Ratio Preference Optimization) for single-stage training
- **Multi-Backend Support**: Train with PyTorch (CUDA/CPU) or MLX (Apple Silicon)
- **Clean Architecture**: Modular, testable, production-ready codebase
- **Comprehensive Testing**: Unit, integration, property-based, and benchmark tests
- **Professional Documentation**: API reference, tutorials, architecture guides
- **Production Infrastructure**: CI/CD, experiment tracking, reproducibility
- **LoRA Adapters**: Parameter-efficient fine-tuning with optimized configuration
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

**ORPO Training (Odds Ratio Preference Optimization)**

```bash
# Train with MLX backend (Mac Studio)
python scripts/run_orpo_training.py \
  --config configs/training_pipeline.yaml \
  --backend mlx

# Train with PyTorch backend (Cloud GPU)
python scripts/run_orpo_training.py \
  --config configs/training_pipeline.yaml \
  --backend pytorch --data-dir ./data/bob_loukas/transcripts
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

# ORPO configuration
orpo:
  steps_per_topic: 100
  learning_rate: 0.0003  # 3e-4
  batch_size: 4
  lambda_orpo: 0.1  # Weight for preference term
  lora_rank: 8
  lora_alpha: 16
  target_modules:
    - q_proj
    - v_proj
    - o_proj
    - up_proj
    - down_proj
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
Application Layer (ORPOPipeline, CLI)
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
- **TrainerInterface**: Unified trainer interface for ORPO training
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
│   │   │   └── pytorch_orpo_trainer.py
│   │   └── mlx/               # MLX implementation
│   │       ├── backend.py
│   │       └── mlx_orpo_trainer.py
│   ├── training/              # Training pipelines
│   │   └── orpo_pipeline.py   # ORPO training pipeline
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

### ORPO Pipeline

The ORPO (Odds Ratio Preference Optimization) pipeline combines SFT and preference alignment in a single training stage:

1. **Data Preparation**: Load preference pairs (chosen vs rejected responses)
2. **ORPO Training**: Single-stage optimization combining NLL loss (SFT) and odds ratio loss (preference)
3. **Evaluation**: Assess voice fidelity and preference alignment

Key advantages over two-stage SFT+DPO:
- **Faster**: Single training stage instead of two
- **Better Quality**: Often produces better instruction following
- **No Reference Model**: Unlike DPO, doesn't require reference model
- **Simpler Pipeline**: Fewer hyperparameters to tune

```python
from src.training.orpo_pipeline import DPOPipeline, PipelineConfig
from src.backends import create_backend

# Create backend
backend = create_backend("mlx", device="auto")

# Configure pipeline
config = PipelineConfig(
    backend_type="mlx",
    orpo_steps_per_topic=100,
    lambda_orpo=0.1,  # Balance between NLL and OR loss
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

Train across multiple topics progressively with ORPO:

```python
topics = [
    "bitcoin_basics",
    "market_cycles",
    "trading_strategy",
]

for topic_id in topics:
    # ORPO: Combined SFT + preference alignment in single stage
    orpo_result = pipeline.train_orpo_on_topic(topic_id)

    # Evaluate
    score = pipeline.evaluate_voice_fidelity(topic_id)
    print(f"Topic {topic_id} voice score: {score:.2f}")
    print(f"ORPO loss: {orpo_result.loss:.4f}")
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

- **Training Loss**: ORPO loss (combined NLL + odds ratio loss)
  - NLL Loss: Cross-entropy on chosen responses
  - Odds Ratio Loss: Preference alignment loss
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

### Mac Studio Training Workflow

The recommended workflow uses git to sync code between your development machine and Mac Studio:

**One-Time Setup on Mac Studio:**

```bash
# SSH to Mac Studio
ssh memetica-studio@100.87.103.70

# Clone the repository
cd ~/Documents/Memetica/Code
git clone https://github.com/kohr2/mindprint-model
cd mindprint-model

# Install dependencies
pip3 install -r requirements.txt
pip3 install mlx mlx-lm
```

**Normal Workflow:**

1. **On your development machine** (MacBook Air):
   ```bash
   # Make changes, commit and push
   git add .
   git commit -m "Update training config"
   git push origin main
   ```

2. **On Mac Studio**:
   ```bash
   # SSH to Mac Studio
   ssh memetica-studio@100.87.103.70
   cd ~/mindprint-model
   
   # Pull latest code and start training
   ./scripts/local_train.sh
   
   # Monitor training (in another terminal)
   ./scripts/local_monitor.sh
   
   # Or follow logs live
   ./scripts/local_monitor.sh --follow
   ```

**Optional: Quick Deploy Script**

If you don't want to SSH manually, you can use the convenience script from your development machine:

```bash
# On MacBook Air
source .env.local
./scripts/quick_deploy.sh
```

This will SSH to Mac Studio and run `local_train.sh` automatically.

**Post-Training Tasks:**

After training completes, run analysis and evaluation directly on Mac Studio:

```bash
# Run post-training pipeline (merge + evaluate + export)
python3 scripts/run_post_training.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter output/transcripts_*/adapters/... \
    --quiz-data data/bob_loukas/transcripts \
    --output output/merged_model

# Run evaluation
python3 scripts/run_evaluation.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapter output/merged_model/merged \
    --quiz-data data/bob_loukas/transcripts \
    --output eval_results

# Run analysis
python3 scripts/analyze_training_results.py \
    --checkpoint checkpoints/latest.json \
    --log logs/training_*.log \
    --output analysis
```

**Benefits of Git-Based Workflow:**

- ✅ Simpler: No complex SSH/rsync scripts
- ✅ Version controlled: All changes tracked in git
- ✅ Standard workflow: Uses git pull/push everyone understands
- ✅ Flexible: Easy to run commands directly on Mac Studio
- ✅ No file syncing: Git handles code synchronization

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
       def create_orpo_trainer(self, model, config): ...
   ```
3. Implement ORPO trainer and model wrapper
4. Add tests
5. Update documentation

### Code Quality

- **Test-First**: Write tests before implementation (TDD)
- **Type Hints**: All functions must have type annotations
- **Linting**: Use `ruff check .` before committing
- **Coverage**: Maintain >80% test coverage

## Performance

### Benchmarks

**Qwen2.5-7B** (ORPO):

| Backend | Device | ORPO Time/100 steps | Memory Usage |
|---------|--------|-------------------|--------------|
| PyTorch | CUDA (A100) | 15 min | 24 GB |
| PyTorch | CPU | 2 hours | 16 GB |
| MLX | M2 Ultra | 45 min | 32 GB unified |

**Qwen2.5-72B** (with INT4 quantization, ORPO):

| Backend | Device | ORPO Time/Topic | Memory Usage |
|---------|--------|----------------|--------------|
| MLX | M2 Ultra (64GB) | 6-8 hours | 44-48 GB unified |
| PyTorch | 8x H100 80GB | 2-3 hours | ~500 GB (distributed) |

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
- ORPO approach from [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- Uses [MLX](https://github.com/ml-explore/mlx) for Apple Silicon

## Contact

[Add contact information]
