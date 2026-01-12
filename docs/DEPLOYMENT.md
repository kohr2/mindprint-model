

# Production Deployment Guide

Guide for deploying the dual-backend training system in production environments.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Backend Selection](#backend-selection)
3. [Configuration](#configuration)
4. [Training Workflows](#training-workflows)
5. [Monitoring & Debugging](#monitoring--debugging)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

---

## Environment Setup

### Mac Studio (M2 Ultra) - MLX Backend

**Recommended for:** Development, testing, Mac-based production training

```bash
# Install MLX dependencies
pip install mlx mlx-lm

# Install shared dependencies
pip install transformers datasets

# Optional: PyTorch for fallback
pip install torch torchvision

# Verify MLX installation
python -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}')"
```

**System Requirements:**
- macOS 13.3+ (Ventura or later)
- Apple Silicon (M1/M2/M3 series)
- 32GB+ RAM recommended for 7B models
- 64GB+ RAM recommended for 13B+ models

### Cloud GPU (CUDA) - PyTorch Backend

**Recommended for:** Large-scale training, production deployments

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install training dependencies
pip install transformers peft trl datasets

# Install MLX for optional fallback
pip install mlx mlx-lm

# Verify CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**System Requirements:**
- CUDA 11.8+ or 12.1+
- NVIDIA GPU with 16GB+ VRAM for 7B models
- 24GB+ VRAM recommended for 13B models
- 40GB+ VRAM for 30B+ models

---

## Backend Selection

### Decision Matrix

| Scenario | Recommended Backend | Reason |
|----------|-------------------|---------|
| Mac Studio M2 Ultra | **MLX** | No corruption bugs, native optimization |
| Cloud GPU (CUDA) | **PyTorch** | Mature ecosystem, TRL support |
| Local development | **MLX** | Fast iteration on Apple Silicon |
| Large-scale production | **PyTorch** | Proven at scale |
| Testing both approaches | **Both** | Validate consistency |

### Configuration Files

#### MLX (Mac Studio)

```yaml
# configs/training_pipeline.yaml
backend:
  type: mlx
  device: auto  # Always uses GPU on Apple Silicon
  dtype: float16

model:
  name: Qwen/Qwen2.5-7B-Instruct
  dtype: float16

sft:
  epochs_per_topic: 3
  learning_rate: 0.0003
  batch_size: 4
  lora_rank: 8
  lora_alpha: 16

dpo:
  steps_per_topic: 100
  learning_rate: 0.0000005
  batch_size: 2
  beta: 0.1
```

#### PyTorch (Cloud GPU)

```yaml
# configs/training_pipeline_cloud.yaml
backend:
  type: pytorch
  device: cuda
  dtype: float16

model:
  name: Qwen/Qwen2.5-7B-Instruct
  dtype: float16
  device: cuda

sft:
  epochs_per_topic: 3
  learning_rate: 0.0003
  batch_size: 8  # Can use larger batches on GPU
  lora_rank: 8
  lora_alpha: 16

dpo:
  steps_per_topic: 100
  learning_rate: 0.0000005
  batch_size: 4  # Larger batches for faster training
  beta: 0.1
```

---

## Configuration

### Environment Variables

```bash
# Set backend type
export MINDPRINT_BACKEND=mlx  # or pytorch

# Set device
export MINDPRINT_DEVICE=auto  # auto, cuda, mps, cpu

# Set output directory
export MINDPRINT_OUTPUT_DIR=./output

# Set log level
export MINDPRINT_LOG_LEVEL=INFO
```

### Python Configuration

```python
from src.backends import create_backend
from src.training.dpo_pipeline import DPOPipeline, PipelineConfig

# Create backend
backend = create_backend(
    backend_type="mlx",  # or "pytorch"
    device="auto",
    dtype="float16",
    seed=42,
)

# Configure pipeline
config = PipelineConfig(
    backend_type="mlx",
    backend_device="auto",
    backend_dtype="float16",
    sft_epochs_per_topic=3,
    sft_learning_rate=3e-4,
    sft_batch_size=4,
    dpo_steps_per_topic=100,
    dpo_learning_rate=5e-7,
    dpo_batch_size=2,
    output_dir="./output",
    checkpoint_dir="./checkpoints",
)

# Create pipeline
pipeline = DPOPipeline(
    model=None,  # Loaded via backend
    tokenizer=None,
    config=config,
    backend=backend,
)
```

---

## Training Workflows

### Workflow 1: Full Curriculum Training

```python
# Load curriculum data
sft_data = load_sft_data("./data/bob_loukas/sft_data.json")
preference_data = load_preference_pairs("./data/bob_loukas/preference_pairs.json")

# Train
result = pipeline.train_curriculum(
    sft_data=sft_data,
    preference_data=preference_data,
)

# Save final model
if result.success:
    final_model_path = Path("./output/final_model")
    pipeline.model.save_pretrained(final_model_path)
    print(f"✓ Saved final model to {final_model_path}")
```

### Workflow 2: Single Topic Training (Testing)

```python
# Test on single topic
topic_data = {
    "sft_data": [
        {"question": "What is Bitcoin?", "answer": "..."},
        # More Q&A pairs
    ],
    "preference_pairs": [
        {"prompt": "...", "chosen": "...", "rejected": "..."},
        # More pairs
    ],
}

result = pipeline.train_topic(topic_data, "test-topic")

if result.success:
    print(f"✓ Topic trained successfully")
    print(f"  SFT loss: {result.sft_loss:.4f}")
    print(f"  DPO loss: {result.dpo_loss:.4f}")
```

### Workflow 3: Incremental Training (Checkpointing)

```python
# Resume from checkpoint
checkpoint_path = Path("./checkpoints/latest.json")
if checkpoint_path.exists():
    pipeline.resume_from_checkpoint(checkpoint_path)
    print("✓ Resumed from checkpoint")

# Train with checkpointing
result = pipeline.train_curriculum(
    sft_data=sft_data,
    preference_data=preference_data,
)

# Checkpoint is saved automatically after each topic
```

---

## Monitoring & Debugging

### Logging Configuration

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/training.log'),
        logging.StreamHandler()
    ]
)

# Get logger
logger = logging.getLogger(__name__)
```

### Metrics Tracking

```python
# Track training metrics
metrics = {
    "topics_completed": 0,
    "total_training_time": 0,
    "average_sft_loss": 0,
    "average_dpo_loss": 0,
    "voice_scores": [],
}

for topic_result in results:
    if topic_result.success:
        metrics["topics_completed"] += 1
        metrics["total_training_time"] += topic_result.training_time_seconds
        metrics["average_sft_loss"] += topic_result.sft_loss
        metrics["voice_scores"].append(topic_result.voice_score)

# Log metrics
logger.info(f"Training complete: {metrics}")
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger("src.backends").setLevel(logging.DEBUG)
logging.getLogger("src.training").setLevel(logging.DEBUG)

# Enable additional checks
config.validate_data = True
config.save_intermediate = True
```

---

## Performance Optimization

### MLX Backend Optimization

**Memory Management:**
```python
# MLX uses unified memory - no manual management needed
# Cache is automatically managed

# Optional: Force evaluation of lazy computations
import mlx.core as mx
mx.eval(model.parameters())
```

**Batch Size Tuning:**
- Start with batch_size=2 for 7B models on 32GB RAM
- Increase to batch_size=4 on 64GB RAM
- Monitor memory usage: `vm_stat | grep "Pages active"`

**Mixed Precision:**
```yaml
backend:
  dtype: float16  # Recommended for MLX
```

### PyTorch Backend Optimization

**Memory Management:**
```python
# Clear CUDA cache between epochs
import torch
torch.cuda.empty_cache()

# Use gradient checkpointing for large models
config.gradient_checkpointing = True
```

**Batch Size Tuning:**
- 7B models: batch_size=8 on 16GB VRAM, batch_size=16 on 24GB VRAM
- 13B models: batch_size=4 on 24GB VRAM, batch_size=8 on 40GB VRAM
- Use gradient accumulation for effective larger batches

**Mixed Precision:**
```yaml
backend:
  dtype: float16  # For CUDA
  # or
  dtype: bfloat16  # For Ampere+ GPUs (RTX 30xx/40xx, A100)
```

**Gradient Accumulation:**
```yaml
sft:
  batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch size = 16
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Symptoms:** Training crashes with OOM error

**Solutions:**

1. **Reduce batch size:**
   ```yaml
   sft:
     batch_size: 2  # Down from 4
   dpo:
     batch_size: 1  # Down from 2
   ```

2. **Enable gradient checkpointing:**
   ```yaml
   sft:
     gradient_checkpointing: true
   ```

3. **Reduce sequence length:**
   ```yaml
   sft:
     max_seq_length: 1024  # Down from 2048
   ```

4. **Switch to lower precision:**
   ```yaml
   backend:
     dtype: float16  # From float32
   ```

### Issue: Training is Slow

**Symptoms:** Training takes much longer than expected

**Solutions:**

1. **Increase batch size (if memory allows):**
   ```yaml
   sft:
     batch_size: 8  # Up from 4
   ```

2. **Use mixed precision:**
   ```yaml
   backend:
     dtype: float16  # or bfloat16
   ```

3. **Optimize data loading:**
   ```python
   # Use DataLoader with multiple workers
   dataloader = DataLoader(
       dataset,
       batch_size=batch_size,
       num_workers=4,
       pin_memory=True,
   )
   ```

4. **Profile to find bottlenecks:**
   ```bash
   # PyTorch profiling
   python -m torch.utils.bottleneck scripts/train.py

   # MLX profiling
   python -m cProfile -o profile.stats scripts/train.py
   ```

### Issue: Loss Not Decreasing

**Symptoms:** Loss stays flat or increases

**Solutions:**

1. **Check learning rate:**
   ```yaml
   sft:
     learning_rate: 0.0001  # Try lower LR
   ```

2. **Verify data quality:**
   ```python
   # Check for empty/malformed data
   for item in train_data:
       assert "question" in item
       assert "answer" in item
       assert len(item["answer"]) > 0
   ```

3. **Increase training epochs:**
   ```yaml
   sft:
     num_epochs: 5  # Up from 3
   ```

4. **Check for NaN losses:**
   ```python
   if torch.isnan(loss):
       logger.error("NaN loss detected!")
       # Skip batch or reduce LR
   ```

### Issue: Adapter Corruption (PyTorch MPS Only)

**Symptoms:**
- Model outputs degrade after training
- Voice scores drop to 0.00
- Model produces nonsense after merge_and_unload()

**Solution:** **Switch to MLX backend**
```yaml
backend:
  type: mlx  # Instead of pytorch
```

This is a known PyTorch MPS bug that cannot be fixed in application code.

### Issue: Backend Not Found

**Symptoms:** `ValueError: Unknown backend: mlx`

**Solutions:**

1. **Install backend dependencies:**
   ```bash
   pip install mlx mlx-lm  # For MLX
   pip install torch transformers peft trl  # For PyTorch
   ```

2. **Import backend module:**
   ```python
   import src.backends.mlx  # Triggers registration
   ```

3. **Verify installation:**
   ```python
   from src.backends import BackendRegistry
   print(BackendRegistry.list_backends())
   ```

---

## Performance Benchmarks

### Mac Studio M2 Ultra (64GB)

| Model | Backend | Batch Size | Samples/sec | Memory |
|-------|---------|------------|-------------|--------|
| Qwen2.5-7B | MLX | 4 | ~2.5 | 32GB |
| Qwen2.5-7B | PyTorch MPS | 2 | ~1.8 | 28GB |
| Qwen2.5-13B | MLX | 2 | ~1.2 | 48GB |

### Cloud GPU (NVIDIA A100 40GB)

| Model | Backend | Batch Size | Samples/sec | Memory |
|-------|---------|------------|-------------|--------|
| Qwen2.5-7B | PyTorch | 16 | ~12.0 | 24GB |
| Qwen2.5-13B | PyTorch | 8 | ~6.5 | 36GB |
| Qwen2.5-30B | PyTorch | 4 | ~2.8 | 38GB |

*Benchmarks are approximate and depend on sequence length, LoRA rank, and other factors.*

---

## Production Checklist

Before deploying to production:

- [ ] Choose appropriate backend (MLX for Mac, PyTorch for cloud)
- [ ] Configure batch sizes for available memory
- [ ] Enable checkpointing for long-running training
- [ ] Set up logging and monitoring
- [ ] Test on single topic before full curriculum
- [ ] Verify adapter save/load works
- [ ] Set up error alerting
- [ ] Document expected training times
- [ ] Create backup strategy for checkpoints
- [ ] Test recovery from interruption

---

## Support

For issues or questions:

1. Check [src/backends/README.md](../src/backends/README.md) for backend-specific docs
2. Review test files in `tests/` for usage examples
3. Check GitHub issues for known problems
4. Enable debug logging for detailed diagnostics

---

## References

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [TRL Documentation](https://huggingface.co/docs/trl/)
