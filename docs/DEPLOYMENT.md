# Production Deployment Guide

## Environment Setup

### Mac Studio (M2 Ultra) — MLX Backend

```bash
pip install mlx mlx-lm transformers datasets
python -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}')"
```

**Requirements**: macOS 13.3+, Apple Silicon, 32GB+ RAM (64GB+ for 13B+ models)

### Cloud GPU (CUDA) — PyTorch Backend

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft trl datasets
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Requirements**: CUDA 11.8+, 16GB+ VRAM for 7B models, 24GB+ for 13B

---

## Backend Selection

| Scenario | Backend | Reason |
|----------|---------|--------|
| Mac Studio M2 Ultra | **MLX** | No MPS corruption bugs, native optimization |
| Cloud GPU (CUDA) | **PyTorch** | Mature ecosystem, TRL support |
| Local development | **MLX** | Fast iteration on Apple Silicon |

---

## Configuration

### MLX (Mac Studio)

```yaml
# configs/training_pipeline.yaml
backend:
  type: mlx
  device: auto
  dtype: float16

model:
  name: Qwen/Qwen2.5-7B-Instruct

orpo:
  steps_per_topic: 100
  learning_rate: 0.0003
  lambda_orpo: 0.1
  batch_size: 4
  lora_rank: 8
  lora_alpha: 16
```

### PyTorch (Cloud GPU)

```yaml
# configs/training_pipeline_cloud.yaml
backend:
  type: pytorch
  device: cuda
  dtype: float16

model:
  name: Qwen/Qwen2.5-7B-Instruct

orpo:
  steps_per_topic: 100
  learning_rate: 0.0003
  lambda_orpo: 0.1
  batch_size: 8
  lora_rank: 8
  lora_alpha: 16
```

### Python Configuration

```python
from src.backends import create_backend
from src.training.orpo_pipeline import ORPOPipeline, PipelineConfig

backend = create_backend(backend_type="mlx", device="auto", dtype="float16")

config = PipelineConfig(
    orpo_steps_per_topic=100,
    orpo_learning_rate=3e-4,
    orpo_lambda=0.1,
    output_dir="./output",
    checkpoint_dir="./checkpoints",
)

pipeline = ORPOPipeline(model=None, tokenizer=None, config=config, backend=backend)
```

---

## Training Workflows

### Full Curriculum Training

```python
preference_data = load_preference_pairs("./data/bob_loukas/textbook/preference_data.jsonl")

result = pipeline.train_curriculum(preference_data=preference_data)

if result.success:
    print(f"Passed: {result.passed_topics}/{result.total_topics}")
```

### Single Topic Training

```python
topic_data = {
    "preference_pairs": [
        {"prompt": "...", "chosen": "...", "rejected": "..."},
    ],
}
result = pipeline.train_topic(topic_data, "test-topic")
```

### Resume from Checkpoint

```python
checkpoint_path = Path("./checkpoints/latest.json")
if checkpoint_path.exists():
    pipeline.resume_from_checkpoint(checkpoint_path)
result = pipeline.train_curriculum(preference_data=preference_data)
```

---

## Monitoring

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/training.log'),
        logging.StreamHandler(),
    ],
)
```

```bash
# Watch training progress
tail -f logs/training.log | grep -E "(Topic|Loss|ETA)"
```

---

## Performance Optimization

### MLX

- Start with `batch_size=2` for 7B on 32GB, `batch_size=4` on 64GB
- Use `dtype: float16`
- MLX uses unified memory — no manual cache management needed

### PyTorch

- Use gradient checkpointing for large models
- `bfloat16` for Ampere+ GPUs (A100, RTX 30xx/40xx)
- Gradient accumulation: `batch_size=4, gradient_accumulation_steps=4` for effective batch 16

---

## Troubleshooting

### Out of Memory

Reduce `batch_size`, enable `gradient_checkpointing`, reduce `max_seq_length`, use `float16`.

### Loss Not Decreasing

Lower learning rate, verify data quality (`prompt`, `chosen`, `rejected` fields), increase `steps_per_topic`.

### Adapter Corruption (PyTorch MPS Only)

Switch to MLX backend. Known PyTorch MPS bug — cannot be fixed in application code.

### Backend Not Found

```bash
pip install mlx mlx-lm        # MLX
pip install torch transformers  # PyTorch
```

---

## Benchmarks

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

---

## Production Checklist

- [ ] Choose backend (MLX for Mac, PyTorch for cloud)
- [ ] Configure batch sizes for available memory
- [ ] Enable checkpointing
- [ ] Set up logging
- [ ] Test on single topic before full curriculum
- [ ] Verify adapter save/load
- [ ] Test recovery from interruption
