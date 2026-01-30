# Training Guide

## Overview

Mindprint Model supports multiple training approaches:

1. **SFT + DPO/SimPO**: Two-stage training (recommended for fine-tuning)
2. **ORPO**: Single-stage training (faster, good for from-scratch)

## Basic Training

### With SimPO (Recommended)

```bash
python scripts/run_dpo_training.py \
    --config configs/training_pipeline.yaml \
    --backend mlx \
    --loss-type simpo
```

### With DPO

```bash
python scripts/run_dpo_training.py \
    --config configs/training_pipeline.yaml \
    --backend mlx \
    --loss-type dpo
```

### With ORPO (Single-stage)

```bash
python scripts/run_dpo_training.py \
    --config configs/training_pipeline.yaml \
    --backend mlx \
    --loss-type orpo
```

## Configuration

### Loss Function Selection

```yaml
training:
  loss_type: simpo  # "dpo", "simpo", "orpo"
```

### Learning Rate Scheduling

```yaml
training:
  warmup_ratio: 0.1  # 10% of steps for warmup
  learning_rate: 3e-4
```

### Gradient Accumulation

```yaml
training:
  gradient_accumulation_steps: 8  # Effective batch = 8 * batch_size
```

### LoRA Configuration

```yaml
sft:
  lora_rank: 32  # Increased from 8 for better quality
  lora_alpha: 64  # Typically 2x rank
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

### NEFTune

```yaml
training:
  neftune_noise_alpha: 5.0  # Enable NEFTune noise injection
```

## Monitoring Training

### On Mac Studio

```bash
./scripts/local_monitor.sh
```

### With Experiment Tracking

```python
from src.adapters.tracking import WandBTracker, WandBConfig

tracker = WandBTracker(
    WandBConfig(project="mindprint"),
    run_config={"loss_type": "simpo", ...}
)

# Training loop
for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    tracker.log_metrics({"loss": loss}, step=step)
```

## Best Practices

1. **Start with SimPO** - Best quality/speed tradeoff
2. **Use gradient accumulation** - DPO/SimPO benefit from larger batches
3. **Enable NEFTune** - Simple 10-15% improvement
4. **Target all linear layers** - Better quality than attention-only
5. **Use learning rate scheduling** - More stable training

## Troubleshooting

### Low Voice Scores

- Increase DPO/SimPO steps
- Check data quality
- Verify LoRA adapters are training

### Training Instability

- Reduce learning rate
- Increase gradient accumulation
- Enable gradient clipping

### Out of Memory

- Reduce batch size
- Increase gradient accumulation
- Use quantization (QLoRA)
