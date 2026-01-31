# Training Guide

## Overview

Mindprint Model uses **ORPO** (Odds Ratio Preference Optimization) for all training. ORPO is a single-stage training approach that combines supervised fine-tuning and preference alignment:

- **Single-stage**: No need for separate SFT and preference optimization phases
- **Faster**: Approximately 2x faster than SFT + DPO pipelines
- **Better quality**: Often produces better instruction following than DPO
- **No reference model**: Unlike DPO, doesn't require a separate reference model

## Basic Training

### Start ORPO Training

```bash
# MLX backend (Mac Studio)
python scripts/run_orpo_training.py \
    --config configs/training_pipeline.yaml \
    --backend mlx

# PyTorch backend (Cloud GPU)
python scripts/run_orpo_training.py \
    --config configs/training_pipeline.yaml \
    --backend pytorch
```

## Configuration

### ORPO Training Parameters

```yaml
orpo:
  steps_per_topic: 100         # Training steps per topic
  learning_rate: 0.0003        # 3e-4 recommended
  batch_size: 4                # Batch size per device
  lambda_orpo: 0.1             # Weight for preference term (0.05-0.2)
  lora_rank: 8                 # LoRA rank
  lora_alpha: 16               # LoRA alpha (typically 2x rank)
  target_modules:              # Attention and projection layers
    - q_proj
    - v_proj
    - o_proj
    - up_proj
    - down_proj
```

### Lambda Parameter (ORPO balance)

The `lambda_orpo` parameter controls the balance between SFT loss and preference loss:
- **Lower values (0.05-0.1)**: More focus on supervised learning
- **Higher values (0.2-0.5)**: More focus on preference alignment

### Learning Rate

ORPO typically uses higher learning rates than DPO:
- **Recommended**: 3e-4 to 5e-4
- **Adjust down** if training is unstable
- **Adjust up** if convergence is slow

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

1. **ORPO is single-stage** - No separate SFT phase needed
2. **Start with default lambda** - lambda_orpo=0.1 works well for most cases
3. **Use gradient accumulation** - For larger effective batches
4. **Target all linear layers** - Better quality than attention-only
5. **Monitor both loss components** - Watch NLL loss and OR loss separately

## Troubleshooting

### Low Voice Scores

- Increase training steps (steps_per_topic)
- Check data quality and preference pair correctness
- Verify LoRA adapters are training properly
- Reduce lambda_orpo to focus more on SFT

### Training Instability

- Reduce learning rate (try 1e-4 instead of 3e-4)
- Increase gradient accumulation
- Check for data quality issues
- Reduce lambda_orpo slightly

### Out of Memory

- Reduce batch_size
- Reduce LoRA rank (try 4 or 6)
- Increase gradient accumulation
- Use quantization (QLoRA)
