# Qwen2.5-72B Model Guide

## Overview

| Spec | Value |
|------|-------|
| Parameters | 72.7B (70.0B non-embedding) |
| Layers | 80 |
| Architecture | Dense, Grouped Query Attention |
| Context | 131,072 tokens (YaRN scaling) |
| Prompt Format | ChatML |
| HuggingFace | `Qwen/Qwen2.5-72B-Instruct` |

## Hardware Requirements

### Mac Studio M2 Ultra (64GB) — Recommended

- **Quantization**: INT4 (NormalFloat 4-bit)
- **Backend**: MLX
- **Batch Size**: 1, Gradient Accumulation: 4

```
Memory Budget:
  Base Model (INT4):     36 GB
  LoRA Adapters:          2 GB
  Optimizer States:       3 GB
  Activations (batch=1):  4 GB
  KV Cache (4K context):  2 GB
  ─────────────────────────────
  Total:                 47 GB (within 48 GB limit)
```

**Training Time**: ~8-12 hours per topic (ORPO single-stage)

### Cloud GPU

| Config | Quantization | Batch | Time/Topic | Cost |
|--------|-------------|-------|------------|------|
| 8x H100 80GB | INT4/BF16 | 2 | 3-5h | ~$98/h |
| 4x A100 80GB | INT4 | 1 | 6-8h | ~$40/h |

## Layer Zone Mapping

### Lexicon Zone (Layers 0-19) — 25%

**Purpose**: Bob Loukas's terminology embedding

**Target Modules**: `q_proj`

**Content**: "4-year cycle", "accumulation", "distribution", "cycle low/high", "parabolic advance"

### Reasoning Zone (Layers 20-59) — 50%

**Purpose**: Cycle theory and market analysis reasoning

**Target Modules**: `v_proj`, `up_proj`, `down_proj`

**Content**: Cycle theory, pattern recognition, multi-timeframe analysis, risk assessment

### Voice Zone (Layers 60-79) — 25%

**Purpose**: Bob's teaching style and confidence markers

**Target Modules**: `o_proj`, `up_proj`, `down_proj`

**Content**: Hedge language, confidence markers, teaching cadence

## LoRA Configuration

### Conservative (Recommended)

```yaml
lora:
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules: [q_proj, v_proj, o_proj, up_proj, down_proj]
```

~41M trainable parameters (0.06% of base model)

### Aggressive (If Underfitting)

```yaml
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

## ORPO Training Config

```yaml
orpo:
  learning_rate: 2e-4
  batch_size: 1
  gradient_accumulation: 4
  steps_per_topic: 100
  lambda_orpo: 0.1
  max_seq_length: 4096
  warmup_steps: 50
  weight_decay: 0.01
  scheduler: cosine
  gradient_checkpointing: true
```

## Usage

### MLX Backend (Recommended)

```python
from src.backends import create_backend
from src.training.orpo_pipeline import ORPOPipeline, PipelineConfig

backend = create_backend("mlx", device="auto", quantization="int4")
model = backend.load_model("Qwen/Qwen2.5-72B-Instruct")

config = PipelineConfig(
    orpo_steps_per_topic=100,
    orpo_learning_rate=2e-4,
    orpo_lambda=0.1,
)

pipeline = ORPOPipeline(model_config=get_model_config("qwen-72b"), config=config)
pipeline.train_curriculum(preference_data=preference_data)
```

## Model Comparison

| Metric | Qwen2.5-7B | Gemma-3-12B | **Qwen2.5-72B** |
|--------|------------|-------------|-----------------|
| VRAM (INT4) | 8GB | 12GB | **36GB** |
| Context | 131K | 128K | **131K** |
| ORPO Time/Topic | ~2h | ~3h | **~10h** |
| Mac Studio (FP16) | Yes | Yes | No |
| Mac Studio (INT4) | Yes | Yes | **Yes** |
| Voice Capacity | Fair | Good | **Excellent** |

## Troubleshooting

**OOM**: Reduce `max_seq_length` to 2048, reduce LoRA rank to 4, use nested quantization.

**Training instability**: Lower learning rate to 1e-4, increase warmup to 100 steps, use `max_grad_norm=0.3`.

**Poor voice fidelity**: Increase LoRA rank to 16, increase `steps_per_topic`, improve preference data quality.
