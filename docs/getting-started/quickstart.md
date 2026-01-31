# Quickstart

## Train Your First Model

### 1. Prepare Data

```bash
python scripts/run_data_prep.py \
    --data-dir ./data/bob_loukas/transcripts \
    --output-dir ./data/bob_loukas/transcripts
```

### 2. Train with SimPO (Recommended)

```bash
python scripts/run_orpo_training.py \
    --config configs/training_pipeline.yaml \
    --backend mlx \
    --loss-type simpo
```

### 3. Evaluate

```bash
python scripts/run_evaluation.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapter ./output/transcripts_*/adapters/... \
    --quiz-data ./data/bob_loukas/transcripts \
    --approach simpo
```

## Configuration

Edit `configs/training_pipeline.yaml`:

```yaml
training:
  loss_type: simpo  # or "dpo", "orpo"
  warmup_ratio: 0.1
  gradient_accumulation_steps: 8

sft:
  lora_rank: 32
  lora_alpha: 64
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

## Next Steps

- [Training Guide](../user-guide/training.md)
- [Loss Functions](../concepts/loss-functions.md)
- [Architecture](../concepts/architecture.md)
