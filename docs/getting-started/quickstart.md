# Quickstart

## Train Your First Model

### 1. Prepare Data

```bash
python scripts/run_data_prep.py \
    --data-dir ./data/bob_loukas/transcripts \
    --output-dir ./data/bob_loukas/transcripts
```

### 2. Train with ORPO

```bash
python scripts/run_orpo_training.py \
    --config configs/training_pipeline.yaml \
    --backend mlx
```

### 3. Evaluate

```bash
python scripts/run_evaluation.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapter ./output/transcripts_*/adapters/... \
    --quiz-data ./data/bob_loukas/transcripts \
    --approach orpo
```

## Configuration

Edit `configs/training_pipeline.yaml`:

```yaml
orpo:
  steps_per_topic: 100
  learning_rate: 0.0003
  lambda_orpo: 0.1
  lora_rank: 8
  lora_alpha: 16
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - up_proj
    - down_proj
```

## Next Steps

- [Training Guide](../user-guide/training.md)
- [Loss Functions](../concepts/loss-functions.md)
- [Architecture](../concepts/architecture.md)
