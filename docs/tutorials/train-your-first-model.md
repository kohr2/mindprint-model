# Train Your First Model

## Step 1: Prepare Data

```bash
python scripts/run_data_prep.py \
    --data-dir ./data/bob_loukas/transcripts \
    --output-dir ./data/bob_loukas/transcripts
```

This creates:
- `preference_data.jsonl` — Preference pairs for ORPO training (`{prompt, chosen, rejected}`)

## Step 2: Configure Training

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

## Step 3: Start Training

### On Mac Studio

```bash
ssh memetica-studio@100.91.229.17
cd ~/mindprint-model
./scripts/local_train.sh
```

### Monitor Progress

```bash
./scripts/local_monitor.sh --follow
```

## Step 4: Evaluate

```bash
./scripts/local_evaluate.sh
```

## Step 5: Review Results

```bash
cat eval_results/report.md
```

Check:
- Voice fidelity scores (target: >0.75)
- Accuracy scores (target: >0.85)
- Pass rate (target: >80%)

## Next Steps

- [Training Guide](../user-guide/training.md)
- [Loss Functions](../concepts/loss-functions.md)
- [Evaluation Guide](../TRANSCRIPTS_EVALUATION.md)
