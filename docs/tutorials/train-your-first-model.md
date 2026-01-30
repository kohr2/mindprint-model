# Train Your First Model

This tutorial walks you through training a personalized language model.

## Step 1: Prepare Data

```bash
# Generate training data from transcripts
python scripts/run_data_prep.py \
    --data-dir ./data/bob_loukas/transcripts \
    --output-dir ./data/bob_loukas/transcripts
```

This creates:
- `sft_data.jsonl` - Supervised fine-tuning pairs
- `preference_data.jsonl` - Preference pairs for alignment

## Step 2: Configure Training

Edit `configs/training_pipeline.yaml`:

```yaml
training:
  loss_type: simpo  # Use SimPO for best results
  warmup_ratio: 0.1
  gradient_accumulation_steps: 8
  neftune_noise_alpha: 5.0

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

## Step 3: Start Training

### On Mac Studio

```bash
ssh memetica-studio@100.87.103.70
cd ~/mindprint-model
./scripts/local_train.sh
```

### Monitor Progress

```bash
./scripts/local_monitor.sh --follow
```

## Step 4: Evaluate

After training completes:

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
- [Evaluation Guide](../user-guide/evaluation.md)
