# Transcripts Training Evaluation Guide

## Quick Start

**Simplest approach - Run directly on Mac Studio:**

```bash
ssh memetica-studio@100.87.103.70
cd ~/mindprint-model
./scripts/local_evaluate.sh
```

This will:
1. Pull latest code from git
2. Auto-discover the trained adapter
3. Run evaluation on transcripts quiz data
4. Generate report in `./eval_results/`

## Evaluation Workflow

### Step 1: Check Training Completed

First, verify training finished successfully:

```bash
# On Mac Studio
cd ~/mindprint-model
cat checkpoints/latest.json | jq '.result.pass_rate'
```

This shows the pass rate from training-time evaluation.

### Step 2: Run Full Evaluation

**Option A: Use the convenience script (recommended)**

```bash
# On Mac Studio
cd ~/mindprint-model
./scripts/local_evaluate.sh
```

**Option B: Run evaluation manually**

```bash
# On Mac Studio
cd ~/mindprint-model
python3 scripts/run_evaluation.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapter ./output/transcripts_*/adapters/... \
    --quiz-data ./data/bob_loukas/transcripts \
    --approach dpo \
    --device mps \
    --trust-remote-code \
    --output ./eval_results
```

**Option C: Use post-training pipeline (merge + evaluate + export)**

```bash
# On Mac Studio
cd ~/mindprint-model
python3 scripts/run_post_training.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter ./output/transcripts_*/adapters/... \
    --quiz-data ./data/bob_loukas/transcripts \
    --output ./output/transcripts_evaluated \
    --approach dpo
```

### Step 3: Review Results

```bash
# View markdown report
cat eval_results/report.md

# Or view JSON (more detailed)
cat eval_results/report.json | jq

# Check specific metrics
cat eval_results/report.json | jq '.summary.pass_rate'
cat eval_results/report.json | jq '.summary.average_voice_score'
```

## Finding the Adapter Path

If you need to manually specify the adapter:

```bash
# Find all adapters
find output -name adapter_config.json -type f

# Find most recent adapter
find output -name adapter_config.json -type f -exec ls -lt {} + | head -1

# Check checkpoint for adapter location
cat checkpoints/latest.json | jq '.result.adapter_path'
```

Common adapter locations:
- `output/transcripts_YYYYMMDD_HHMMSS/adapters/unit-*/chapter-*/topic-*/sft_adapter/`
- `output/transcripts_YYYYMMDD_HHMMSS/merged_adapters/`
- `output/adapters/unit-*/chapter-*/topic-*/sft_adapter/`

## Evaluation Metrics

The evaluation measures:

1. **Accuracy**: Semantic similarity to reference answers (0.0 - 1.0)
2. **Voice Fidelity**: How well model matches Bob Loukas's style (0.0 - 1.0)
3. **Combined Score**: Weighted combination of both metrics

**Pass Thresholds**:
- Topic level: 90% accuracy, 0.75 voice score
- Chapter level: 85% accuracy, 0.75 voice score
- Unit level: 80% accuracy, 0.75 voice score
- Final assessment: 85% accuracy, 0.80 voice score

## Evaluation Levels

The evaluation runs hierarchically:

1. **Topic Evaluation**: Tests individual topic mastery
2. **Chapter Evaluation**: Tests cross-topic understanding
3. **Unit Evaluation**: Tests comprehensive knowledge
4. **Final Assessment**: Overall model performance

Each level must pass to proceed to the next.

## Troubleshooting

### Problem: "No adapter found"

**Solution**: Specify adapter path manually:
```bash
./scripts/local_evaluate.sh ./output/path/to/adapter
```

### Problem: "Quiz data not found"

**Check**:
```bash
ls -la data/bob_loukas/transcripts/quiz_data.json
```

If missing or empty, you may need to regenerate quiz data:
```bash
python3 scripts/run_data_prep.py --data-dir data/bob_loukas/transcripts
```

### Problem: "Evaluation is slow"

**Solutions**:
- Use `--device mps` for Apple Silicon acceleration
- Evaluation runs on CPU if MPS unavailable
- Large models may take 10-30 minutes

### Problem: "Voice score is 0.0"

**Possible causes**:
- Model wasn't trained with DPO
- Voice evaluator not working
- Missing voice markers in responses

**Check**:
```bash
cat eval_results/report.json | jq '.levels[].voice_score'
```

## Comparing Models

To compare transcripts model vs. other models:

```bash
# Evaluate transcripts model
python3 scripts/run_evaluation.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapter ./output/transcripts_adapter \
    --quiz-data ./data/bob_loukas/transcripts \
    --output ./eval_results/transcripts \
    --name "transcripts-model" \
    --approach dpo \
    --device mps \
    --trust-remote-code

# Evaluate textbook model
python3 scripts/run_evaluation.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapter ./output/textbook_adapter \
    --quiz-data ./data/bob_loukas/textbook \
    --output ./eval_results/textbook \
    --name "textbook-model" \
    --approach dpo \
    --device mps \
    --trust-remote-code

# Compare results
# Note: compare_models.py script has been removed. Use evaluation pipeline instead:
python3 scripts/run_evaluation.py \
    --results ./eval_results/transcripts/report.json \
    --results ./eval_results/textbook/report.json \
    --output ./eval_results/comparison.md
```

## Best Practices

1. **Run evaluation after each training run** - Don't wait until the end
2. **Save evaluation results** - Keep reports for comparison
3. **Check both accuracy and voice** - Both metrics matter
4. **Review failed topics** - Understand what didn't work
5. **Compare with baseline** - Know if you're improving

## Next Steps

After evaluation:

1. **If pass rate is low**: Review failed topics, check training data quality
2. **If voice score is low**: May need more DPO training or better preference data
3. **If accuracy is low**: May need more SFT training or better quiz questions
4. **If both are good**: Consider merging adapters and exporting for production

See also:
- [EVALUATION_QUICKSTART.md](./EVALUATION_QUICKSTART.md) - Quick reference
- [POST_TRAINING_EVALUATION.md](./POST_TRAINING_EVALUATION.md) - Full workflow
- [MODEL_EVALUATION.md](./MODEL_EVALUATION.md) - Detailed pipeline docs
