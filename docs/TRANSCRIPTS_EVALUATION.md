# Transcripts Evaluation Guide

## Quick Start

```bash
ssh memetica-studio@100.91.229.17
cd ~/mindprint-model
./scripts/local_evaluate.sh
```

This will:
1. Pull latest code
2. Auto-discover the trained adapter
3. Run evaluation on transcripts quiz data
4. Generate report in `./eval_results/`

## Evaluation Workflow

### 1. Check Training Completed

```bash
cd ~/mindprint-model
cat checkpoints/latest.json | jq '.result.pass_rate'
```

### 2. Run Evaluation

**Option A: Convenience script (recommended)**

```bash
./scripts/local_evaluate.sh
```

**Option B: Manual**

```bash
python3 scripts/run_evaluation.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapter ./output/transcripts_*/adapters/... \
    --quiz-data ./data/bob_loukas/transcripts \
    --approach orpo \
    --device mps \
    --trust-remote-code \
    --output ./eval_results
```

**Option C: Post-training pipeline (merge + evaluate + export)**

```bash
python3 scripts/run_post_training.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter ./output/transcripts_*/adapters/... \
    --quiz-data ./data/bob_loukas/transcripts \
    --output ./output/transcripts_evaluated \
    --approach orpo
```

### 3. Review Results

```bash
cat eval_results/report.md
cat eval_results/report.json | jq '.summary.pass_rate'
cat eval_results/report.json | jq '.summary.average_voice_score'
```

## Finding the Adapter Path

```bash
find output -name adapter_config.json -type f
cat checkpoints/latest.json | jq '.result.adapter_path'
```

Common locations:
- `output/transcripts_YYYYMMDD_HHMMSS/adapters/episode-*/`
- `output/transcripts_YYYYMMDD_HHMMSS/merged_adapters/`

## Evaluation Metrics

| Metric | Description | Scale |
|--------|-------------|-------|
| **Accuracy** | Semantic similarity to reference answers | 0.0–1.0 |
| **Voice Fidelity** | How well model matches Bob Loukas's style | 0.0–1.0 |
| **Combined Score** | Weighted combination | 0.0–1.0 |

**Pass Thresholds**:
- Topic: 90% accuracy, 0.75 voice
- Chapter: 85% accuracy, 0.75 voice
- Unit: 80% accuracy, 0.75 voice
- Final: 85% accuracy, 0.80 voice

## Troubleshooting

### No adapter found

Specify path manually: `./scripts/local_evaluate.sh ./output/path/to/adapter`

### Voice score is 0.0

- Verify LoRA adapters are attached and trained
- Check preference data quality
- Increase `steps_per_topic` or reduce `lambda_orpo` for more SFT focus

### Evaluation is slow

- Use `--device mps` for Apple Silicon acceleration
- Large models may take 10-30 minutes

## Next Steps

1. **Low pass rate**: Review failed topics, check preference data quality
2. **Low voice score**: Increase training steps, improve preference pairs
3. **Low accuracy**: Increase training steps, verify quiz question quality
4. **Both good**: Merge adapters and export for production
