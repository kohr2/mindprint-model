# Curriculum Training Plan

## Overview

This plan covers training the Bob Loukas curriculum dataset (10 topics) using the ORPO training pipeline with early stopping and separate output directories.

**Training Details:**
- **Dataset**: Curriculum (textbook) - 10 topics, 14 chapters, 4 units
- **Model**: Qwen/Qwen2.5-7B-Instruct
- **Backend**: MLX (Mac Studio M2 Ultra)
- **Expected Duration**: ~21.7 hours (10 topics × ~130 min/topic)
- **Output Location**: `./output/curriculum/` (separate from transcript model)

## Pre-Training Checklist

### 1. System Requirements

- [ ] **Mac Studio M2 Ultra** is available and accessible
- [ ] **64GB RAM** available (check: `sysctl hw.memsize`)
- [ ] **Disk Space**: At least 50GB free (check: `df -h`)
- [ ] **Python 3.8+** installed (check: `python3 --version`)
- [ ] **MLX** backend dependencies installed
- [ ] **Network**: Stable connection for model downloads (if needed)

### 2. Code Verification

- [ ] **Latest code pulled**: `git pull origin main`
- [ ] **Config file exists**: `configs/training_pipeline_curriculum.yaml`
- [ ] **Training script exists**: `scripts/run_orpo_training.py`
- [ ] **No syntax errors**: Run `python3 -m py_compile scripts/run_orpo_training.py`

### 3. Data Verification

- [ ] **Curriculum data exists**: `data/bob_loukas/textbook/`
- [ ] **Data files present**:
  - `preference_data.jsonl` (~223KB)
  - `curriculum.yaml` (if exists)
- [ ] **Data integrity**: Verify files are not corrupted

### 4. Output Directory Isolation

- [ ] **Transcript model safe**: Verify transcript outputs exist and won't be overwritten
  ```bash
  ls -la output/transcripts_*/
  stat output/transcripts_*/adapters/ | grep Modify
  ```
- [ ] **Curriculum directories ready**:
  ```bash
  mkdir -p output/curriculum
  mkdir -p checkpoints/curriculum
  mkdir -p logs
  ```
- [ ] **Verify separation**: Confirm paths are different
  - Transcript: `./output/` vs Curriculum: `./output/curriculum/`
  - Transcript: `./checkpoints/` vs Curriculum: `./checkpoints/curriculum/`

### 5. Configuration Review

- [ ] **Config loaded successfully**: Test config loading
  ```bash
  python3 -c "from scripts.run_orpo_training import load_config; \
    config, _ = load_config('configs/training_pipeline_curriculum.yaml'); \
    print(f'Output: {config.output_dir}, Checkpoint: {config.checkpoint_dir}')"
  ```
- [ ] **Early stopping settings**: Review and adjust if needed
  - `early_stopping_enabled: true`
  - `early_stopping_min_topics: 5`
  - `early_stopping_cv_threshold: 15.0`
  - `early_stopping_patience: 3`
- [ ] **Max topics**: Set to `null` (train all 10 topics)

### 6. Resource Check

- [ ] **No conflicting training**: Check for running training processes
  ```bash
  ps aux | grep run_orpo_training
  ```
- [ ] **Memory available**: Check current memory usage
  ```bash
  vm_stat | head -10
  ```
- [ ] **GPU/MPS available**: Verify MLX can access Apple Silicon
  ```bash
  python3 -c "import mlx.core as mx; print('MLX available:', mx.metal.is_available())"
  ```

## Training Execution

### Step 1: Start Training

**On Mac Studio**, run:

```bash
cd ~/mindprint-model

# Option 1: Direct execution (recommended for testing)
python3 scripts/run_orpo_training.py \
    --config configs/training_pipeline_curriculum.yaml \
    --backend mlx

# Option 2: Background execution with nohup
nohup python3 scripts/run_orpo_training.py \
    --config configs/training_pipeline_curriculum.yaml \
    --backend mlx \
    > logs/training_curriculum_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Save PID for monitoring
echo $! > logs/training_curriculum.pid
```

### Step 2: Verify Training Started

- [ ] **Process running**: `ps aux | grep run_orpo_training`
- [ ] **Log file created**: `ls -lh logs/training_curriculum.log`
- [ ] **Initial output**: Check first 50 lines of log
  ```bash
  tail -50 logs/training_curriculum.log
  ```
- [ ] **No errors**: Verify no immediate failures

### Step 3: Monitor Training Progress

**Key Metrics to Track:**

1. **Topic Progress**
   - Current topic number (1-10)
   - Topics completed
   - Topics remaining

2. **Loss Values**
   - Current ORPO loss
   - Loss trend (decreasing/stabilizing)
   - CV (coefficient of variation) for early stopping

3. **Training Speed**
   - Time per topic
   - Estimated completion time (ETA)
   - Total elapsed time

4. **System Resources**
   - Memory usage
   - CPU/GPU utilization
   - Temperature (if available)

**Monitoring Commands:**

```bash
# Watch log file in real-time
tail -f logs/training_curriculum.log

# Extract key metrics
tail -f logs/training_curriculum.log | grep -E "(Topic|Loss|ETA|CV|Early stopping)"

# Check process status
ps aux | grep run_orpo_training | grep -v grep

# Monitor system resources
top -pid $(cat logs/training_curriculum.pid)

# Check output directory growth
watch -n 60 'du -sh output/curriculum/ checkpoints/curriculum/'
```

**Expected Log Patterns:**

```
INFO - Training topic: unit1_chapter1_topic1
INFO - ORPO training started
INFO - Loss: 2500.0
INFO - Topic completed: unit1_chapter1_topic1
INFO - Loss: 2450.0
...
INFO - Early stopping triggered: Loss CV (12.5%) < threshold (15.0%)
```

## Monitoring Schedule

### During Training

**First Hour:**
- Check every 15 minutes
- Verify topic 1 completes successfully
- Confirm loss is decreasing

**Hours 2-10:**
- Check every 30-60 minutes
- Monitor loss trend
- Watch for early stopping triggers

**Hours 10-22:**
- Check every 1-2 hours
- Monitor final topics
- Prepare for completion

### Key Checkpoints

1. **Topic 1 Complete** (~2 hours)
   - Verify output directory created
   - Check loss value
   - Confirm no errors

2. **Topic 5 Complete** (~10 hours)
   - Early stopping threshold reached
   - Check if CV triggers early stop
   - Review loss progression

3. **Topic 10 Complete** (~21.7 hours)
   - Training complete
   - Verify all outputs
   - Check final loss

## Post-Training Verification

### 1. Training Completion

- [ ] **All topics trained**: Verify 10 topics completed
  ```bash
  grep "Topic completed" logs/training_curriculum.log | wc -l
  ```
- [ ] **No critical errors**: Review log for failures
  ```bash
  grep -i "error\|failed\|exception" logs/training_curriculum.log
  ```
- [ ] **Final loss recorded**: Extract final loss value
  ```bash
  grep "Final loss\|orpo_loss" logs/training_curriculum.log | tail -10
  ```

### 2. Output Verification

- [ ] **Output directory exists**: `ls -la output/curriculum/`
- [ ] **Timestamped subdirectory created**: Should see `textbook_YYYYMMDD_HHMMSS/`
- [ ] **Adapters saved**: `ls -lh output/curriculum/*/adapters/`
- [ ] **Merged model exists** (if merge enabled): `ls -lh output/curriculum/*/merged/`
- [ ] **Checkpoints saved**: `ls -lh checkpoints/curriculum/`

### 3. Model Quality Checks

- [ ] **Loss progression**: Review loss over topics
  ```bash
  grep "orpo_loss" logs/training_curriculum.log | tail -10
  ```
- [ ] **Topic pass rate**: Check how many topics passed thresholds
  ```bash
  grep "Topic.*passed\|Topic.*failed" logs/training_curriculum.log
  ```
- [ ] **Early stopping behavior**: Verify if early stopping triggered
  ```bash
  grep "Early stopping" logs/training_curriculum.log
  ```

### 4. Isolation Verification

- [ ] **Transcript model untouched**: Verify modification times unchanged
  ```bash
  stat output/transcripts_*/adapters/ | grep Modify
  ```
- [ ] **Separate directories**: Confirm no overlap
  ```bash
  diff -r output/transcripts_*/ output/curriculum/ 2>&1 | head -5
  ```

## Expected Outcomes

### Training Metrics

**Loss Progression** (estimated):
- Topic 1: ~2500 → ~2000 (high improvement)
- Topic 5: ~500 → ~300 (moderate improvement)
- Topic 10: ~200 → ~180 (diminishing returns)

**Early Stopping**:
- May trigger after topic 5-7 if loss stabilizes (CV < 15%)
- If loss continues improving, will train all 10 topics

**Training Time**:
- Per topic: ~130 minutes average
- Total: ~21.7 hours (if all 10 topics)
- Early stop: ~10-15 hours (if triggered)

### Output Structure

```
output/curriculum/
└── textbook_YYYYMMDD_HHMMSS/
    ├── adapters/
    │   ├── unit1_chapter1_topic1/
    │   ├── unit1_chapter1_topic2/
    │   └── ...
    └── merged/
        └── (merged model if merge_after_unit enabled)

checkpoints/curriculum/
└── latest.json

logs/
└── training_curriculum.log
```

## Troubleshooting

### Common Issues

**1. Training Fails to Start**
- **Symptom**: Process exits immediately
- **Check**: Log file for errors
- **Fix**: Verify MLX backend installed, check config syntax

**2. Out of Memory**
- **Symptom**: Process killed, "Killed" in logs
- **Check**: `dmesg | grep -i killed`
- **Fix**: Reduce batch size, close other applications

**3. Loss Not Decreasing**
- **Symptom**: Loss plateaus or increases
- **Check**: Review learning rate, data quality
- **Fix**: Adjust learning rate, verify data preprocessing

**4. Early Stopping Triggers Too Early**
- **Symptom**: Training stops after 5 topics
- **Check**: CV calculation, loss values
- **Fix**: Increase `early_stopping_min_topics` or `early_stopping_cv_threshold`

**5. Output Directory Conflicts**
- **Symptom**: Files in wrong directory
- **Check**: Config paths, verify separation
- **Fix**: Review config file paths

### Recovery Procedures

**Resume from Checkpoint:**
```bash
python3 scripts/run_orpo_training.py \
    --config configs/training_pipeline_curriculum.yaml \
    --backend mlx \
    --resume checkpoints/curriculum/latest.json
```

**Restart Training:**
```bash
# Stop current training
pkill -f run_orpo_training

# Clear checkpoints (if needed)
rm -rf checkpoints/curriculum/*

# Restart training
python3 scripts/run_orpo_training.py \
    --config configs/training_pipeline_curriculum.yaml \
    --backend mlx
```

## Post-Training Steps

### 1. Model Evaluation

- [ ] Run evaluation script (if available)
- [ ] Test model on sample queries
- [ ] Compare with transcript model performance

### 2. Model Deployment

- [ ] Merge adapters (if not already merged)
- [ ] Prepare for serving (vLLM/MLX-LM)
- [ ] Update model registry (if using dynamic selection)

### 3. Documentation

- [ ] Record final loss values
- [ ] Document training duration
- [ ] Note any early stopping events
- [ ] Update model comparison notes

### 4. Cleanup (Optional)

- [ ] Archive old checkpoints
- [ ] Compress log files
- [ ] Update training history

## Success Criteria

✅ **Training Complete**: All 10 topics trained successfully  
✅ **Loss Improvement**: Final loss < 500 (or significant reduction from initial)  
✅ **Output Isolation**: Transcript model untouched  
✅ **No Critical Errors**: Training completed without failures  
✅ **Model Usable**: Output model can be loaded and tested  

## Next Steps After Training

1. **Model Comparison**: Compare curriculum vs transcript model performance
2. **Model Merging**: Consider merging both models (if beneficial)
3. **Production Deployment**: Deploy curriculum model for testing
4. **Iteration**: Plan next training run based on results

## Quick Reference

**Start Training:**
```bash
python3 scripts/run_orpo_training.py \
    --config configs/training_pipeline_curriculum.yaml \
    --backend mlx
```

**Monitor Training:**
```bash
tail -f logs/training_curriculum.log | grep -E "(Topic|Loss|ETA)"
```

**Check Progress:**
```bash
ps aux | grep run_orpo_training
ls -lh output/curriculum/
```

**Stop Training:**
```bash
pkill -f run_orpo_training
```

---

**Last Updated**: 2026-02-05  
**Plan Version**: 1.0  
**Training Config**: `configs/training_pipeline_curriculum.yaml`
