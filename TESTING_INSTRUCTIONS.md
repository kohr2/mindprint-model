# Testing Instructions: Adapter Stacking Fix

## Quick Start on Mac Studio

```bash
# 1. SSH to Mac Studio
ssh memetica-studio@100.87.103.70

# 2. Navigate to project
cd /Users/memetica-studio/dev/mindprint-model

# 3. Pull latest changes (includes the fix)
git pull origin dpo

# 4. Run single-topic test (automated)
./scripts/test_single_topic.sh
```

The script will:
- Pull latest changes
- Clean previous test outputs
- Run training on one topic (unit-01/chapter-01)
- Validate the fix worked
- Show key log excerpts

**Expected runtime**: 3-5 minutes for single topic

---

## What the Fix Does

### Problem (Before):
- SFT creates rank-8 LoRA adapter
- DPO tries to add rank-1 LoRA on top
- Adapters stack → model corruption
- Result: accuracy=0.02, voice=0.00 (all topics failed)

### Solution (After):
1. After SFT completes, call `merge_and_unload()` on model
2. Merges SFT adapter weights into base model
3. Returns clean model (not a PeftModel)
4. DPO can then apply its rank-1 adapter cleanly

---

## Success Indicators

### ✅ In Logs (look for these):
```
INFO - Unloading SFT adapter before DPO training
INFO - SFT adapter merged and unloaded successfully
```

### ✅ No Warnings (should NOT see):
```
UserWarning: You are trying to modify a model with PEFT for a second time
```

### ✅ Non-Zero Scores After DPO:
```
INFO - Re-evaluate after DPO
INFO - Topic unit-01/chapter-01/topic-01: accuracy=0.XX, voice=0.XX
```
- Accuracy should be > 0.70
- Voice should be > 0.00 (ideally 0.40-0.80)

### ✅ Topic Passes:
```
INFO - Topic unit-01/chapter-01/topic-01: ✓ PASSED (combined_score=0.XX)
```

---

## Manual Testing (Alternative)

If you prefer to run manually:

```bash
cd /Users/memetica-studio/dev/mindprint-model
git pull origin dpo

# Run on single topic
python scripts/run_dpo_training.py \
  --config configs/training_pipeline.yaml \
  --data-dir data/bob_loukas/unit-01/chapter-01 \
  --output-dir ./output_test

# Monitor log in another terminal
tail -f logs/dpo_training_*.log
```

---

## After Single Topic Succeeds

If single-topic test passes, run full training:

```bash
# Full 28-topic training (~1.5-2 hours)
python scripts/run_dpo_training.py \
  --config configs/training_pipeline.yaml

# Monitor progress
tail -f logs/dpo_training_*.log
```

### Expected Results (Full Training):
- All 28 topics train without adapter stacking errors
- Topics meeting thresholds should pass (passed_topics > 0)
- Final report should show success

---

## Troubleshooting

### If single-topic test fails:

1. **Check log for errors**:
   ```bash
   tail -100 logs/dpo_training_*.log
   ```

2. **Verify changes applied**:
   ```bash
   git log -1 --oneline
   # Should show: "Fix adapter stacking issue..."

   grep -n "merge_and_unload" src/training/dpo_pipeline.py
   # Should show line 388
   ```

3. **Check Python environment**:
   ```bash
   python -c "import peft; print(peft.__version__)"
   # Should be >= 0.18.0
   ```

4. **Try with verbose logging**:
   ```bash
   python scripts/run_dpo_training.py \
     --config configs/training_pipeline.yaml \
     --data-dir data/bob_loukas/unit-01/chapter-01 \
     2>&1 | tee test_output.log
   ```

### If you see "PeftModel" error:

```
ValueError: Cannot apply DPO adapter: base model is a PeftModel
```

This means the defensive check caught a problem! The fix isn't working as expected. Check:
- Did `merge_and_unload()` get called? (should see log message)
- Is there an exception during merge? (check log for errors before this)

---

## File Changes Summary

Files modified in the fix:
1. `src/training/dpo_pipeline.py` - Lines 385-398 (adapter unloading)
2. `src/training/dpo_trainer.py` - Lines 130-142 (defensive check)
3. `tests/unit/test_adapter_management.py` - New test file
4. `scripts/test_single_topic.sh` - New automated test script

To verify changes:
```bash
git diff fa3b302^..fa3b302 --stat
```

---

## Next Steps

1. ✅ Run single-topic test
2. ✅ Verify logs show adapter unloading
3. ✅ Verify scores are non-zero after DPO
4. ✅ Verify topic passes
5. ⏭️ If successful, run full 28-topic training
6. ⏭️ Check final report: passed_topics, failed_topics, training time

---

## Contact

If issues persist after following these steps, share:
1. Latest log file: `logs/dpo_training_*.log`
2. Git commit hash: `git rev-parse HEAD`
3. Error messages or unexpected behavior

Good luck! 🚀
