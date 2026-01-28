# Adapter Stacking Debug Report

## Problem Summary

DPO training was failing with all 28 topics showing voice_score=0.00, indicating complete model corruption.

## Root Causes Identified

### Issue #1: SFT → DPO Adapter Stacking (FIXED ✅)
**Problem**: SFT creates rank-8 LoRA adapter, then DPO tries to add rank-1 LoRA on top without unloading.

**Fix Applied** (commit fa3b302):
```python
# In dpo_pipeline.py after SFT (lines 385-398)
sft_model = sft_trainer.get_model()
logger.info("Unloading SFT adapter before DPO training")
if hasattr(sft_model, 'merge_and_unload'):
    self.model = sft_model.merge_and_unload()
    logger.info("SFT adapter merged and unloaded successfully")
mps_empty_cache()
```

**Result**: ✅ SFT adapter now properly unloaded before DPO

---

### Issue #2: DPO → Next Topic Adapter Stacking (FIXED ✅)
**Problem**: After DPO completes, rank-1 DPO adapter stays attached. Next topic's SFT tries to add rank-8 on top.

**Fix Applied** (commit b834875):
```python
# In dpo_pipeline.py after DPO (lines 433-443)
if dpo_result.success:
    progress.status = TopicStatus.DPO_COMPLETE
    progress.dpo_loss = dpo_result.final_loss

    # Get DPO model and unload adapter for next topic
    dpo_model = dpo_trainer.get_model()
    logger.info("Unloading DPO adapter after training")
    if hasattr(dpo_model, 'merge_and_unload'):
        self.model = dpo_model.merge_and_unload()
        logger.info("DPO adapter merged and unloaded successfully")

    mps_empty_cache()
```

**Result**: ✅ DPO adapter now properly unloaded after training

---

## Current Status (Training Run: 2026-01-10 09:24)

### ✅ Adapter Unloading Working:
- Logs confirm: "Unloading SFT adapter before DPO training" ✅
- Logs confirm: "SFT adapter merged and unloaded successfully" ✅
- Logs confirm: "Unloading DPO adapter after training" ✅
- Logs confirm: "DPO adapter merged and unloaded successfully" ✅
- No more "trying to modify a model with PEFT for a second time" errors ✅

### ❌ New Problem: Voice Scores Still 0.00

**Observation**:
```
Topic unit-01/chapter-01:
- Before DPO: accuracy=0.77, voice=0.47 ✅
- DPO training: Completes 100 steps, loss decreases
- After DPO evaluation: accuracy=0.02, voice=0.00 ❌
```

All subsequent topics also show voice=0.00.

**Warning Still Present**:
```
UserWarning: Already found a `peft_config` attribute in the model.
This will lead to having multiple adapters in the model.
```

This warning appears even after we unload adapters.

---

## Remaining Issues to Investigate

### Hypothesis 1: PEFT Config Metadata Persistence
The `peft_config` attribute remains even after `merge_and_unload()`. While the model type changes from `PeftModel` → base model, the metadata persists and may confuse subsequent adapter additions.

**Evidence**:
- Our unit tests show `peft_config` attribute exists after merge_and_unload()
- Warning appears: "Already found a `peft_config` attribute"
- Defensive check in dpo_trainer.py uses `isinstance(PeftModel)` not `hasattr(peft_config)`

**Potential Solution**:
Explicitly delete the `peft_config` attribute after merge_and_unload():
```python
merged_model = sft_model.merge_and_unload()
if hasattr(merged_model, 'peft_config'):
    delattr(merged_model, 'peft_config')
```

### Hypothesis 2: DPO Training Itself Corrupts Model
DPO training completes, loss decreases, but model can't generate text afterward.

**Evidence**:
- DPO training metrics show improvement (loss 0.69 → 0.67)
- But post-DPO evaluation shows voice=0.00
- This happens even though we unload the adapter before evaluation

**Possible Causes**:
- DPO training on MPS has bugs?
- Rank-1 LoRA is too low and breaks model?
- Evaluation happens on model with peft_config metadata confusing generation?

### Hypothesis 3: Evaluation After DPO is Broken
The evaluation function might be failing when run on a model that has been through DPO.

**Evidence**:
- Evaluation works fine after SFT (voice=0.47)
- Evaluation fails after DPO (voice=0.00)
- Same evaluation code used in both cases

**Potential Issues**:
- Model state not properly reset?
- Generation parameters affected by peft_config metadata?
- MPS cache issues?

---

## Test Results

### Unit Tests (All Passing ✅)
Located in: `tests/unit/test_adapter_management.py`

1. ✅ `test_sft_adapter_unload_before_dpo` - Verifies merge_and_unload works
2. ✅ `test_adapter_stacking_detection` - Confirms PeftModel detection
3. ✅ `test_dpo_trainer_rejects_peft_model` - Tests defensive check
4. ✅ `test_clean_model_accepted_by_dpo` - Verifies clean models work
5. ✅ `test_merged_model_accepted_by_dpo` - Confirms merged models work

### Integration Test (Partial Success ⚠️)
Training started: 2026-01-10 09:24 on Mac Studio

**Progress**:
- Topic 1: SFT complete, DPO complete, adapters unloaded properly
- Topics 2, 3, 4: Continuing to train
- No crashes, training stable

**Issues**:
- All evaluations after DPO show voice=0.00
- Expected: voice scores to improve with DPO
- Actual: voice scores drop to 0

---

## Files Modified

### Primary Changes:
1. `src/training/dpo_pipeline.py` (lines 385-398, 433-443)
   - Added adapter unloading after SFT
   - Added adapter unloading after DPO

2. `src/training/dpo_trainer.py` (lines 130-142)
   - Added defensive check for existing PeftModel instances

3. `tests/unit/test_adapter_management.py` (NEW)
   - Comprehensive adapter management tests

### Supporting Files:
4. `scripts/test_single_topic.sh` (NEW)
   - Automated single-topic testing script

5. `TESTING_INSTRUCTIONS.md` (NEW)
   - Complete testing guide

---

## Next Steps

### Immediate (Currently Running):
- Let full training run complete to gather more data
- Check if ANY topics pass with current fix
- Examine logs for patterns in failures

### Short Term:
1. **Test Hypothesis 1**: Try explicitly deleting `peft_config` attribute
2. **Test Hypothesis 2**: Try DPO with higher rank (r=4 instead of r=1)
3. **Test Hypothesis 3**: Debug evaluation function after DPO

### Medium Term:
- Consider alternative approach: Skip DPO if it consistently corrupts model
- Or: Use same adapter for both SFT and DPO (Option 2 from original plan)
- Or: Investigate why DPO works in other projects but not here

---

## Commands to Monitor Training

```bash
# SSH to Mac Studio
ssh memetica-studio@100.87.103.70

# Check latest log
cd /Users/memetica-studio/dev/mindprint-model
tail -f training_output.log

# Check for adapter unloading
grep -E "(Unloading|merged)" training_output.log

# Check voice scores
grep "voice=" training_output.log | tail -20

# Check for warnings
grep "WARNING\|second time" training_output.log
```

---

## References

- Original training run (broken): `logs/dpo_training_20260109_124009.log`
- First fix attempt (partial): `training_output_broken.log`
- Current run (testing complete fix): `training_output.log`

---

## Conclusion

**Progress**: ✅ Fixed adapter stacking between training phases
**Remaining Issue**: ❌ Voice scores still 0.00 after DPO
**Status**: Training running, investigating root cause

The adapter management is now correct (confirmed by logs), but there's still an issue with model quality after DPO training. Further investigation needed.

---

## Update: MLX Backend LoRA Training Issue (Jan 28, 2026)

### New Discovery: MLX Not Using LoRA At All

While investigating continued `voice_score=0.00` issues with MLX backend, discovered that the MLX backend was never actually using LoRA adapters:

**Problem**: 
- `MLXAdapterManager.add_adapter()` was a stub implementation
- Only set `_has_adapter = True` flag without converting layers
- Training modified full model weights (7B parameters)
- Caused model corruption similar to PyTorch MPS issues
- Generated 512 `<|endoftext|>` tokens after training

**Diagnostic Test Results**:
```
Parameter Statistics:
  Total parameters: 2 (model, lm_head)
  LoRA parameters found: 0
  
Generation After Training:
  <|endoftext|> count: 512 (vs 0 baseline)
  Voice score: 0.0000 (vs 0.1167 baseline)
```

**Solution Implemented**:
- Proper LoRA layer conversion using `mlx_lm.tuner.lora.LoRALinear`
- Recursive traversal and conversion of Linear layers
- Gradient filtering to train only LoRA parameters (~8M vs 7B)
- Trains only LoRA parameters, base model remains frozen
- Prevents base model corruption

**Impact**:
This explains why MLX backend had similar issues to PyTorch MPS despite being a different framework. Both were corrupting the base model, just for different reasons:

| Backend | Issue | Corruption Point | Solution |
|---------|-------|------------------|----------|
| PyTorch MPS | Non-contiguous tensor bug | During `merge_and_unload()` | Use MLX backend |
| MLX (before fix) | No LoRA adapters | During training (full model) | Implement LoRA (fixed) |
| MLX (after fix) | None | N/A | LoRA protects base model |

**Documentation**:
- Complete investigation: `docs/mlx/MLX_LORA_TRAINING_ISSUE.md`
- Architecture details: `docs/mlx/MLX_LORA_ARCHITECTURE.md`
- Troubleshooting: `docs/mlx/MLX_BACKEND_TROUBLESHOOTING.md`

**Status**: ✅ Fixed and ready for testing

---

**Last Updated**: 2026-01-28 14:30
**Training Status**: MLX LoRA fix implemented
**Next Check**: Run full training pipeline with LoRA adapters
