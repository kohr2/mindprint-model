# Prompt Format Fix - Qwen2.5 Evaluation Issue

## Problem

Voice scores were consistently 0.00 after SFT training on Mac Studio with MLX backend.

## Root Causes Identified

### 1. Evaluation Format Mismatch
The `VoiceFidelityEvaluator` was using hardcoded Gemma-3 chat format:
```
<start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
```

But Qwen2.5-7B-Instruct requires its own format:
```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
```

### 2. Training Format Mismatch
The MLX SFT trainer was also using Gemma-3 format for training, creating a double mismatch where:
- Training used Gemma format
- Evaluation used Qwen format
- Model expected Qwen format

## Fixes Applied

### Fix 1: Evaluation Format (`src/evaluation/voice_evaluator.py`)
1. **`_format_prompt()` method**: Now uses `tokenizer.apply_chat_template()` for models that support it
2. **Fallback**: Still supports Gemma-3 format for models without chat templates
3. **MLX output extraction**: Improved logic to extract only generated text

### Fix 2: Training Format (`src/backends/mlx/mlx_sft_trainer.py`)
1. **Training prompts**: Now use `tokenizer.apply_chat_template()` to match evaluation format
2. **Consistency**: Both training and evaluation now use the same Qwen format
3. **Fallback**: Supports Gemma-3 format for models without chat templates

## Test Results

### Before Fix
- Voice Score: 0.00
- Accuracy: Very low or negative
- Generated answers: Empty or malformed

### After Evaluation Fix Only (Base Model)
- Voice Score: **0.1167** ✅
- Accuracy: **0.7962** ✅
- Generated answers: Proper responses (1109 characters)

### After Both Fixes (Trained Model)
- Training: Uses correct Qwen format
- Evaluation: Uses correct Qwen format
- Status: Ready for full testing

## Key Learnings

1. **Chat templates matter**: Different models use completely different formats
2. **Training/eval consistency**: Both must use the same format
3. **Use tokenizer methods**: Always use `apply_chat_template()` instead of hardcoding
4. **Model-specific**: Qwen uses `<|im_start|>`/`<|im_end|>`, Gemma uses `<start_of_turn>`/`<end_of_turn>`

## Files Modified

- `src/evaluation/voice_evaluator.py` - Updated prompt formatting and MLX output extraction
- `src/backends/mlx/mlx_sft_trainer.py` - Updated training prompt formatting to use chat templates

## Next Steps

1. Run full training pipeline with both fixes applied
2. Monitor voice scores throughout training
3. Verify scores are non-zero and improving
4. Test DPO training if SFT succeeds

---

## Update: Deeper Issue Discovered (Jan 28, 2026)

While prompt format fixes resolved evaluation issues with the base model, they didn't fix the training corruption. Further investigation revealed a critical issue: **the MLX backend was not using LoRA adapters at all**.

### The Real Problem

The prompt format fixes allowed the base model to be evaluated correctly (voice score 0.1167), but after training, voice scores still dropped to 0.00. A comprehensive diagnostic test revealed:

- MLX backend trained full model weights (7B parameters)
- No LoRA adapters were attached before training
- Training corrupted base model weights
- Model generated 512 `<|endoftext|>` tokens after training

### Root Cause

The `MLXAdapterManager.add_adapter()` method was a stub that only set a flag without actually converting Linear layers to LoRA. This meant:
1. Prompt formats were correct ✅
2. But training modified all 7B parameters ❌
3. Base model became corrupted ❌
4. Generation collapsed to EOS tokens ❌

### Resolution

Implemented proper LoRA adapter support in MLX backend:
- Proper LoRA layer conversion using `mlx_lm.tuner.lora.LoRALinear`
- Gradient filtering to train only LoRA parameters
- Model path tracking for config lookup
- Comprehensive diagnostic tests

**See**: `docs/mlx/MLX_LORA_TRAINING_ISSUE.md` for complete investigation and resolution details.

**Status**: Both prompt format AND LoRA adapter issues now fixed ✅
