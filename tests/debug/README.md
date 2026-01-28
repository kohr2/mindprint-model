# MLX Training State Diagnostic Tests

This directory contains diagnostic tests for investigating MLX backend training issues.

## Test: `test_mlx_training_state.py`

Comprehensive diagnostic test to investigate why MLX-trained models generate poor output (mostly `<|endoftext|>` tokens) despite successful training completion.

### Problem

After SFT training with MLX backend:
- Training completes successfully (loss decreases)
- Base model generates properly (voice score ~0.12)
- Trained model generates mostly `<|endoftext|>` tokens
- Voice scores drop to 0.00 after training
- Prompt formats are correct (both use Qwen `<|im_start|>` format)

### Test Phases

1. **Baseline Verification**
   - Load base Qwen2.5-7B-Instruct model
   - Capture parameter snapshot
   - Generate baseline answer
   - Evaluate baseline voice score

2. **Training Simulation**
   - Create minimal training data (2 samples)
   - Run 1 epoch of SFT training
   - Force MLX evaluation with multiple methods
   - Verify loss decreases

3. **Parameter Verification**
   - Compare parameters before/after training
   - Check if parameters actually changed
   - Verify LoRA parameters exist and are non-zero
   - Report parameter statistics

4. **Generation After Training**
   - Generate answer with trained model
   - Compare output quality with baseline
   - Check for `<|endoftext|>` token spam
   - Evaluate trained voice score

5. **Model State Inspection**
   - Inspect model's internal state
   - Check if LoRA layers are properly attached
   - Test if explicit `mx.eval()` helps
   - Verify model object identity

### Running the Test

```bash
# From project root
cd mindprint-model
python3 tests/debug/test_mlx_training_state.py
```

Or via SSH on Mac Studio:

```bash
ssh memetica-studio@100.87.103.70
cd ~/mindprint-model
export PATH=$PATH:/Users/memetica-studio/Library/Python/3.9/bin
python3 tests/debug/test_mlx_training_state.py
```

### Expected Output

The test will output detailed diagnostics including:

- Parameter change statistics
- LoRA parameter verification
- Generation quality comparison
- Voice score changes
- Diagnosis summary with specific issues identified

### Diagnosis Outcomes

#### If parameters didn't change:
- Training loop has a bug
- Optimizer not updating weights
- Gradients not flowing properly

#### If parameters changed but generation is bad:
- Model state not being used during generation
- Need to force evaluation before generation
- Generation function using cached/stale model

#### If LoRA parameters missing:
- LoRA layers not properly attached
- Training on wrong parameters
- Adapter management issue

### Key Files Referenced

- `src/backends/mlx/mlx_sft_trainer.py` - Training loop (line 214: `mx.eval()` call)
- `src/backends/mlx/mlx_model.py` - Generation method (line 84: `mlx_generate()`)
- `src/evaluation/voice_evaluator.py` - Evaluation logic (line 323: `_generate_answers()`)

### Success Criteria

The test should clearly identify:
1. Whether parameters are actually changing during training
2. Whether the changed parameters are being used during generation
3. What specific MLX evaluation/state management step is missing
4. A concrete fix to make trained models generate properly

---

## Test Results: MLX LoRA Training Investigation

### Initial Diagnostic Test Run (Jan 28, 2026)

**Before Fix**:
```
Parameter Statistics:
  Total parameters: 2
  Changed parameters: 2
  Unchanged parameters: 0

LoRA Parameter Check:
  ⚠ WARNING: No LoRA parameters found!

Generation After Training:
  Baseline <|endoftext|> count: 0
  Trained <|endoftext|> count: 512
  
Voice Scores:
  Baseline: 0.1167
  Trained: 0.0000
```

**Diagnosis**: No LoRA adapters attached, training full model weights (7B parameters)

**Root Cause**: `MLXAdapterManager.add_adapter()` was a stub that only set `_has_adapter = True` without actually converting Linear layers to LoRA.

### After Fix (Expected Results)

**With LoRA Adapters**:
```
Parameter Statistics:
  Total parameters: 150+
  Changed parameters: 150+
  Unchanged parameters: 0

LoRA Parameter Check:
  ✓ LoRA parameters found: 150 groups
  Sample: model.layers.0.self_attn.q_proj.lora_a

Generation After Training:
  Baseline <|endoftext|> count: 0
  Trained <|endoftext|> count: 0-5
  
Voice Scores:
  Baseline: 0.1167
  Trained: 0.70+ (maintained or improved)
```

### Running Verification Test

After implementing the fix, run:
```bash
python3 tests/debug/test_mlx_training_state.py
```

**Look for these indicators**:
- "✓ LoRA adapter added: has_adapter=True"
- "✓ LoRA parameters found: X groups" (X > 100)
- "✓ Parameters changed: X/Y" (X > 100)
- Voice score > 0.10 after training
- `<|endoftext|>` count < 10

**Red flags** (indicates fix not working):
- "⚠ WARNING: No LoRA parameters found!"
- "Total parameters: 2" (should be 100+)
- "Trained <|endoftext|> count: 512"
- Voice score: 0.0000

---

## Related Documentation

- **Investigation**: `docs/mlx/MLX_LORA_TRAINING_ISSUE.md` - Complete timeline and resolution
- **Architecture**: `docs/mlx/MLX_LORA_ARCHITECTURE.md` - Technical details
- **Troubleshooting**: `docs/mlx/MLX_BACKEND_TROUBLESHOOTING.md` - Common issues and solutions
