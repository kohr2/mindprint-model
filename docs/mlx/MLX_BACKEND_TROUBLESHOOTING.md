# MLX Backend Troubleshooting Guide

Complete troubleshooting guide for MLX backend issues in the mindprint-model training pipeline.

---

## Table of Contents

1. [Common Issues](#common-issues)
2. [Diagnostic Procedures](#diagnostic-procedures)
3. [Performance Issues](#performance-issues)
4. [Error Messages](#error-messages)
5. [Verification Tools](#verification-tools)

---

## Common Issues

### Issue 1: Voice Scores 0.00 After Training

**Symptoms**:
- Training completes successfully
- Loss decreases normally
- Voice scores drop to 0.00 after SFT
- Model generates `<|endoftext|>` tokens repeatedly
- Accuracy near 0.00

**Diagnosis**:

Run the diagnostic test:
```bash
cd mindprint-model
python3 tests/debug/test_mlx_training_state.py
```

Look for these indicators:
```
LoRA Parameter Check:
  ⚠ WARNING: No LoRA parameters found!
  
Parameter Statistics:
  Total parameters: 2  # Should be 100+
  Changed parameters: 2  # Should be 100+
```

**Possible Causes**:

1. **LoRA adapters not attached** (Most Common)
   - `MLXAdapterManager.add_adapter()` not called
   - Stub implementation not replaced
   - Check: `model.has_adapter()` returns False

2. **Prompt format mismatch**
   - Training uses different format than evaluation
   - Check: Tokenizer's `chat_template` attribute
   - See: `docs/PROMPT_FORMAT_FIX.md`

3. **Model corruption during training**
   - Full model weights trained instead of LoRA
   - Base model parameters modified
   - Check: Parameter count (should be millions, not billions)

**Solutions**:

**For LoRA adapter issue**:
```python
# Verify adapter is added before training
from src.backends import create_backend

backend = create_backend('mlx')
model = backend.load_model('Qwen/Qwen2.5-7B-Instruct')

# This should automatically add LoRA adapter
trainer = backend.create_sft_trainer(model, {})

# Verify
print(f"Has adapter: {model.has_adapter()}")  # Should be True
print(f"Trainable params: {model.num_trainable_parameters:,}")  # Should be < 10M
```

**For prompt format issue**:
```python
# Ensure consistent format
tokenizer = model.tokenizer

# Check if chat template exists
if hasattr(tokenizer, 'apply_chat_template'):
    # Use it for both training and evaluation
    messages = [{"role": "user", "content": "Test"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print(f"Using chat template: {prompt[:100]}")
else:
    print("No chat template - using fallback format")
```

**For model corruption**:
- Update to latest version with LoRA fix
- See: `docs/MLX_LORA_TRAINING_ISSUE.md`

---

### Issue 2: Training Too Slow

**Symptoms**:
- Training takes significantly longer than expected
- GPU utilization low
- Memory usage higher than expected

**Diagnosis**:

Check training configuration:
```bash
# Monitor training
python3 scripts/monitor_training.sh

# Check memory usage
python3 -c "
import mlx.core as mx
print(f'GPU memory: {mx.metal.get_active_memory() / 1e9:.2f} GB')
print(f'Peak memory: {mx.metal.get_peak_memory() / 1e9:.2f} GB')
"
```

**Possible Causes**:

1. **Training full model instead of LoRA**
   - Check trainable parameter count
   - Should be < 10M, not 7B

2. **Batch size too small**
   - Default: 4
   - Try: 8 or 16 for Mac Studio M2 Ultra

3. **Sequence length too long**
   - Default: 2048
   - Try: 1024 or 512 for faster iteration

**Solutions**:

```yaml
# In configs/training_pipeline.yaml
sft:
  per_device_batch_size: 16  # Increase for faster training
  max_seq_length: 1024  # Reduce for faster training
```

---

### Issue 3: Out of Memory Errors

**Symptoms**:
- Training crashes with OOM error
- `mx.metal.get_active_memory()` shows high usage
- System becomes unresponsive

**Diagnosis**:

```python
# Check memory before training
import mlx.core as mx

print(f"Available memory: {mx.metal.get_cache_memory() / 1e9:.2f} GB")
print(f"Active memory: {mx.metal.get_active_memory() / 1e9:.2f} GB")
```

**Solutions**:

1. **Reduce batch size**:
```yaml
sft:
  per_device_batch_size: 2  # Reduce from 4
```

2. **Reduce sequence length**:
```yaml
sft:
  max_seq_length: 1024  # Reduce from 2048
```

3. **Clear cache between topics**:
```python
import mlx.core as mx
mx.metal.clear_cache()
```

4. **Use gradient checkpointing** (if available):
```yaml
sft:
  gradient_checkpointing: true
```

---

### Issue 4: Model Not Loading

**Symptoms**:
- `mlx_lm.load()` fails
- Model files not found
- Tokenizer errors

**Diagnosis**:

```bash
# Test model loading
python3 -c "
from mlx_lm import load

try:
    model, tokenizer = load('Qwen/Qwen2.5-7B-Instruct')
    print('✓ Model loaded successfully')
except Exception as e:
    print(f'✗ Error: {e}')
"
```

**Possible Causes**:

1. **Model not downloaded**
   - First load downloads from HuggingFace
   - Requires internet connection
   - May take 10-20 minutes

2. **Incorrect model name**
   - Check HuggingFace model ID
   - Case-sensitive

3. **MLX not installed**
   - Check: `pip list | grep mlx`

**Solutions**:

```bash
# Install MLX dependencies
pip install mlx mlx-lm

# Pre-download model
python3 -c "
from mlx_lm import load
model, tokenizer = load('Qwen/Qwen2.5-7B-Instruct')
print('Model downloaded')
"
```

---

## Diagnostic Procedures

### Procedure 1: Verify LoRA Adapter Setup

**Purpose**: Confirm LoRA adapters are properly attached and configured

**Steps**:

1. **Load model and check adapter**:
```python
from src.backends import create_backend

backend = create_backend('mlx')
model = backend.load_model('Qwen/Qwen2.5-7B-Instruct')

# Check initial state
print(f"1. Initial has_adapter: {model.has_adapter()}")
print(f"2. Initial trainable params: {model.num_trainable_parameters:,}")

# Create trainer (should add adapter)
trainer = backend.create_sft_trainer(model, {})

# Check after trainer creation
print(f"3. After trainer has_adapter: {model.has_adapter()}")
print(f"4. After trainer trainable params: {model.num_trainable_parameters:,}")
```

**Expected Output**:
```
1. Initial has_adapter: False
2. Initial trainable params: 7,000,000,000
3. After trainer has_adapter: True
4. After trainer trainable params: 8,000,000
```

**If Failed**:
- Trainable params still 7B → LoRA not added
- See: `docs/MLX_LORA_TRAINING_ISSUE.md`

2. **Check LoRA parameters**:
```python
# Inspect model parameters
mlx_model = model.get_underlying_model()
params_dict = mlx_model.parameters()

lora_params = [name for name in params_dict.keys() if 'lora' in name.lower()]
print(f"LoRA parameters found: {len(lora_params)}")
print(f"Sample LoRA params: {lora_params[:5]}")
```

**Expected Output**:
```
LoRA parameters found: 150
Sample LoRA params: ['model.layers.0.self_attn.q_proj.lora_a', ...]
```

### Procedure 2: Verify Prompt Format

**Purpose**: Ensure training and evaluation use same format

**Steps**:

1. **Check tokenizer chat template**:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')

# Check for chat template
if hasattr(tokenizer, 'apply_chat_template'):
    messages = [
        {"role": "user", "content": "What is a cycle low?"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("Chat template format:")
    print(prompt)
else:
    print("No chat template available")
```

2. **Compare training vs evaluation format**:
```python
# Training format (from mlx_sft_trainer.py)
training_messages = [
    {"role": "user", "content": "Question"},
    {"role": "assistant", "content": "Answer"}
]
training_format = tokenizer.apply_chat_template(training_messages, tokenize=False, add_generation_prompt=False)

# Evaluation format (from voice_evaluator.py)
eval_messages = [{"role": "user", "content": "Question"}]
eval_format = tokenizer.apply_chat_template(eval_messages, tokenize=False, add_generation_prompt=True)

print("Training format:", training_format[:100])
print("Evaluation format:", eval_format[:100])
```

### Procedure 3: Run Diagnostic Test

**Purpose**: Comprehensive diagnosis of training issues

**Command**:
```bash
python3 tests/debug/test_mlx_training_state.py
```

**What It Tests**:
1. Baseline model generation
2. Parameter changes during training
3. LoRA parameter presence
4. Post-training generation quality
5. Model state inspection

**Interpreting Results**:

**Good Output**:
```
✓ LoRA adapter added: has_adapter=True
✓ LoRA parameters found: 150 groups
✓ Parameters changed: 150/150
✓ Trained voice score: 0.7500
✓ Generated <|endoftext|> tokens: 2
```

**Bad Output**:
```
⚠ WARNING: No LoRA parameters found!
✓ Parameters changed: 2/2
✓ Trained voice score: 0.0000
⚠ WARNING: Significant increase in <|endoftext|> tokens!
```

---

## Performance Issues

### Slow Training

**Checklist**:
- [ ] LoRA adapters enabled (not training full model)
- [ ] Batch size optimized for hardware
- [ ] Sequence length not excessive
- [ ] MLX using GPU (not CPU)

**Optimization Tips**:

1. **Increase batch size**:
```yaml
sft:
  per_device_batch_size: 16  # Mac Studio M2 Ultra can handle this
```

2. **Reduce sequence length**:
```yaml
sft:
  max_seq_length: 1024  # Faster than 2048
```

3. **Verify GPU usage**:
```python
import mlx.core as mx
print(f"Device: {mx.default_device()}")  # Should be 'gpu'
```

### High Memory Usage

**Checklist**:
- [ ] LoRA adapters enabled
- [ ] Batch size appropriate
- [ ] Cache cleared between topics
- [ ] No memory leaks

**Memory Optimization**:

1. **Clear cache**:
```python
import mlx.core as mx
mx.metal.clear_cache()
```

2. **Reduce batch size**:
```yaml
sft:
  per_device_batch_size: 2
```

3. **Monitor memory**:
```python
import mlx.core as mx
print(f"Active: {mx.metal.get_active_memory() / 1e9:.2f} GB")
print(f"Peak: {mx.metal.get_peak_memory() / 1e9:.2f} GB")
print(f"Cache: {mx.metal.get_cache_memory() / 1e9:.2f} GB")
```

---

## Error Messages

### "MLX not installed"

**Error**:
```
RuntimeError: mlx-lm not installed. Install with: pip install mlx-lm
```

**Solution**:
```bash
pip install mlx mlx-lm
```

### "Model object has no attribute 'named_parameters'"

**Error**:
```
AttributeError: 'Model' object has no attribute 'named_parameters'
```

**Cause**: MLX models use `.parameters()` (returns dict), not `.named_parameters()`

**Solution**: Use `.parameters().items()` instead:
```python
# Wrong (PyTorch style)
for name, param in model.named_parameters():
    ...

# Correct (MLX style)
for name, param in model.parameters().items():
    ...
```

### "No LoRA gradients found"

**Warning**:
```
WARNING: No LoRA gradients found, skipping update
```

**Cause**: LoRA adapter not properly attached

**Solution**: Verify adapter setup (see Procedure 1 above)

---

## Verification Tools

### Quick Verification Script

```python
#!/usr/bin/env python3
"""Quick verification of MLX backend setup."""

import sys
sys.path.insert(0, '.')

from src.backends import create_backend

def verify_mlx_setup():
    print("=" * 60)
    print("MLX Backend Verification")
    print("=" * 60)
    
    # 1. Create backend
    print("\n1. Creating MLX backend...")
    try:
        backend = create_backend('mlx')
        print("   ✓ Backend created")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # 2. Load model
    print("\n2. Loading model...")
    try:
        model = backend.load_model('Qwen/Qwen2.5-7B-Instruct')
        print(f"   ✓ Model loaded: {model.num_parameters:,} parameters")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # 3. Create trainer (should add LoRA)
    print("\n3. Creating SFT trainer...")
    try:
        trainer = backend.create_sft_trainer(model, {})
        print("   ✓ Trainer created")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # 4. Verify LoRA
    print("\n4. Verifying LoRA adapter...")
    has_adapter = model.has_adapter()
    trainable_params = model.num_trainable_parameters
    
    print(f"   Has adapter: {has_adapter}")
    print(f"   Trainable params: {trainable_params:,}")
    
    if not has_adapter:
        print("   ✗ No LoRA adapter found!")
        return False
    
    if trainable_params > 100_000_000:
        print("   ✗ Too many trainable params (should be < 10M)")
        return False
    
    print("   ✓ LoRA adapter properly configured")
    
    # 5. Check LoRA parameters
    print("\n5. Checking LoRA parameters...")
    mlx_model = model.get_underlying_model()
    params_dict = mlx_model.parameters()
    lora_params = [n for n in params_dict.keys() if 'lora' in n.lower()]
    
    print(f"   LoRA parameters: {len(lora_params)}")
    if len(lora_params) > 0:
        print(f"   Sample: {lora_params[0]}")
        print("   ✓ LoRA parameters found")
    else:
        print("   ✗ No LoRA parameters found!")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All checks passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = verify_mlx_setup()
    sys.exit(0 if success else 1)
```

Save as `scripts/verify_mlx_setup.py` and run:
```bash
python3 scripts/verify_mlx_setup.py
```

---

## Getting Help

### Documentation

- **MLX LoRA Issue**: `MLX_LORA_TRAINING_ISSUE.md`
- **Architecture**: `MLX_LORA_ARCHITECTURE.md`
- **Testing Guide**: `MLX_REAL_WORLD_TESTING.md`
- **Diagnostic Tests**: `../../tests/debug/README.md`

### Diagnostic Commands

```bash
# Run full diagnostic test
python3 tests/debug/test_mlx_training_state.py

# Check MLX installation
python3 -c "import mlx.core as mx; print(mx.__version__)"

# Check model loading
python3 -c "from mlx_lm import load; load('Qwen/Qwen2.5-7B-Instruct')"

# Monitor training
python3 scripts/monitor_training.sh
```

### Common Solutions Summary

| Issue | Quick Fix |
|-------|-----------|
| Voice scores 0.00 | Run diagnostic test, verify LoRA adapter |
| Training slow | Increase batch size, reduce sequence length |
| Out of memory | Reduce batch size, clear cache |
| Model not loading | Install mlx-lm, check internet connection |
| No LoRA params | Update to latest version with fix |

---

**Last Updated**: January 28, 2026  
**Version**: 1.0
