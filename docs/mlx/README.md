# MLX Backend Documentation

Documentation for the MLX backend implementation for Apple Silicon.

**Last Updated**: January 28, 2026

---

## Overview

The MLX backend provides native Apple Silicon support for training language models with LoRA adapters. It implements manual training loops using Apple's MLX framework, avoiding the PyTorch MPS adapter corruption issues.

### Status

✅ **Production Ready** (as of January 28, 2026)
- LoRA training fully functional
- Generation quality preserved
- No model corruption
- Verified on Mac Studio M2 Ultra

---

## Documentation Files

### Core Documentation

1. **[MLX LoRA Training Issue](MLX_LORA_TRAINING_ISSUE.md)** ⭐ **START HERE**
   - Complete investigation of the LoRA training bug
   - Root cause analysis
   - Solution implementation
   - Verification results
   - **Read this first to understand the fix**

2. **[MLX LoRA Architecture](MLX_LORA_ARCHITECTURE.md)**
   - Technical architecture details
   - LoRA implementation specifics
   - Best practices for MLX training
   - Code examples and patterns

3. **[MLX Backend Troubleshooting](MLX_BACKEND_TROUBLESHOOTING.md)**
   - Common issues and solutions
   - Diagnostic procedures
   - Performance optimization tips
   - Error message reference

4. **[MLX Real-World Testing](MLX_REAL_WORLD_TESTING.md)**
   - Testing procedures for Mac Studio
   - Remote testing via SSH
   - Diagnostic test guide
   - Training verification steps

---

## Quick Reference

### Key Concepts

**LoRA Training Fix (Jan 28, 2026)**:
- Use official `linear_to_lora_layers()` for conversion
- Selective unfreeze: `m.unfreeze(keys=['lora_a', 'lora_b'], strict=False)`
- Initialize optimizer: `optimizer.init(model.trainable_parameters())`
- Use `nn.value_and_grad()` not `mx.value_and_grad()`

**Result**: 392 trainable parameters (LoRA only), 196 LoRA layers, no corruption

### Common Tasks

| Task | Command/Code |
|------|--------------|
| Run diagnostic test | `./scripts/run_test_on_mac_studio.sh <host> <user>` |
| Check LoRA params | `model.num_trainable_parameters` (should be 392 for Qwen2.5-7B) |
| Verify generation | Check for `<\|endoftext\|>` spam (should be 0) |
| Monitor training | `scripts/monitor_training.sh` |

### Issue Resolution

| Symptom | Solution | Documentation |
|---------|----------|---------------|
| `<\|endoftext\|>` spam | Run diagnostic test | [Troubleshooting](MLX_BACKEND_TROUBLESHOOTING.md) |
| Voice score 0.00 | Verify LoRA adapters | [Training Issue](MLX_LORA_TRAINING_ISSUE.md) |
| Full model training | Check trainable params | [Architecture](MLX_LORA_ARCHITECTURE.md) |
| Generation corrupted | Re-apply fix | [Training Issue](MLX_LORA_TRAINING_ISSUE.md) |

---

## Implementation Files

### Core Backend Files

Located in `src/backends/mlx/`:

- `mlx_backend.py` - Main backend implementation
- `mlx_model.py` - Model wrapper with `tree_flatten` parameter counting
- `mlx_adapter_manager.py` - LoRA adapter operations using `linear_to_lora_layers()`
- `mlx_sft_trainer.py` - SFT training loop with `nn.value_and_grad()`
- `mlx_dpo_trainer.py` - DPO training loop
- `mlx_device_manager.py` - Unified memory management

### Test Files

Located in `tests/`:

- `tests/debug/test_mlx_training_state.py` - Comprehensive diagnostic test
- `tests/unit/backends/mlx/test_mlx_backend.py` - Unit tests
- `tests/integration/test_backend_pipeline.py` - Integration tests

### Scripts

Located in `scripts/`:

- `scripts/run_test_on_mac_studio.sh` - Remote diagnostic test runner
- `scripts/train_on_mac_studio.sh` - Remote training script
- `scripts/monitor_training.sh` - Training progress monitor

---

## Timeline

### January 10, 2026: Initial Implementation
- Multi-backend architecture completed
- MLX backend with manual training loops
- Initial LoRA adapter support (had bugs)

### January 28, 2026: LoRA Training Fix
- Discovered critical LoRA training bug
- Implemented proper `linear_to_lora_layers()` usage
- Added selective parameter unfreezing
- Verified on Mac Studio M2 Ultra
- **Result**: MLX backend now fully functional

---

## Key Learnings

### What Works

✅ Official `mlx_lm.tuner.utils.linear_to_lora_layers()`
✅ Selective unfreeze: `m.unfreeze(keys=['lora_a', 'lora_b'], strict=False)`
✅ Optimizer init: `optimizer.init(model.trainable_parameters())`
✅ Gradient computation: `nn.value_and_grad(model, loss_fn)`
✅ Parameter counting: `tree_flatten(model.trainable_parameters())`

### What Doesn't Work

❌ Manual `LoRALinear.from_base()` conversion
❌ Calling `unfreeze()` without `keys` parameter
❌ Skipping `optimizer.init()`
❌ Using `mx.value_and_grad()` instead of `nn.value_and_grad()`
❌ Top-level parameter counting without `tree_flatten`

---

## Getting Help

### Diagnostic Steps

1. **Run diagnostic test**: `./scripts/run_test_on_mac_studio.sh <host> <user>`
2. **Check trainable parameters**: Should be 392 for Qwen2.5-7B with LoRA rank 8
3. **Verify LoRA layers**: Should be 196 layers
4. **Test generation**: Should produce coherent text, not `<|endoftext|>` spam
5. **Check voice score**: Should maintain baseline score (e.g., 0.1167)

### Documentation

- **Issue Investigation**: [MLX_LORA_TRAINING_ISSUE.md](MLX_LORA_TRAINING_ISSUE.md)
- **Technical Details**: [MLX_LORA_ARCHITECTURE.md](MLX_LORA_ARCHITECTURE.md)
- **Troubleshooting**: [MLX_BACKEND_TROUBLESHOOTING.md](MLX_BACKEND_TROUBLESHOOTING.md)
- **Testing Guide**: [MLX_REAL_WORLD_TESTING.md](MLX_REAL_WORLD_TESTING.md)

### Support

For issues or questions:
1. Check [Troubleshooting Guide](MLX_BACKEND_TROUBLESHOOTING.md)
2. Review [Training Issue Doc](MLX_LORA_TRAINING_ISSUE.md)
3. Run diagnostic test
4. Check implementation files for code examples

---

## Contributing

When updating MLX backend documentation:

1. Update the relevant documentation file
2. Add entry to this README if needed
3. Update "Last Updated" date
4. Cross-reference related docs
5. Include code examples where helpful

---

**Document Version**: 1.0  
**Last Updated**: January 28, 2026  
**Status**: Production Ready
