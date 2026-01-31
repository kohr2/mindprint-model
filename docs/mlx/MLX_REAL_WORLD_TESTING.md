# MLX Real-World Testing Guide

This guide documents the process for running real-world training tests on Mac Studio with the MLX backend.

## Prerequisites

- Mac Studio with 64GB+ RAM
- MLX and mlx-lm installed
- SSH access configured (preferably with SSH keys)
- Training data available in `data/bob_loukas/`

## Known Issues and Fixes

### LoRA Adapter Support (Fixed Jan 28, 2026)

**Issue**: Early versions of MLX backend did not properly implement LoRA adapters, causing model corruption during training.

**Symptoms**:
- Voice scores 0.00 after training
- Model generates `<|endoftext|>` tokens
- Full 7B model trained instead of LoRA adapters

**Verification**: Check if your version has the fix by running:

```python
python3 -c "
import sys
sys.path.insert(0, '.')
import src.backends.pytorch
import src.backends.mlx
from src.backends import create_backend

backend = create_backend('mlx')
model = backend.load_model('Qwen/Qwen2.5-7B-Instruct')

# Create trainer (should add LoRA adapter)
trainer = backend.create_sft_trainer(model, {})

# Check for LoRA
print(f'Has adapter: {model.has_adapter()}')
print(f'Trainable params: {model.num_trainable_parameters:,}')
"
```

**Expected output** (with fix):
```
Has adapter: True
Trainable params: 8,028,160
```

**Bad output** (without fix):
```
Has adapter: False
Trainable params: 7,000,000,000
```

If you see 7+ billion trainable parameters, you need to update to the fixed version.

**Documentation**: See `MLX_LORA_TRAINING_ISSUE.md` for complete investigation details.

## Quick Start

### 1. Run Diagnostic Test (Recommended First Step)

Before running training, verify MLX LoRA training is working correctly:

```bash
# From local machine (with SSH access to Mac Studio)
cd ~/mindprint-model
MAC_STUDIO_PASSWORD="your_password" ./scripts/run_test_on_mac_studio.sh memeticas-mac-studio memetica-studio
```

Or manually via SSH:
```bash
# On Mac Studio
cd ~/mindprint-model
python3 tests/debug/test_mlx_training_state.py
```

This diagnostic test verifies:
- LoRA adapters are properly attached
- Only LoRA parameters are trainable
- Training doesn't corrupt base model
- Generation quality is maintained

**Expected output**: LoRA parameters found (100+ groups), voice score > 0.10 after training

### 2. Verify Configuration (Dry-Run)

Before running actual training, verify the configuration:

```bash
# On Mac Studio
cd ~/mindprint-model
./scripts/test_dry_run.sh
```

Or manually:
```bash
python3 scripts/run_orpo_training.py \
    --config configs/training_pipeline.yaml \
    --backend mlx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dry-run
```

Expected output: Configuration details without errors.

### 3. Single-Topic Test

Test on a single topic first to validate the MLX backend:

```bash
# On Mac Studio
cd ~/mindprint-model
./scripts/test_single_topic_mlx.sh unit-01 chapter-01 topic-01
```

This will:
- Verify MLX backend loads correctly
- Run SFT training on one topic
- Save adapters and checkpoints
- Produce evaluation metrics

**Success Criteria:**
- Training completes without errors
- Voice scores > 0.0 after SFT
- Adapter checkpoints saved in `output/`
- Memory usage stays within bounds

### 3. Monitor Training

In another terminal, monitor progress:

```bash
# On Mac Studio
./scripts/monitor_training.sh
```

Or follow logs directly:
```bash
tail -f logs/training_*.log
```

### 4. Full Curriculum Training

Once single-topic test succeeds, run full curriculum:

```bash
# On Mac Studio
python3 scripts/run_orpo_training.py \
    --config configs/training_pipeline.yaml \
    --backend mlx \
    --model Qwen/Qwen2.5-7B-Instruct \
    > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Or use the deployment script from your local machine:
```bash
# From your local machine
cd mindprint-model
# On Mac Studio:
./scripts/local_train.sh

# Or from MacBook (optional):
./scripts/quick_deploy.sh
```

## Metrics to Collect

During and after training, collect these metrics:

### Performance Metrics
- **Training time per topic** (SFT): Target < 2 hours
- **Memory usage**: Should stay under 64GB
- **GPU utilization**: Monitor with Activity Monitor or `top`

### Quality Metrics
- **Voice scores**: Before/after SFT, before/after DPO
- **Accuracy scores**: Quiz evaluation results
- **Topic pass rate**: Percentage of topics passing threshold

### Example Collection Script

```bash
# Extract metrics from log file
grep "voice_score" logs/training_*.log | tail -20
grep "accuracy" logs/training_*.log | tail -20
grep "Training time" logs/training_*.log
```

## Current Configuration

The current config (`configs/training_pipeline.yaml`) is set for **SFT-only testing**:

```yaml
thresholds:
  accuracy_threshold: 999.0  # DPO disabled
  dpo_trigger_threshold: 999.0
  topic_pass_threshold: 0.70  # Lowered for SFT-only
```

This allows testing SFT training without DPO complications.

## Testing DPO on MLX

Once SFT produces good results, test DPO:

1. **Update config** to re-enable DPO:
```yaml
thresholds:
  accuracy_threshold: 0.70
  dpo_trigger_threshold: 0.75
  topic_pass_threshold: 0.90
```

2. **Run training** and monitor voice scores after DPO
3. **Compare** with PyTorch MPS results (which showed voice=0.00 after DPO)

**Hypothesis**: DPO corruption may be PyTorch MPS-specific. MLX backend may not have this issue.

## Troubleshooting

### MLX Not Found
```bash
pip3 install mlx mlx-lm
```

### Out of Memory
- Reduce batch size in config
- Use smaller model (Qwen2.5-7B instead of 72B)
- Close other applications

### Training Hangs
- Check logs for errors
- Verify data directory exists
- Check disk space

### Voice Scores = 0.00
- This was the PyTorch MPS issue
- If it happens on MLX, investigate evaluation function
- Check adapter loading/saving

## Expected Results

### SFT Training (Current Test)
- **Time**: ~1-2 hours per topic
- **Memory**: ~20-30GB peak
- **Voice Score**: 0.40-0.60 (target > 0.50)
- **Accuracy**: 0.70-0.90 (target > 0.70)

### DPO Training (Future Test)
- **Time**: +1-2 hours per topic
- **Memory**: Similar to SFT
- **Voice Score**: Should improve or maintain (not drop to 0.00)
- **Accuracy**: Should maintain or improve

## Next Steps After Testing

1. **Update Benchmarks**: Replace placeholder benchmarks in docs with real metrics
2. **Fix Issues**: Address any problems found during testing
3. **Enable DPO**: If SFT works well, test DPO
4. **Production Deployment**: Once validated, use for full curriculum training

## Files Created

- `scripts/test_dry_run.sh` - Configuration verification
- `scripts/test_single_topic_mlx.sh` - Single-topic test
- `scripts/monitor_training.sh` - Training monitoring
- `scripts/local_train.sh` - Local training script (run on Mac Studio)
- `scripts/local_monitor.sh` - Monitor training progress

## References

- [Backend Implementation Summary](IMPLEMENTATION_SUMMARY.md)
- [Adapter Stacking Debug](../ADAPTER_STACKING_DEBUG.md)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
