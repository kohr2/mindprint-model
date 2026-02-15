# MLX Real-World Testing Guide

Guide for running training tests on Mac Studio with the MLX backend.

## Prerequisites

- Mac Studio with 64GB+ RAM
- MLX and mlx-lm installed
- Training data available in `data/bob_loukas/`

## Known Issues and Fixes

### LoRA Adapter Support (Fixed Jan 28, 2026)

**Issue**: Early versions of MLX backend did not properly implement LoRA adapters, causing model corruption during training.

**Symptoms**:
- Voice scores 0.00 after training
- Model generates `<|endoftext|>` tokens
- Full 7B model trained instead of LoRA adapters

**Verification**:

```python
python3 -c "
import sys
sys.path.insert(0, '.')
import src.backends.pytorch
import src.backends.mlx
from src.backends import create_backend

backend = create_backend('mlx')
model = backend.load_model('Qwen/Qwen2.5-7B-Instruct')
trainer = backend.create_sft_trainer(model, {})

print(f'Has adapter: {model.has_adapter()}')
print(f'Trainable params: {model.num_trainable_parameters:,}')
"
```

**Expected**: `Has adapter: True`, `Trainable params: 8,028,160`
**Bad**: `Has adapter: False`, `Trainable params: 7,000,000,000`

See `MLX_LORA_TRAINING_ISSUE.md` for investigation details.

## Quick Start

### 1. Run Diagnostic Test

```bash
cd ~/mindprint-model
python3 tests/debug/test_mlx_training_state.py
```

Verifies: LoRA adapters attached, only LoRA params trainable, no base model corruption, generation quality maintained.

### 2. Verify Configuration (Dry-Run)

```bash
cd ~/mindprint-model
python3 scripts/run_orpo_training.py \
    --config configs/training_pipeline.yaml \
    --backend mlx \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dry-run
```

### 3. Single-Topic Test

```bash
cd ~/mindprint-model
./scripts/test_single_topic_mlx.sh unit-01 chapter-01 topic-01
```

**Success criteria**: Training completes, voice scores > 0.0, adapters saved, memory within bounds.

### 4. Monitor Training

```bash
./scripts/monitor_training.sh
# Or follow logs directly
tail -f logs/training_*.log
```

### 5. Full Curriculum Training

```bash
cd ~/mindprint-model
python3 scripts/run_orpo_training.py \
    --config configs/training_pipeline.yaml \
    --backend mlx \
    --model Qwen/Qwen2.5-7B-Instruct \
    > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## Metrics to Collect

### Performance
- **Training time per topic**: Target < 2 hours
- **Memory usage**: Should stay under 64GB
- **GPU utilization**: Monitor with Activity Monitor or `top`

### Quality
- **Voice scores**: Before/after training
- **Accuracy scores**: Quiz evaluation results
- **Topic pass rate**: Percentage passing threshold

```bash
grep "voice_score" logs/training_*.log | tail -20
grep "accuracy" logs/training_*.log | tail -20
grep "Training time" logs/training_*.log
```

## Troubleshooting

**MLX Not Found**: `pip3 install mlx mlx-lm`

**Out of Memory**: Reduce batch size, use smaller model, close other applications.

**Training Hangs**: Check logs, verify data directory exists, check disk space.

**Voice Scores = 0.00**: Was the PyTorch MPS issue. On MLX, check adapter loading/saving and evaluation function.

## Expected Results

- **Time**: ~1-2 hours per topic
- **Memory**: ~20-30GB peak
- **Voice Score**: 0.40-0.60 (target > 0.50)
- **Accuracy**: 0.70-0.90 (target > 0.70)

## References

- [MLX LoRA Training Issue](MLX_LORA_TRAINING_ISSUE.md)
- [MLX LoRA Architecture](MLX_LORA_ARCHITECTURE.md)
- [MLX Backend Troubleshooting](MLX_BACKEND_TROUBLESHOOTING.md)
