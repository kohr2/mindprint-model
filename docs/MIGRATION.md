# Migration Guide

Guide for migrating existing training code to use the new backend system.

## Overview

The backend system provides a unified interface for training with PyTorch or MLX. This guide shows how to migrate existing code that directly uses PyTorch trainers.

---

## Why Migrate?

**Benefits:**
- ✅ Switch backends by changing one config line
- ✅ Avoid PyTorch MPS corruption bugs (use MLX on Mac Studio)
- ✅ Unified interface across frameworks
- ✅ Better testing and maintainability
- ✅ Future-proof for new backends

**Backward Compatibility:**
- ✅ Old code continues to work (legacy mode)
- ✅ No breaking changes
- ✅ Gradual migration possible

---

## Migration Paths

### Path 1: Keep Legacy Mode (No Changes)

If you're happy with direct PyTorch usage and don't need MLX:

```yaml
# configs/training_pipeline.yaml
backend:
  type: null  # or omit backend section entirely
```

Your existing code continues to work unchanged.

### Path 2: Use PyTorch Backend (Minimal Changes)

Switch to backend interface while keeping PyTorch:

**Before:**
```python
from src.training.sft_trainer import SFTTrainer, SFTConfig

config = SFTConfig(
    learning_rate=3e-4,
    num_epochs=3,
    per_device_batch_size=4,
)
trainer = SFTTrainer(model, tokenizer, config)
result = trainer.train(train_data)
```

**After:**
```python
from src.backends import create_backend

backend = create_backend("pytorch", device="cuda")
model_interface = backend.load_model("Qwen/Qwen2.5-7B-Instruct")

config = {
    "learning_rate": 3e-4,
    "num_epochs": 3,
    "per_device_batch_size": 4,
}
trainer = backend.create_sft_trainer(model_interface, config)
result = trainer.train(train_data)
```

**Changes:**
- Use `create_backend()` instead of importing trainers directly
- Config is now a dict instead of dataclass
- Model is loaded via backend
- Same training API

### Path 3: Use MLX Backend (Mac Studio)

Switch to MLX to avoid MPS corruption:

**Before (PyTorch with MPS issues):**
```python
# Corruption after merge_and_unload() on MPS
model = AutoModelForCausalLM.from_pretrained(...)
trainer = SFTTrainer(model, tokenizer, config)
result = trainer.train(train_data)
model = trainer.get_model()
model = model.merge_and_unload()  # ❌ Corrupts on MPS!
```

**After (MLX without corruption):**
```python
from src.backends import create_backend

backend = create_backend("mlx", device="auto")  # No MPS corruption!
model = backend.load_model("Qwen/Qwen2.5-7B-Instruct")

trainer = backend.create_sft_trainer(model, config)
result = trainer.train(train_data)

# Adapter operations work reliably
trainer.save_adapter(Path("./adapter"))  # ✅ No corruption
```

---

## Step-by-Step Migration

### Step 1: Update Imports

**Before:**
```python
from src.training.sft_trainer import SFTTrainer, SFTConfig
from src.training.dpo_trainer import Rank1DPOTrainer, Rank1DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
```

**After:**
```python
from src.backends import create_backend
# Old imports still available for reference
```

### Step 2: Create Backend

**Before:**
```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="mps",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
```

**After:**
```python
backend = create_backend(
    backend_type="mlx",  # or "pytorch"
    device="auto",
    dtype="float16",
)
model = backend.load_model("Qwen/Qwen2.5-7B-Instruct")
```

### Step 3: Convert Config

**Before:**
```python
from src.training.sft_trainer import SFTConfig

config = SFTConfig(
    learning_rate=3e-4,
    num_epochs=3,
    per_device_batch_size=4,
    gradient_accumulation_steps=4,
    max_seq_length=2048,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "o_proj"],
    device="mps",
    dtype="float16",
)
```

**After:**
```python
config = {
    "learning_rate": 3e-4,
    "num_epochs": 3,
    "per_device_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 2048,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "o_proj"],
    # device/dtype come from backend config
}
```

### Step 4: Create Trainer

**Before:**
```python
trainer = SFTTrainer(model, tokenizer, config)
```

**After:**
```python
trainer = backend.create_sft_trainer(model, config)
# Tokenizer is embedded in model interface
```

### Step 5: Train

**Training API is identical:**
```python
# Both old and new
result = trainer.train(train_data)
result = trainer.train_on_topic(topic_data, topic_id)
trainer.save_adapter(Path("./adapter"))
```

---

## DPOPipeline Migration

### Before (Legacy Mode)

```python
from src.training.dpo_pipeline import DPOPipeline, PipelineConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model directly
model = AutoModelForCausalLM.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(...)

# Create pipeline
config = PipelineConfig(
    sft_epochs_per_topic=3,
    dpo_steps_per_topic=100,
)
pipeline = DPOPipeline(model, tokenizer, config)

# Train
result = pipeline.train_curriculum()
```

### After (Backend Mode)

```python
from src.training.dpo_pipeline import DPOPipeline, PipelineConfig
from src.backends import create_backend

# Create backend
backend = create_backend("mlx", device="auto")

# Create pipeline with backend
config = PipelineConfig(
    backend_type="mlx",
    backend_device="auto",
    backend_dtype="float16",
    sft_epochs_per_topic=3,
    dpo_steps_per_topic=100,
)
pipeline = DPOPipeline(
    model=None,  # Loaded via backend
    tokenizer=None,
    config=config,
    backend=backend,
)

# Train (same API)
result = pipeline.train_curriculum()
```

---

## Config File Migration

### Before

```yaml
# configs/training_pipeline.yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct
  dtype: float16
  device: mps

sft:
  epochs_per_topic: 3
  learning_rate: 0.0003
  batch_size: 4
```

### After

```yaml
# configs/training_pipeline.yaml
backend:
  type: mlx  # or pytorch, or null for legacy
  device: auto
  dtype: float16

model:
  name: Qwen/Qwen2.5-7B-Instruct
  # device/dtype now come from backend

sft:
  epochs_per_topic: 3
  learning_rate: 0.0003
  batch_size: 4
```

---

## Common Migration Patterns

### Pattern 1: Model Loading

**Before:**
```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="mps",
)
```

**After:**
```python
from src.backends import create_backend

backend = create_backend("mlx", dtype="float16")
model = backend.load_model("Qwen/Qwen2.5-7B-Instruct")
```

### Pattern 2: Adapter Operations

**Before:**
```python
from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)

# Later: save adapter
model.save_pretrained("./adapter")

# Later: merge (❌ corrupts on MPS!)
model = model.merge_and_unload()
```

**After:**
```python
# Adapter is added automatically by trainer
trainer = backend.create_sft_trainer(model, config)
result = trainer.train(data)

# Save adapter (✅ no corruption)
trainer.save_adapter(Path("./adapter"))

# Model interface has adapter methods
model.save_adapter(Path("./adapter"))
model.load_adapter(Path("./adapter"))
```

### Pattern 3: Device Management

**Before:**
```python
# Manual device placement
model = model.to("mps")
data = {k: v.to("mps") for k, v in data.items()}

# Manual cache clearing
import torch
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
```

**After:**
```python
# Automatic device management
backend = create_backend("mlx", device="auto")

# Backend handles device placement
model = backend.load_model(...)

# Unified cache clearing
backend.get_device_manager().empty_cache()
```

---

## Testing Your Migration

### 1. Verify Backend Registration

```python
from src.backends import BackendRegistry

print("Available backends:", BackendRegistry.list_backends())
# Should show: ['pytorch', 'mlx']
```

### 2. Test Backend Creation

```python
from src.backends import create_backend

for backend_type in ["pytorch", "mlx"]:
    try:
        backend = create_backend(backend_type)
        print(f"✓ {backend_type} backend created")
    except Exception as e:
        print(f"✗ {backend_type} failed: {e}")
```

### 3. Compare Results

Run the same training with both backends:

```bash
# Compare backends
python scripts/compare_backends.py --model Qwen/Qwen2.5-7B-Instruct
```

### 4. Run Tests

```bash
# Unit tests
pytest tests/unit/backends/ -v

# Integration tests
pytest tests/integration/test_backend_pipeline.py -v

# Equivalence tests
pytest tests/integration/test_backend_equivalence.py -v
```

---

## Rollback Strategy

If you need to rollback:

### Option 1: Use Legacy Mode

```yaml
backend:
  type: null  # Disables backend system
```

### Option 2: Keep Old Code Branch

```bash
# Create backup branch before migration
git checkout -b pre-backend-migration
git checkout main

# If migration issues, revert
git checkout pre-backend-migration
```

### Option 3: Conditional Usage

```python
# Use backend if available, fallback to legacy
try:
    from src.backends import create_backend
    backend = create_backend("mlx")
    use_backend = True
except ImportError:
    use_backend = False

if use_backend:
    # New backend code
    model = backend.load_model(...)
else:
    # Legacy code
    model = AutoModelForCausalLM.from_pretrained(...)
```

---

## Migration Checklist

- [ ] Review backend documentation (`src/backends/README.md`)
- [ ] Choose backend (MLX for Mac, PyTorch for cloud)
- [ ] Update config files with backend section
- [ ] Update import statements
- [ ] Convert config dataclasses to dicts
- [ ] Replace direct trainer instantiation with backend.create_*_trainer()
- [ ] Test on single topic
- [ ] Run equivalence tests
- [ ] Update deployment scripts
- [ ] Update CI/CD pipelines
- [ ] Document backend choice for team

---

## FAQ

### Q: Do I have to migrate?

**A:** No. Legacy mode is fully supported. Migrate when:
- You want to use MLX on Mac Studio
- You need to avoid PyTorch MPS bugs
- You want unified interface across frameworks

### Q: Can I use both backends?

**A:** Yes! Install both and switch via config:

```yaml
backend:
  type: mlx  # Change to pytorch as needed
```

### Q: Will migration break my code?

**A:** No. The migration is backward compatible:
- Legacy mode works unchanged
- New interface is additive
- No breaking changes to existing code

### Q: What about performance?

**A:** Backend interface adds minimal overhead:
- PyTorch backend: ~1% overhead (wrapper layer)
- MLX backend: Comparable or faster on Apple Silicon
- See benchmarks in `docs/DEPLOYMENT.md`

### Q: How do I handle adapter corruption on MPS?

**A:** Switch to MLX backend:

```yaml
backend:
  type: mlx  # No corruption on MLX!
```

This is the recommended solution for Mac Studio training.

---

## Support

For migration help:

1. Check [src/backends/README.md](../src/backends/README.md) for API docs
2. See examples in `tests/integration/`
3. Run comparison script: `python scripts/compare_backends.py`
4. Open GitHub issue if you encounter problems

---

## References

- [Backend README](../src/backends/README.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Test Examples](../tests/integration/)
- [PyTorch MPS Issues](https://github.com/pytorch/pytorch/issues/78043)
