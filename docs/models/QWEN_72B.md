# Qwen2.5-72B Model Guide

Complete guide for using Qwen2.5-72B-Instruct (72.7B parameters) with the mindprint-model training pipeline.

## Overview

Qwen2.5-72B is a large-scale language model from Alibaba Cloud that offers significant capabilities while remaining feasible to train on high-end consumer hardware with INT4 quantization.

**Key Specifications:**
- **Parameters**: 72.7B (70.0B non-embedding)
- **Layers**: 80
- **Architecture**: Dense (not MoE), Grouped Query Attention
- **Context Length**: 131,072 tokens (with YaRN scaling)
- **Prompt Format**: ChatML (same as Qwen2.5-7B)
- **HuggingFace Path**: `Qwen/Qwen2.5-72B-Instruct`

## Hardware Requirements

### Mac Studio M2 Ultra (64GB) - RECOMMENDED ✅

**Configuration:**
- **Memory**: 64GB unified memory (48GB usable for VRAM)
- **Quantization**: INT4 (NormalFloat 4-bit)
- **Backend**: MLX (Apple Silicon optimized)
- **Batch Size**: 1
- **Gradient Accumulation**: 4

**Memory Budget:**
```
Base Model (INT4):          36 GB
LoRA Adapters:              2 GB
Optimizer States:           3 GB
Activations (batch=1):      4 GB
KV Cache (4K context):      2 GB
--------------------------------------
Total:                      47 GB (within 48 GB limit)
```

**Training Time Estimates:**
- SFT: 6-8 hours per topic
- DPO: 8-10 hours per topic
- Full curriculum (10 topics): 5-7 days

### Cloud GPU Options

**8x H100 80GB (640GB total):**
- Precision: INT4 or BF16
- Batch size: 2
- Training time: 2-3 hours per topic (SFT), 3-4 hours (DPO)
- Cost: ~$98/hour = $2,000-$4,000 for full training

**4x A100 80GB (320GB total):**
- Precision: INT4 required
- Batch size: 1
- Training time: 4-5 hours per topic (SFT), 5-6 hours (DPO)
- Cost: ~$40/hour = $1,000-$2,000 for full training

## Layer Zone Mapping

Qwen2.5-72B uses an 80-layer architecture with the following zone distribution:

### Lexicon Zone (Layers 0-19) - 20 layers (25%)

**Purpose**: Bob Loukas's terminology embedding

**Layers**: 0, 1, 2, ..., 19 (first 20 layers)

**Target Modules**: `q_proj` only (query projection)

**Content Learned**:
- "4-year cycle", "accumulation", "distribution"
- "cycle low", "cycle high"
- "parabolic advance", "capitulation"
- Market-specific terminology

**Rationale**: Early layers are best for lexical patterns and terminology retrieval.

### Reasoning Zone (Layers 20-59) - 40 layers (50%)

**Purpose**: Cycle theory and market analysis reasoning

**Layers**: 20, 21, 22, ..., 59 (middle 40 layers)

**Target Modules**: `v_proj`, `up_proj`, `down_proj`

**Content Learned**:
- Cycle theory application
- Pattern recognition (market tops/bottoms)
- Multi-timeframe analysis
- Risk assessment methodology
- Market psychology analysis

**Rationale**: Middle layers handle complex reasoning and pattern integration. Gets the largest allocation (50%) for Bob's analytical depth.

### Voice Zone (Layers 60-79) - 20 layers (25%)

**Purpose**: Bob's teaching style and confidence markers

**Layers**: 60, 61, 62, ..., 79 (final 20 layers)

**Target Modules**: `o_proj`, `up_proj`, `down_proj`

**Content Learned**:
- Hedge language ("I think", "I believe", "in my view")
- Confidence markers ("clearly", "obviously", "definitely")
- Teaching cadence and engagement patterns
- Conversational style

**Rationale**: Late layers control output style and generation patterns.

## LoRA Configuration

### Conservative Strategy (Recommended)

```yaml
lora:
  r: 8              # Same as Gemma-3-12B (proven effective)
  alpha: 16         # Alpha = 2 * rank
  dropout: 0.05     # Light regularization
  target_modules:
    - q_proj        # Query (lexicon retrieval)
    - v_proj        # Value (reasoning integration)
    - o_proj        # Output (style projection)
    - up_proj       # FFN up-projection
    - down_proj     # FFN down-projection
```

**Trainable Parameters**: ~41M per adapter (0.06% of base model)

**Memory**: ~164MB per adapter (BF16)

### Aggressive Strategy (If Memory Allows)

```yaml
lora:
  r: 16             # 2x rank for more capacity
  alpha: 32         # Alpha = 2 * rank
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj        # Add key projection
    - v_proj
    - o_proj
    - gate_proj     # Add gate (SwiGLU)
    - up_proj
    - down_proj
```

**When to Use**: If initial results show underfitting or voice fidelity scores below 0.85.

## Training Configuration

### SFT Phase (Supervised Fine-Tuning)

```yaml
sft:
  learning_rate: 2e-4         # Slightly lower than Gemma (3e-4)
  batch_size: 1               # Limited by memory
  gradient_accumulation: 4    # Effective batch = 4
  epochs_per_topic: 3
  max_seq_length: 4096        # Conservative context (not full 131K)
  warmup_steps: 50
  weight_decay: 0.01
  scheduler: cosine
  gradient_checkpointing: true
  bf16_mixed_precision: true
```

### DPO Phase (Direct Preference Optimization)

```yaml
dpo:
  learning_rate: 1e-7         # Lower than Gemma (5e-7) due to model size
  batch_size: 1
  gradient_accumulation: 2    # Effective batch = 2
  steps_per_topic: 100
  beta: 0.1                   # KL penalty
  max_seq_length: 4096
  gradient_checkpointing: true
```

### Memory Optimization (Required for Mac Studio)

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use BF16 mixed precision
training_args = TrainingArguments(
    bf16=True,
    bf16_full_eval=False,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
)

# INT4 quantization with QLoRA (PyTorch)
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # Nested quantization
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-72B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
```

## Usage Examples

### Loading the Model (MLX Backend - Recommended)

```python
from src.backends import create_backend

# Create MLX backend with INT4 quantization
backend = create_backend(
    "mlx",
    device="auto",
    quantization="int4",
)

# Load model
model = backend.load_model("Qwen/Qwen2.5-72B-Instruct")

# Generate text
response = model.generate(
    "What is a cycle low in Bob Loukas's market analysis framework?",
    max_length=200,
)
print(response)
```

### Loading the Model (PyTorch Backend)

```python
from src.backends import create_backend

# Create PyTorch backend with INT4 quantization
backend = create_backend(
    "pytorch",
    device="cuda",  # or "mps" for Mac
    quantization="int4",
)

# Load model
model = backend.load_model("Qwen/Qwen2.5-72B-Instruct")
```

### Training with Zone-Specific LoRA

```python
from src.models.config import get_model_config
from src.training.dpo_pipeline import DPOPipeline, PipelineConfig

# Load Qwen-72B config
model_config = get_model_config("qwen-72b")

# Create pipeline config
pipeline_config = PipelineConfig(
    backend_type="mlx",
    backend_device="auto",
    backend_dtype="float16",
    sft_learning_rate=2e-4,
    sft_batch_size=1,
    dpo_learning_rate=1e-7,
    dpo_batch_size=1,
)

# Initialize pipeline
pipeline = DPOPipeline(
    model_config=model_config,
    config=pipeline_config,
)

# Train on specific topics
pipeline.train_topics(
    topics=["market_cycles", "accumulation"],
    data_dir="data/bob_loukas/",
    output_dir="checkpoints/qwen_72b/",
)
```

### Using Pre-trained Config

```python
from src.models.config import load_model_config

# Load from YAML
config = load_model_config("src/models/qwen_72b_config.yaml")

print(f"Model: {config.name}")
print(f"Layers: {config.layers}")
print(f"Lexicon layers: {len(config.get_layers_for_zone('lexicon'))}")
print(f"Reasoning layers: {len(config.get_layers_for_zone('reasoning'))}")
print(f"Voice layers: {len(config.get_layers_for_zone('voice'))}")
```

## Performance Optimization

### Mac Studio M2 Ultra

**Recommended Settings:**
- Backend: MLX (native Apple Silicon support)
- Quantization: INT4
- Batch size: 1
- Gradient accumulation: 4
- Max sequence length: 4096 (down from 131K)
- Gradient checkpointing: Enabled
- Mixed precision: BF16

**Memory Monitoring:**
```bash
# Monitor memory usage during training
watch -n 1 'ps aux | grep python'

# Or use Activity Monitor app
open -a "Activity Monitor"
```

### Cloud GPU

**Recommended Settings:**
- Backend: PyTorch
- Quantization: INT4 or BF16 (if memory allows)
- Batch size: 2
- Gradient accumulation: 2
- Flash Attention 2: Enabled
- DeepSpeed ZeRO-3: For multi-GPU

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms:**
- "CUDA out of memory" or "MPS backend out of memory"
- System freezes
- Training crashes

**Solutions:**
1. **Reduce batch size to 1** (if not already)
2. **Limit max_seq_length to 2048** (down from 4096)
3. **Reduce LoRA rank to 4** (down from 8)
4. **Use nested quantization**: `bnb_4bit_use_double_quant=True`
5. **Close other applications** to free memory
6. **Restart Python kernel** between training runs

### Slow Training Speed

**Symptoms:**
- More than 12 hours per topic
- Training progress stalls

**Solutions:**
1. **Use MLX backend** on Mac (faster than PyTorch MPS)
2. **Reduce max_seq_length to 2048**
3. **Increase gradient accumulation** instead of batch size
4. **Profile training loop** to identify bottlenecks
5. **Disable evaluation during training**: Only evaluate at end

### INT4 Training Instability

**Symptoms:**
- Loss spikes
- NaN gradients
- Poor convergence

**Solutions:**
1. **Lower learning rates**:
   - SFT: 1e-4 (down from 2e-4)
   - DPO: 5e-8 (down from 1e-7)
2. **Aggressive gradient clipping**: `max_grad_norm=0.3`
3. **Increase warmup steps**: From 50 to 100
4. **Monitor gradient norms**: Add logging in training loop
5. **Use BF16 for optimizer states**: `optim_bits=32`

### Poor Voice Fidelity

**Symptoms:**
- Voice fidelity scores < 0.75
- Model doesn't sound like Bob

**Solutions:**
1. **Increase LoRA rank to 16**: More capacity for style
2. **Train for more epochs**: SFT 5 epochs (up from 3)
3. **Adjust voice zone layers**: May need different split
4. **Generate more preference pairs**: Increase DPO data
5. **Check voice marker accuracy**: Review evaluation metrics

### Model Architecture Errors

**Symptoms:**
- "Module not found" errors
- PEFT config failures

**Solutions:**
1. **Inspect module names**:
   ```python
   model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-72B-Instruct")
   for name, _ in model.named_modules():
       print(name)
   ```
2. **Verify target modules match**: Check against actual architecture
3. **Test on Qwen2.5-7B first**: Ensure modules are correct
4. **Refer to HuggingFace model card**: Check for architecture notes

## Comparison with Other Models

| Metric | Gemma-3-12B | Qwen2.5-7B | **Qwen2.5-72B** |
|--------|-------------|------------|-----------------|
| Parameters | 12B | 7B | **72.7B** |
| Layers | 48 | 28 | **80** |
| VRAM (INT4) | 12GB | 8GB | **36GB** |
| VRAM (FP16) | 24GB | 14GB | **145GB** |
| Context Length | 128K | 131K | **131K** |
| Training Time (SFT/topic) | 2-3h | 1-2h | **6-8h** |
| Mac Studio Compatible (FP16) | ✅ Yes | ✅ Yes | ❌ No |
| Mac Studio Compatible (INT4) | ✅ Yes | ✅ Yes | ✅ **Yes** |
| Voice Capacity | Good | Fair | **Excellent** |
| Reasoning Depth | Good | Fair | **Excellent** |

**When to Use Qwen2.5-72B:**
- Need maximum voice fidelity
- Complex reasoning requirements
- Large context understanding
- Have Mac Studio M2 Ultra (64GB+) or cloud GPU access

**When to Use Gemma-3-12B:**
- Balanced performance and resource usage
- Faster iteration during development
- Limited memory (< 64GB)

**When to Use Qwen2.5-7B:**
- Quick experiments
- Very limited memory (< 32GB)
- Acceptable voice quality is sufficient

## Best Practices

1. **Start Small**: Test on Qwen2.5-7B first to validate your pipeline
2. **Monitor Memory**: Watch memory usage during first training run
3. **Save Checkpoints**: Save adapters after each topic
4. **Evaluate Regularly**: Check voice fidelity after each zone
5. **Compare Baselines**: Benchmark against Gemma-3-12B results
6. **Use Version Control**: Track adapter versions with git
7. **Document Hyperparameters**: Record all settings for reproducibility

## Expected Results

**Voice Fidelity:**
- Target: ≥ 0.85
- Excellent: ≥ 0.90
- With 72B parameters, should exceed Gemma-3-12B baseline

**Accuracy:**
- Target: ≥ 0.85 on Bob Loukas curriculum
- Excellent: ≥ 0.90

**Training Stability:**
- Loss should decrease smoothly
- No NaN gradients with proper settings
- Voice fidelity should improve across DPO steps

## Additional Resources

- [HuggingFace Model Card](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)
- [Qwen2.5 Official Blog](https://qwenlm.github.io/blog/qwen2.5/)
- [INT4 Quantization Guide](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- [MLX Documentation](https://ml-explore.github.io/mlx/)

## Support

For issues specific to Qwen2.5-72B:
1. Check [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)
2. Review [GitHub Issues](https://github.com/your-repo/mindprint-model/issues)
3. Join [Discord Community](https://discord.gg/your-server)

For HuggingFace model issues:
- [Qwen Discussion Board](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/discussions)
