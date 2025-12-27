# DPO Training Execution

## Pre-Training Checklist

```bash
# 1. Environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Verify MPS (Apple Silicon)
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# 3. Download model
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('google/gemma-3-12b')"
```

## Training Script

```python
#!/usr/bin/env python3
# scripts/train_dpo.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.dpo_pipeline import DPOPipeline
from src.config import load_config

def main():
    config = load_config("configs/dpo.yaml")

    # Load model (fp16 on Apple Silicon - no quantization needed)
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-12b",
        torch_dtype=torch.float16,
        device_map="mps",
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b")

    # Train
    pipeline = DPOPipeline(model, tokenizer, config)
    pipeline.train_curriculum("../omnia/projects/bob_loukas/textbook")

    # Save
    pipeline.save("./output/bob_loukas_dpo")

if __name__ == "__main__":
    main()
```

## Configuration

```yaml
# configs/dpo.yaml

model:
  name: "google/gemma-3-12b"
  dtype: "float16"  # No quantization needed on Mac Studio (64GB unified memory)
  device: "mps"

lora:
  rank: 1
  alpha: 1.0
  target_modules: ["o_proj", "v_proj", "up_proj", "down_proj"]

dpo:
  beta: 0.1
  learning_rate: 5e-7
  max_steps: 100

training:
  sft_learning_rate: 3e-4
  sft_epochs: 3
  batch_size: 8  # Larger batch with 64GB unified memory

evaluation:
  topic_threshold: 0.90
  voice_threshold: 0.75
```

## Expected Timeline (Mac Studio M2 Ultra)

| Phase | Est. Time |
|-------|-----------|
| Data prep | 1 hour |
| Unit 1 | 10 hours |
| Unit 2 | 12 hours |
| Unit 3 | 10 hours |
| Unit 4 | 10 hours |
| **Total** | **~43 hours** |

*Note: MPS is ~1.2x slower than A100 for transformer training, but larger batch sizes partially compensate.*

## Monitoring

Key metrics to track:
- `accuracy`: Quiz accuracy per topic
- `voice_score`: Voice fidelity score
- `dpo_loss`: DPO loss (should decrease)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM | Reduce batch_size to 4 |
| Voice stuck low | Increase dpo.max_steps |
| Accuracy drops | Enable merge_after_chapter |
| MPS not available | Ensure PyTorch >= 2.0 with MPS support |
| Slow training | Enable gradient checkpointing |
| NaN losses | Try fp32 instead of fp16 for stability |

### Apple Silicon Specific

- **No bitsandbytes**: 4-bit quantization not needed with 64GB unified memory
- **No Flash Attention**: Uses standard attention (slightly slower but works)
- **Memory monitoring**: Use `sudo powermetrics --samplers gpu_power` to monitor GPU usage
- **Thermal throttling**: Ensure adequate cooling for sustained training

---

*DPO Branch - Bob Loukas Mindprint RLHF LoRA*
*Target Hardware: Mac Studio M2 Ultra (64GB)*

