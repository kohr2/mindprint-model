# DPO Training Execution

## Pre-Training Checklist

```bash
# 1. Environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

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
    
    # Load model (4-bit quantized)
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-12b",
        load_in_4bit=True,
        device_map="auto",
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
  quantization: "4bit"

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
  batch_size: 4

evaluation:
  topic_threshold: 0.90
  voice_threshold: 0.75
```

## Expected Timeline

| Phase | Est. Time |
|-------|-----------|
| Data prep | 1 hour |
| Unit 1 | 8 hours |
| Unit 2 | 10 hours |
| Unit 3 | 8 hours |
| Unit 4 | 8 hours |
| **Total** | **~35 hours** |

## Monitoring

Key metrics to track:
- `accuracy`: Quiz accuracy per topic
- `voice_score`: Voice fidelity score
- `dpo_loss`: DPO loss (should decrease)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM | Reduce batch_size |
| Voice stuck low | Increase dpo.max_steps |
| Accuracy drops | Enable merge_after_chapter |

---

*DPO Branch - Bob Loukas Mindprint RLHF LoRA*

