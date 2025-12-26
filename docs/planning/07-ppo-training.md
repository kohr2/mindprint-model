# PPO Training Execution

## Pre-Training Steps

```bash
# 1. Train reward model first
python scripts/train_reward_model.py \
    --data ./data/bob_loukas/preference_data.jsonl \
    --output ./reward_model

# 2. Verify reward model
python scripts/evaluate_reward_model.py --model ./reward_model
```

## Training Script

```python
#!/usr/bin/env python3
# scripts/train_ppo.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.ppo_pipeline import PPOPipeline
from src.reward_model import load_reward_model

def main():
    # Load models
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-12b",
        load_in_4bit=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b")
    reward_model = load_reward_model("./reward_model")
    
    # Train
    pipeline = PPOPipeline(model, tokenizer, reward_model)
    pipeline.train_curriculum("../omnia/projects/bob_loukas/textbook")
    
    # Save
    pipeline.save("./output/bob_loukas_ppo")

if __name__ == "__main__":
    main()
```

## Configuration

```yaml
# configs/ppo.yaml

model:
  name: "google/gemma-3-12b"
  quantization: "4bit"

reward_model:
  path: "./reward_model"

lora:
  rank: 1
  alpha: 1.0
  target_modules: ["o_proj", "v_proj", "up_proj", "down_proj"]

ppo:
  learning_rate: 1.41e-5
  batch_size: 4
  ppo_epochs: 4
  init_kl_coef: 0.2
  target_kl: 6.0
  steps_per_topic: 50

training:
  sft_learning_rate: 3e-4
  sft_epochs: 3

evaluation:
  topic_threshold: 0.90
  voice_threshold: 0.75
```

## Expected Timeline

| Phase | Est. Time |
|-------|-----------|
| Reward model | 6 hours |
| Data prep | 1 hour |
| Unit 1 | 10 hours |
| Unit 2 | 12 hours |
| Unit 3 | 10 hours |
| Unit 4 | 10 hours |
| **Total** | **~49 hours** |

## Monitoring

Key metrics:
- `ppo/mean_scores`: Mean reward (should increase)
- `objective/kl`: KL divergence (should stay 1-10)
- `voice_score`: Voice fidelity (target ≥0.75)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| KL too high (>10) | Increase init_kl_coef |
| Reward flat | Check reward model |
| OOM | Reduce batch_size |

---

*PPO Branch - Bob Loukas Mindprint RLHF LoRA*

