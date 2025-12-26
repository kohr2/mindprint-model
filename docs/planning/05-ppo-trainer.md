# PPO Trainer Implementation

## Objective

Implement Proximal Policy Optimization (PPO) with Rank-1 LoRA, using the trained reward model to guide voice alignment.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       PPO TRAINING                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐                      ┌──────────────┐   │
│   │   POLICY     │──── generates ──────▶│   RESPONSE   │   │
│   │   (LoRA)     │                      │              │   │
│   └──────────────┘                      └──────┬───────┘   │
│          │                                     │           │
│          │                                     ▼           │
│   ┌──────────────┐                      ┌──────────────┐   │
│   │  REFERENCE   │──── KL penalty ─────▶│   REWARD     │   │
│   │   (frozen)   │                      │    MODEL     │   │
│   └──────────────┘                      └──────┬───────┘   │
│                                                │           │
│                              ┌─────────────────┘           │
│                              ▼                              │
│                    ┌──────────────┐                         │
│                    │  R - β*KL    │──── policy update       │
│                    └──────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Implementation

```python
# src/ppo_trainer.py

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig

def create_ppo_trainer(sft_model, reward_model, tokenizer):
    """Create PPO trainer with Rank-1 LoRA."""
    
    # Add value head and LoRA
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        sft_model,
        peft_config=LoraConfig(r=1, lora_alpha=1.0)
    )
    
    ppo_config = PPOConfig(
        learning_rate=1.41e-5,
        batch_size=4,
        ppo_epochs=4,
        init_kl_coef=0.2,
        target_kl=6.0,
    )
    
    return PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
    ), reward_model


def train_step(trainer, reward_model, prompts):
    """Single PPO training step."""
    
    # Generate responses
    query_tensors = tokenize(prompts)
    response_tensors = trainer.generate(query_tensors)
    
    # Get rewards
    responses = decode(response_tensors)
    rewards = [reward_model.score(p, r) for p, r in zip(prompts, responses)]
    
    # PPO update
    stats = trainer.step(query_tensors, response_tensors, rewards)
    return stats
```

## Configuration

```yaml
# configs/ppo.yaml

ppo:
  learning_rate: 1.41e-5
  batch_size: 4
  ppo_epochs: 4
  init_kl_coef: 0.2
  target_kl: 6.0
  cliprange: 0.2
  steps_per_topic: 50

lora:
  rank: 1
  alpha: 1.0
  target_modules: ["o_proj", "v_proj", "up_proj", "down_proj"]
```

## Key Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `init_kl_coef` | 0.2 | KL penalty strength |
| `target_kl` | 6.0 | Target KL divergence |
| `cliprange` | 0.2 | Policy clip range |

---

*PPO Branch - Bob Loukas Mindprint RLHF LoRA*

