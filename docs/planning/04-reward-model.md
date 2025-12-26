# Reward Model Training

## Objective

Train a reward model that scores responses based on Bob Loukas's voice, factual accuracy, and teaching style. This reward model guides PPO optimization.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      REWARD MODEL                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input: (prompt, response)                                 │
│                              │                              │
│                              ▼                              │
│   ┌─────────────────────────────────────────────────────┐  │
│   │         BASE LM ENCODER (Gemma-2-2B)                │  │
│   └─────────────────────────────────────────────────────┘  │
│                              │                              │
│                              ▼                              │
│   ┌─────────────────────────────────────────────────────┐  │
│   │           REWARD HEAD (Linear → 1)                  │  │
│   └─────────────────────────────────────────────────────┘  │
│                              │                              │
│                              ▼                              │
│   Output: scalar reward                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Training Data

Uses same preference pairs as DPO, plus additional synthetic pairs:

| Source | Pairs | Purpose |
|--------|-------|---------|
| Bob's answers vs generic | ~520 | Voice learning |
| Bob's answers vs base model | ~520 | Realistic contrast |
| Critical misconceptions | ~100 | Halving vs cycle |
| **Total** | ~1140 | |

## Implementation

```python
# src/reward_model.py

from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification

def train_reward_model(preference_pairs, output_dir):
    """Train reward model on preference pairs."""
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "google/gemma-2-2b",
        num_labels=1
    )
    
    config = RewardConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=1e-5,
    )
    
    trainer = RewardTrainer(
        model=model,
        args=config,
        train_dataset=format_dataset(preference_pairs),
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    return model
```

## Validation Criteria

| Metric | Target |
|--------|--------|
| Training accuracy | ≥95% |
| Validation accuracy | ≥90% |
| Critical pairs | 100% |

## Training Time

~4-6 hours on A100

---

*PPO Branch - Bob Loukas Mindprint RLHF LoRA*

