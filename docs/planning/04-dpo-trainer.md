# DPO Trainer Implementation

## Objective

Implement Direct Preference Optimization (DPO) with Rank-1 LoRA for efficient preference alignment without requiring a separate reward model.

## Why DPO?

| Aspect | DPO | PPO |
|--------|-----|-----|
| Complexity | Simple, single-stage | Complex, needs reward model |
| Stability | More stable | Can be unstable |
| Data efficiency | Works with ~500 pairs | Needs more data |
| Compute | Lower | ~2x higher |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DPO TRAINING FLOW                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐                      ┌──────────────┐   │
│   │   POLICY     │                      │  REFERENCE   │   │
│   │   MODEL      │                      │    MODEL     │   │
│   │   (LoRA)     │                      │   (frozen)   │   │
│   └──────┬───────┘                      └──────┬───────┘   │
│          │                                     │           │
│          ▼                                     ▼           │
│   ┌────────────────────────────────────────────────────┐  │
│   │              DPO LOSS                               │  │
│   │   L = -log σ(β(log π/π_ref(chosen) -               │  │
│   │              log π/π_ref(rejected)))               │  │
│   └────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Implementation

```python
# src/dpo_trainer.py

from dataclasses import dataclass
from typing import List, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig


@dataclass
class Rank1DPOConfig:
    """DPO configuration."""
    beta: float = 0.1
    learning_rate: float = 5e-7
    max_steps: int = 100
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4


class Rank1DPOTrainer:
    """Rank-1 LoRA trainer with DPO."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Rank1DPOConfig
    ):
        self.base_model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Create Rank-1 LoRA adapter
        lora_config = LoraConfig(
            r=1,
            lora_alpha=1.0,
            target_modules=["o_proj", "v_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM"
        )
        self.peft_model = get_peft_model(model, lora_config)
    
    def train(self, preference_pairs: List[dict]):
        """Train with DPO on preference pairs."""
        
        dpo_config = DPOConfig(
            beta=self.config.beta,
            learning_rate=self.config.learning_rate,
            max_steps=self.config.max_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            output_dir="./dpo_output",
        )
        
        trainer = DPOTrainer(
            model=self.peft_model,
            ref_model=self.base_model,
            args=dpo_config,
            train_dataset=self._format_dataset(preference_pairs),
            tokenizer=self.tokenizer,
        )
        
        trainer.train()
        return self.peft_model
    
    def _format_dataset(self, pairs):
        """Format preference pairs for DPO."""
        from datasets import Dataset
        return Dataset.from_list(pairs)
    
    def merge(self):
        """Merge LoRA into base model."""
        return self.peft_model.merge_and_unload()
```

## Configuration

```yaml
# configs/dpo.yaml

dpo:
  beta: 0.1                    # KL penalty coefficient
  learning_rate: 5e-7          # Lower than SFT
  max_steps: 100               # Steps per topic
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4

lora:
  rank: 1
  alpha: 1.0
  target_modules:
    - "o_proj"
    - "v_proj"
    - "up_proj"
    - "down_proj"
```

## Beta Selection Guide

| Beta | Effect | Use Case |
|------|--------|----------|
| 0.01 | High deviation | Strong alignment |
| 0.1 | Balanced (default) | General use |
| 0.5 | Conservative | Preserve base |

## Training Protocol

```python
for topic in topics:
    # 1. SFT pre-training
    trainer.train_sft(topic.content, epochs=3)
    
    # 2. Evaluate
    accuracy = evaluator.evaluate(topic.quiz)
    
    # 3. DPO if voice needs work
    if accuracy >= 0.70 and voice_score < 0.75:
        trainer.train_dpo(topic.preference_pairs)
    
    # 4. Re-evaluate
    final_score = evaluator.evaluate(topic.quiz)
```

## Dependencies

```
trl>=0.8.0
peft>=0.10.0
```

---

*DPO Branch - Bob Loukas Mindprint RLHF LoRA*

