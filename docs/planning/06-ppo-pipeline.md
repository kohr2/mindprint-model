# PPO Pipeline Integration

## Objective

Combine SFT, reward model, and PPO into a unified training pipeline.

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    PPO TRAINING PIPELINE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Pre-training (once):                                      │
│   ┌─────────────────────────────────────────────────────┐  │
│   │ Train Reward Model on preference pairs (~6 hours)   │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   For each topic:                                           │
│   ┌─────────────────────────────────────────────────────┐  │
│   │ 1. SFT Training (3 epochs)                          │  │
│   │                    ↓                                 │  │
│   │ 2. Quiz Evaluation                                   │  │
│   │                    ↓                                 │  │
│   │ 3. PPO Refinement with reward model (50 steps)      │  │
│   │                    ↓                                 │  │
│   │ 4. Final Evaluation                                  │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   After each unit:                                          │
│   ┌─────────────────────────────────────────────────────┐  │
│   │ Merge LoRA → Verify → Continue                      │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Implementation

```python
# src/ppo_pipeline.py

class PPOPipeline:
    """SFT + PPO training pipeline with reward model."""
    
    def __init__(self, model, tokenizer, reward_model, config):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.config = config
    
    def train_topic(self, topic):
        """Train single topic with SFT + PPO."""
        
        # 1. SFT
        self.train_sft(topic["content"])
        
        # 2. Evaluate
        result = self.evaluate(topic["quiz"])
        
        # 3. PPO if needed
        if result["voice_score"] < 0.75:
            self.train_ppo(topic["prompts"], steps=50)
            result = self.evaluate(topic["quiz"])
        
        return result
    
    def train_ppo(self, prompts, steps):
        """PPO training with reward model."""
        for step in range(steps):
            # Generate
            responses = self.generate(prompts)
            
            # Reward
            rewards = self.reward_model.score_batch(prompts, responses)
            
            # Update
            self.ppo_step(prompts, responses, rewards)
```

## Configuration

```yaml
# configs/ppo_pipeline.yaml

reward_model:
  path: "./reward_model"

training:
  sft_epochs: 3
  ppo_steps_per_topic: 50
  ppo_trigger_threshold: 0.75

merge:
  merge_after_unit: true
```

## Timeline

| Phase | Est. Time |
|-------|-----------|
| Reward Model | 6 hours |
| Unit 1 | 10 hours |
| Unit 2 | 12 hours |
| Unit 3 | 10 hours |
| Unit 4 | 10 hours |
| **Total** | **~48 hours** |

---

*PPO Branch - Bob Loukas Mindprint RLHF LoRA*

