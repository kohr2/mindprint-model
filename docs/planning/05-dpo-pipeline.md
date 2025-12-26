# DPO Pipeline Integration

## Objective

Combine SFT and DPO into a unified training pipeline with voice evaluation gates.

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    DPO TRAINING PIPELINE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   For each topic:                                           │
│   ┌─────────────────────────────────────────────────────┐  │
│   │ 1. SFT Training (3 epochs on Q&A content)           │  │
│   │                    ↓                                 │  │
│   │ 2. Quiz Evaluation (accuracy + voice)               │  │
│   │                    ↓                                 │  │
│   │ 3. DPO Refinement (if voice < 0.75)                 │  │
│   │                    ↓                                 │  │
│   │ 4. Final Evaluation                                  │  │
│   │                    ↓                                 │  │
│   │ 5. Pass/Retry Decision                               │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   After each unit:                                          │
│   ┌─────────────────────────────────────────────────────┐  │
│   │ Merge LoRA → Verify no regression → Continue        │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Implementation

```python
# src/dpo_pipeline.py

from src.dpo_trainer import Rank1DPOTrainer
from src.voice_evaluator import QuizEvaluator


class DPOPipeline:
    """SFT + DPO training pipeline."""
    
    def __init__(self, model, tokenizer, config):
        self.trainer = Rank1DPOTrainer(model, tokenizer, config)
        self.evaluator = QuizEvaluator(model, tokenizer)
        self.model = model
    
    def train_topic(self, topic):
        """Train a single topic with SFT + optional DPO."""
        
        # 1. SFT
        self.trainer.train_sft(topic["content"])
        
        # 2. Evaluate
        result = self.evaluator.evaluate(topic["quiz"])
        
        # 3. DPO if needed
        if result["accuracy"] >= 0.70 and result["voice_score"] < 0.75:
            self.trainer.train_dpo(topic["preference_pairs"])
            result = self.evaluator.evaluate(topic["quiz"])
        
        return result
    
    def train_curriculum(self, textbook_path):
        """Train full curriculum."""
        
        for unit in self._load_units(textbook_path):
            for chapter in unit["chapters"]:
                for topic in chapter["topics"]:
                    result = self.train_topic(topic)
                    
                    if not result["passed"]:
                        self.failed_topics.append(topic)
            
            # Merge after unit
            self.model = self.trainer.merge()
```

## Configuration

```yaml
# configs/dpo_pipeline.yaml

training:
  sft_epochs_per_topic: 3
  dpo_steps_per_topic: 100
  dpo_trigger_threshold: 0.75

evaluation:
  topic_pass_threshold: 0.90
  chapter_pass_threshold: 0.85
  unit_pass_threshold: 0.80

merge:
  merge_after_unit: true
  verify_after_merge: true
```

## Timeline

| Phase | Duration |
|-------|----------|
| Unit 1 | 6-8 hours |
| Unit 2 | 8-10 hours |
| Unit 3 | 6-8 hours |
| Unit 4 | 6-8 hours |
| **Total** | **~34 hours** |

---

*DPO Branch - Bob Loukas Mindprint RLHF LoRA*

