# Bob Loukas Mindprint RLHF LoRA - PPO Approach

## Executive Summary

This branch implements **full RLHF with PPO** (Proximal Policy Optimization) for the Bob Loukas mindprint. PPO uses a learned reward model to provide richer training signals than direct preference optimization.

**Advantages of PPO:**
- Learned reward model captures nuanced voice characteristics
- Better generalization to unseen questions
- More control over KL divergence from base model
- ~48 hours training time (vs ~35 hours for DPO)

## Goals

1. **Knowledge Transfer**: Embed Bob's 4-year cycle theory, market psychology frameworks, and trading rules into a language model
2. **Voice Fidelity**: Capture Bob's distinctive communication style (confident, educational, pattern-focused)
3. **Factual Accuracy**: Ensure correct representation of Bob's key distinctions (e.g., cycle vs halving)
4. **Evaluation Framework**: Establish measurable metrics for mindprint quality

## Architecture (Shared Components)

```
┌─────────────────────────────────────────────────────────────────┐
│                    MINDPRINT TRAINING PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │   TEXTBOOK   │───▶│  DATA PREP   │───▶│ PREFERENCE   │     │
│   │   (Source)   │    │  (Convert)   │    │    PAIRS     │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                  │              │
│                                                  ▼              │
│                                     ┌────────────────────────┐  │
│                                     │   APPROACH-SPECIFIC    │  │
│                                     │   (See DPO/PPO branch) │  │
│                                     └────────────────────────┘  │
│                                                  │              │
│   ┌──────────────┐    ┌──────────────┐          │              │
│   │    VOICE     │◀───│   EVALUATE   │◀─────────┘              │
│   │   FIDELITY   │    │    (Quiz)    │                         │
│   └──────────────┘    └──────────────┘                         │
│          │                   │                                  │
│          └───────────────────┘                                  │
│                    ▼                                            │
│          ┌──────────────┐                                       │
│          │    MERGE     │                                       │
│          │  (Per Unit)  │                                       │
│          └──────────────┘                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Shared Phases

| Phase | Name | Description | Branch |
|-------|------|-------------|--------|
| 1 | Data Preparation | Convert textbook to training format + preference pairs | shared |
| 2a | Voice Evaluator | Build voice fidelity scoring system | shared |
| 2b | Evaluation Pipeline | Hierarchical quiz + voice evaluation system | shared |
| 3 | Model Selection | Choose and configure base model | shared |
| - | **Training Approach** | DPO or PPO implementation | dpo/ppo |
| - | **Pipeline Integration** | Combine SFT + approach | dpo/ppo |
| - | **Training Execution** | Train and evaluate | dpo/ppo |
| - | **Final Evaluation** | Run shared evaluation pipeline | shared |

## Model Selection

### Primary: Gemma-3-12B

**Rationale:**
- Layer architecture already documented in cookbook
- 48 layers with clear functional zones
- Good balance of capability vs compute requirements
- RoPE scaling supports 128k context

**Layer Targeting Strategy:**
- **Lexicon/Terminology** (layers 0-11): `q_proj` - Bob's specific terms
- **Reasoning/Frameworks** (layers 12-35): `v_proj` + MLP - Cycle theory
- **Voice/Style** (layers 36-47): `o_proj` + MLP - Confident, educational tone

### Alternative: Qwen2.5-7B

For resource-constrained environments (~16GB vs ~24GB VRAM).

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Topic Quiz Accuracy | ≥90% | Per-topic quiz pass rate |
| Chapter Test Accuracy | ≥85% | Chapter test pass rate |
| Unit Test Accuracy | ≥80% | Unit test pass rate |
| Voice Fidelity | ≥0.75 | Semantic similarity to reference answers |
| Halving Distinction | 100% | Never conflates cycle with halving |

## Timeline Estimate

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Data Preparation | 2-3 days | Day 3 |
| Voice Evaluator | 1-2 days | Day 5 |
| Evaluation Pipeline | 1-2 days | Day 6 |
| Model Setup | 1 day | Day 7 |
| **DPO Training** | 5-7 days | Day 14 |
| **PPO Training** | 7-10 days | Day 17 |
| **Final Evaluation** | ~2-4 hours | (per run) |

## Resources Required

- **Compute**: 1x A100 40GB or 2x RTX 4090 (for 4-bit training)
- **Storage**: ~100GB for model checkpoints and logs
- **Source Data**: Bob Loukas textbook (already complete in omnia)

## Branch Structure

```
shared (this branch)
├── 00-overview.md            # This file
├── 01-data-preparation.md    # Preference pair generation
├── 02-voice-evaluator.md     # Voice fidelity scoring
├── 02b-evaluation-pipeline.md # Hierarchical evaluation system
└── 03-model-selection.md     # Model analysis

dpo (extends shared)
├── 04-dpo-trainer.md       # DPO implementation
├── 05-dpo-pipeline.md      # SFT + DPO pipeline
└── 06-dpo-training.md      # DPO training execution

ppo (extends shared)
├── 04-reward-model.md      # Reward model training
├── 05-ppo-trainer.md       # PPO implementation
├── 06-ppo-pipeline.md      # SFT + PPO pipeline
└── 07-ppo-training.md      # PPO training execution
```

## Related Documents (This Branch)

**Shared (from base):**
- [Phase 1: Data Preparation](./01-data-preparation.md)
- [Phase 2a: Voice Evaluator](./02-voice-evaluator.md)
- [Phase 2b: Evaluation Pipeline](./02b-evaluation-pipeline.md)
- [Phase 3: Model Selection](./03-model-selection.md)

**PPO-Specific:**
- [04: Reward Model](./04-reward-model.md)
- [05: PPO Trainer](./05-ppo-trainer.md)
- [06: PPO Pipeline](./06-ppo-pipeline.md)
- [07: PPO Training](./07-ppo-training.md)

---

*Created: December 2025*
*Project: Memetica Corporation - Mindprint Agent*
*Branch: ppo (Full RLHF with Reward Model)*
