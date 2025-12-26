# Bob Loukas Mindprint RLHF LoRA - Project Overview

## Executive Summary

This project implements RLHF-enhanced LoRA fine-tuning for the Bob Loukas mindprint, a Bitcoin cycle trading expert persona. The implementation builds on existing infrastructure from the Rank-1 ReLoRA Mindprinting Cookbook.

**Two approaches are available:**
- `dpo` branch: Direct Preference Optimization (simpler, faster)
- `ppo` branch: Full RLHF with reward model (richer signal, more compute)

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
| 2 | Voice Evaluator | Build voice fidelity scoring system | shared |
| 3 | Model Selection | Choose and configure base model | shared |
| - | **Training Approach** | DPO or PPO implementation | dpo/ppo |
| - | **Pipeline Integration** | Combine SFT + approach | dpo/ppo |
| - | **Training Execution** | Train and evaluate | dpo/ppo |

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
| Voice Evaluator | 2-3 days | Day 6 |
| Model Setup | 1 day | Day 7 |
| **DPO Training** | 5-7 days | Day 14 |
| **PPO Training** | 7-10 days | Day 17 |

## Resources Required

- **Compute**: 1x A100 40GB or 2x RTX 4090 (for 4-bit training)
- **Storage**: ~100GB for model checkpoints and logs
- **Source Data**: Bob Loukas textbook (already complete in omnia)

## Branch Structure

```
shared (this branch)
├── 00-overview.md          # This file
├── 01-data-preparation.md  # Preference pair generation
├── 02-voice-evaluator.md   # Voice fidelity scoring
└── 03-model-selection.md   # Model analysis

dpo (extends shared)
├── 01-dpo-trainer.md       # DPO implementation
├── 02-pipeline.md          # SFT + DPO pipeline
└── 03-training.md          # DPO training execution

ppo (extends shared)
├── 01-reward-model.md      # Reward model training
├── 02-ppo-trainer.md       # PPO implementation
├── 03-pipeline.md          # SFT + PPO pipeline
└── 04-training.md          # PPO training execution
```

## Related Documents (This Branch)

- [Phase 1: Data Preparation](./01-data-preparation.md)
- [Phase 2: Voice Evaluator](./02-voice-evaluator.md)
- [Phase 3: Model Selection](./03-model-selection.md)

---

*Created: December 2025*
*Project: Memetica Corporation - Mindprint Agent*
*Branch: shared (common foundation)*
