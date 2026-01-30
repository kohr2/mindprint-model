# Loss Functions

## Overview

Mindprint Model supports multiple state-of-the-art preference learning loss functions:

- **DPO**: Direct Preference Optimization (standard)
- **SimPO**: Simple Preference Optimization (length-normalized, no reference)
- **ORPO**: Odds Ratio Preference Optimization (combined SFT+alignment)

## DPO (Direct Preference Optimization)

**Paper**: https://arxiv.org/abs/2305.18290

Standard DPO using Bradley-Terry model with KL regularization.

**Pros:**
- Well-established, widely used
- KL regularization prevents mode collapse
- Good for production systems

**Cons:**
- Requires reference model (slower training)
- Can be sensitive to hyperparameters

**When to use:**
- When you need proven stability
- When you have compute for reference model

## SimPO (Simple Preference Optimization)

**Paper**: https://arxiv.org/abs/2405.14734

Length-normalized preference optimization without reference model.

**Pros:**
- No reference model needed (2x faster)
- Length normalization prevents bias
- Often outperforms DPO on benchmarks
- Target margin improves separation

**Cons:**
- Newer technique (less battle-tested)
- Requires careful hyperparameter tuning

**When to use:**
- When training speed matters
- When you want best quality
- Recommended for most use cases

## ORPO (Odds Ratio Preference Optimization)

**Paper**: https://arxiv.org/abs/2403.07691

Combines SFT and preference alignment in single stage.

**Pros:**
- Single-stage training (2x faster than SFT+DPO)
- No reference model needed
- Better instruction following than DPO

**Cons:**
- More complex implementation
- Less control over SFT vs alignment balance

**When to use:**
- When starting from scratch
- When you want fastest training
- When instruction following is priority

## Comparison

| Feature | DPO | SimPO | ORPO |
|---------|-----|-------|------|
| Reference Model | Required | Not needed | Not needed |
| Training Speed | Baseline | 2x faster | 2x faster |
| Length Normalized | No | Yes | No |
| Single Stage | No (SFT+DPO) | No (SFT+DPO) | Yes |
| Best Quality | Good | Excellent | Excellent |

## Choosing a Loss Function

**Start with SimPO** - Best balance of quality and speed.

Switch to DPO if:
- You need maximum stability
- You have reference model compute

Switch to ORPO if:
- You're training from scratch
- Instruction following is critical
