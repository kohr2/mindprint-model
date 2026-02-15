# Loss Functions

## ORPO (Odds Ratio Preference Optimization) — Default

**Paper**: https://arxiv.org/abs/2403.07691

Combines SFT and preference alignment in a single training stage.

```
L_ORPO = L_NLL + lambda * L_OR
```

- `L_NLL`: Standard next-token prediction (SFT component)
- `L_OR`: Odds ratio between chosen and rejected responses
- `lambda`: Balance parameter (default: 0.1)

**Why ORPO**:
- Single-stage (no separate SFT then DPO phases)
- No reference model needed
- ~2x faster than SFT+DPO pipelines
- Better instruction following than DPO

**Key parameters**:

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `lambda_orpo` | 0.1 | 0.05–0.5 | Lower = more SFT focus, higher = more alignment |
| `learning_rate` | 3e-4 | 1e-4–5e-4 | Higher than DPO (single stage) |
| `steps_per_topic` | 100 | 50–200 | Training steps per curriculum topic |

## Other Loss Functions (Available in Core)

### SimPO (Simple Preference Optimization)

**Paper**: https://arxiv.org/abs/2405.14734

Length-normalized preference optimization. No reference model. Requires separate SFT phase.

### DPO (Direct Preference Optimization)

**Paper**: https://arxiv.org/abs/2305.18290

Standard Bradley-Terry preference optimization. Requires reference model and separate SFT phase.

## Comparison

| Feature | ORPO | SimPO | DPO |
|---------|------|-------|-----|
| Single stage | Yes | No | No |
| Reference model | Not needed | Not needed | Required |
| Training speed | Fastest | Fast | Baseline |
| Separate SFT | No | Yes | Yes |
