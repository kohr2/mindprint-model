# Loss Functions API

## BaseLoss

Abstract base class for all preference learning losses.

```python
from src.core.losses import BaseLoss

class MyLoss(BaseLoss):
    def compute(self, ...) -> LossOutput:
        ...

    @property
    def requires_reference_model(self) -> bool:
        ...
```

## ORPOLoss

Odds Ratio Preference Optimization loss. Combines SFT and alignment in single stage.

```python
from src.core.losses import ORPOLoss, ORPOConfig

config = ORPOConfig(lambda_orpo=0.1)
loss_fn = ORPOLoss(config)

result = loss_fn.compute(
    logits=...,
    chosen_ids=...,
    rejected_ids=...,
)
```

## SimPOLoss

Simple Preference Optimization loss. Length-normalized, no reference model.

```python
from src.core.losses import SimPOLoss, SimPOConfig

config = SimPOConfig(beta=2.0, gamma=0.5)
loss_fn = SimPOLoss(config)

result = loss_fn.compute(
    policy_chosen_logps=...,
    policy_rejected_logps=...,
    chosen_lengths=...,
    rejected_lengths=...,
)
```

## DPOLoss

Direct Preference Optimization loss. Requires reference model.

```python
from src.core.losses import DPOLoss, DPOConfig

config = DPOConfig(beta=0.1)
loss_fn = DPOLoss(config)

result = loss_fn.compute(
    policy_chosen_logps=...,
    policy_rejected_logps=...,
    ref_chosen_logps=...,
    ref_rejected_logps=...,
)
```
