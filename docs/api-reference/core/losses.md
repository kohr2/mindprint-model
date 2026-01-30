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

## DPOLoss

Direct Preference Optimization loss.

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

## SimPOLoss

Simple Preference Optimization loss.

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

## ORPOLoss

Odds Ratio Preference Optimization loss.

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
