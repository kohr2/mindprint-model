"""
Learning rate schedulers.

Implements various LR scheduling strategies:
- Cosine annealing with warmup
- Linear warmup
- One-cycle policy
"""

from .cosine import CosineScheduler, CosineSchedulerConfig

__all__ = ["CosineScheduler", "CosineSchedulerConfig"]
