"""
Cosine annealing learning rate scheduler with warmup.

Based on: https://arxiv.org/abs/1608.03983

Standard scheduler for training large language models.
"""

import math
from dataclasses import dataclass


@dataclass
class CosineSchedulerConfig:
    """Cosine annealing with warmup configuration."""
    warmup_ratio: float = 0.1  # Fraction of steps for warmup
    min_lr_ratio: float = 0.1  # Minimum LR as fraction of base LR
    num_cycles: float = 0.5  # Number of cosine cycles


class CosineScheduler:
    """
    Cosine annealing with linear warmup.
    
    Learning rate schedule:
    1. Linear warmup from 0 to base_lr over warmup_steps
    2. Cosine decay from base_lr to min_lr over remaining steps
    
    This is the standard scheduler used in most LLM training.
    """
    
    def __init__(
        self,
        base_lr: float,
        total_steps: int,
        config: CosineSchedulerConfig = None,
    ):
        """
        Initialize cosine scheduler.
        
        Args:
            base_lr: Maximum learning rate
            total_steps: Total number of training steps
            config: Scheduler configuration (uses defaults if None)
        """
        self.config = config or CosineSchedulerConfig()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * self.config.warmup_ratio)
        self.min_lr = base_lr * self.config.min_lr_ratio
    
    def get_lr(self, step: int) -> float:
        """
        Get learning rate for a given step.
        
        Args:
            step: Current training step (0-indexed)
        
        Returns:
            Learning rate for this step
        """
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (step / self.warmup_steps)
        
        # Cosine decay
        progress = (step - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )
        cosine = math.cos(
            math.pi * self.config.num_cycles * progress
        )
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + cosine)
    
    def get_lr_at_step(self, step: int) -> float:
        """Alias for get_lr for compatibility."""
        return self.get_lr(step)
