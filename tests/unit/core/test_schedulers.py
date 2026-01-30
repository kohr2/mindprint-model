"""
Unit tests for learning rate schedulers.
"""

import pytest

from src.core.schedulers import CosineScheduler, CosineSchedulerConfig


def test_cosine_scheduler_warmup():
    """Test cosine scheduler warmup phase."""
    scheduler = CosineScheduler(
        base_lr=1e-3,
        total_steps=1000,
        config=CosineSchedulerConfig(warmup_ratio=0.1)
    )
    
    # First step should be very small
    lr_0 = scheduler.get_lr(0)
    assert lr_0 > 0
    assert lr_0 < scheduler.base_lr
    
    # Middle of warmup
    lr_50 = scheduler.get_lr(50)
    assert lr_50 > lr_0
    assert lr_50 < scheduler.base_lr
    
    # End of warmup should be base_lr
    lr_100 = scheduler.get_lr(100)
    assert abs(lr_100 - scheduler.base_lr) < 1e-6


def test_cosine_scheduler_decay():
    """Test cosine scheduler decay phase."""
    scheduler = CosineScheduler(
        base_lr=1e-3,
        total_steps=1000,
        config=CosineSchedulerConfig(warmup_ratio=0.1, min_lr_ratio=0.1)
    )
    
    # After warmup, LR should decay
    lr_500 = scheduler.get_lr(500)
    lr_900 = scheduler.get_lr(900)
    
    assert lr_500 < scheduler.base_lr
    assert lr_900 < lr_500
    assert lr_900 >= scheduler.min_lr


def test_cosine_scheduler_final_lr():
    """Test that final LR equals min_lr."""
    scheduler = CosineScheduler(
        base_lr=1e-3,
        total_steps=1000,
        config=CosineSchedulerConfig(warmup_ratio=0.1, min_lr_ratio=0.1)
    )
    
    final_lr = scheduler.get_lr(999)
    assert abs(final_lr - scheduler.min_lr) < 1e-4
