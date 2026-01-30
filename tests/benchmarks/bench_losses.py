"""
Benchmark loss function implementations.

Compares computation speed and memory usage.
"""

import pytest
import numpy as np

from src.core.losses import DPOLoss, SimPOLoss, DPOConfig, SimPOConfig


@pytest.fixture
def sample_batch():
    """Create sample batch for benchmarking."""
    batch_size = 8
    return {
        "policy_chosen_logps": np.random.randn(batch_size),
        "policy_rejected_logps": np.random.randn(batch_size) - 0.5,
        "ref_chosen_logps": np.random.randn(batch_size),
        "ref_rejected_logps": np.random.randn(batch_size) - 0.5,
        "chosen_lengths": np.ones(batch_size) * 512,
        "rejected_lengths": np.ones(batch_size) * 400,
    }


@pytest.mark.benchmark(group="losses")
def test_dpo_loss_speed(benchmark, sample_batch):
    """Benchmark DPO loss computation."""
    loss_fn = DPOLoss(DPOConfig(beta=0.1))
    
    def compute():
        return loss_fn.compute(
            policy_chosen_logps=sample_batch["policy_chosen_logps"],
            policy_rejected_logps=sample_batch["policy_rejected_logps"],
            ref_chosen_logps=sample_batch["ref_chosen_logps"],
            ref_rejected_logps=sample_batch["ref_rejected_logps"],
        )
    
    result = benchmark(compute)
    assert result.loss is not None


@pytest.mark.benchmark(group="losses")
def test_simpo_loss_speed(benchmark, sample_batch):
    """Benchmark SimPO loss computation."""
    loss_fn = SimPOLoss(SimPOConfig(beta=2.0, gamma=0.5))
    
    def compute():
        return loss_fn.compute(
            policy_chosen_logps=sample_batch["policy_chosen_logps"],
            policy_rejected_logps=sample_batch["policy_rejected_logps"],
            chosen_lengths=sample_batch["chosen_lengths"],
            rejected_lengths=sample_batch["rejected_lengths"],
        )
    
    result = benchmark(compute)
    assert result.loss is not None
