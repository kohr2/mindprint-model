"""
Unit tests for loss functions.

Tests DPO, SimPO, and ORPO implementations.
"""

import pytest
import numpy as np

from src.core.losses import (
    DPOLoss,
    SimPOLoss,
    ORPOLoss,
    DPOConfig,
    SimPOConfig,
    ORPOConfig,
)


@pytest.fixture
def sample_logps():
    """Create sample log probabilities."""
    batch_size = 4
    return {
        "policy_chosen": np.random.randn(batch_size),
        "policy_rejected": np.random.randn(batch_size) - 0.5,
        "ref_chosen": np.random.randn(batch_size),
        "ref_rejected": np.random.randn(batch_size) - 0.5,
        "chosen_lengths": np.ones(batch_size) * 100,
        "rejected_lengths": np.ones(batch_size) * 80,
    }


def test_dpo_loss_computation(sample_logps):
    """Test DPO loss computation."""
    loss_fn = DPOLoss(DPOConfig(beta=0.1))
    
    result = loss_fn.compute(
        policy_chosen_logps=sample_logps["policy_chosen"],
        policy_rejected_logps=sample_logps["policy_rejected"],
        ref_chosen_logps=sample_logps["ref_chosen"],
        ref_rejected_logps=sample_logps["ref_rejected"],
    )
    
    assert result.loss is not None
    assert "dpo_loss" in result.metrics
    assert "accuracy" in result.metrics
    assert result.metrics["accuracy"] >= 0.0
    assert result.metrics["accuracy"] <= 1.0


def test_simpo_loss_computation(sample_logps):
    """Test SimPO loss computation."""
    loss_fn = SimPOLoss(SimPOConfig(beta=2.0, gamma=0.5))
    
    result = loss_fn.compute(
        policy_chosen_logps=sample_logps["policy_chosen"],
        policy_rejected_logps=sample_logps["policy_rejected"],
        chosen_lengths=sample_logps["chosen_lengths"],
        rejected_lengths=sample_logps["rejected_lengths"],
    )
    
    assert result.loss is not None
    assert "simpo_loss" in result.metrics
    assert "reward_margin" in result.metrics
    assert not loss_fn.requires_reference_model


def test_orpo_loss_computation():
    """Test ORPO loss computation."""
    loss_fn = ORPOLoss(ORPOConfig(lambda_orpo=0.1))
    
    # Create mock logits and token IDs
    batch_size, seq_len, vocab_size = 2, 10, 1000
    logits = np.random.randn(batch_size, seq_len, vocab_size)
    chosen_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    rejected_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    
    result = loss_fn.compute(
        logits=logits,
        chosen_ids=chosen_ids,
        rejected_ids=rejected_ids,
    )
    
    assert result.loss is not None
    assert "orpo_loss" in result.metrics
    assert "nll_loss" in result.metrics
    assert "or_loss" in result.metrics
    assert not loss_fn.requires_reference_model


def test_loss_output_structure(sample_logps):
    """Test that loss outputs have correct structure."""
    loss_fn = DPOLoss(DPOConfig())
    result = loss_fn.compute(
        policy_chosen_logps=sample_logps["policy_chosen"],
        policy_rejected_logps=sample_logps["policy_rejected"],
        ref_chosen_logps=sample_logps["ref_chosen"],
        ref_rejected_logps=sample_logps["ref_rejected"],
    )
    
    assert hasattr(result, "loss")
    assert hasattr(result, "metrics")
    assert isinstance(result.metrics, dict)
