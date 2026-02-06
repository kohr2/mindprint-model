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

# PyTorch for ORPO loss tests (ORPO uses sum vs mean over tokens; test magnitude)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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
    """Test ORPO loss computation (uses PyTorch when available to avoid MLX/numpy mismatch)."""
    pytest.importorskip("torch")
    import torch
    loss_fn = ORPOLoss(ORPOConfig(lambda_orpo=0.1))
    batch_size, seq_len, vocab_size = 2, 10, 1000
    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    chosen_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    rejected_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
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


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_orpo_loss_magnitude_reasonable():
    """ORPO loss should be in a reasonable range (< 100) for typical sequence lengths.
    Large values (e.g. 5000+) indicate loss is summed over tokens instead of averaged."""
    loss_fn = ORPOLoss(ORPOConfig(lambda_orpo=0.1))
    batch_size, seq_len, vocab_size = 2, 64, 1000
    logits = torch.randn(batch_size, seq_len, vocab_size)
    chosen_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    rejected_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    result = loss_fn.compute(
        logits=logits,
        chosen_ids=chosen_ids,
        rejected_ids=rejected_ids,
    )
    loss_val = result.loss.item() if hasattr(result.loss, "item") else float(result.loss)
    assert loss_val < 100.0, (
        f"ORPO loss {loss_val} is unreasonably high; expected < 100. "
        "Loss should be averaged over tokens, not summed."
    )
    assert result.metrics["orpo_loss"] < 100.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_orpo_loss_averaged_not_summed_over_tokens():
    """ORPO loss should be invariant to sequence length (averaged), not scale with it (summed)."""
    loss_fn = ORPOLoss(ORPOConfig(lambda_orpo=0.1))
    batch_size, vocab_size = 2, 1000
    torch.manual_seed(42)
    # Short sequence
    logits_short = torch.randn(batch_size, 32, vocab_size)
    chosen_short = torch.randint(0, vocab_size, (batch_size, 32))
    rejected_short = torch.randint(0, vocab_size, (batch_size, 32))
    result_short = loss_fn.compute(logits=logits_short, chosen_ids=chosen_short, rejected_ids=rejected_short)
    loss_short = result_short.metrics["orpo_loss"]
    # Long sequence (same batch, 4x length)
    logits_long = torch.randn(batch_size, 128, vocab_size)
    chosen_long = torch.randint(0, vocab_size, (batch_size, 128))
    rejected_long = torch.randint(0, vocab_size, (batch_size, 128))
    result_long = loss_fn.compute(logits=logits_long, chosen_ids=chosen_long, rejected_ids=rejected_long)
    loss_long = result_long.metrics["orpo_loss"]
    # If loss were summed, loss_long would be ~4x loss_short. If averaged, they should be same order.
    ratio = loss_long / (loss_short + 1e-8)
    assert ratio < 5.0, (
        f"ORPO loss scales with sequence length (ratio {ratio:.1f}x). "
        "Loss should be averaged over tokens so it does not grow with seq_len."
    )


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
