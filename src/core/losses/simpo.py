"""
SimPO: Simple Preference Optimization.

Paper: https://arxiv.org/abs/2405.14734

Length-normalized preference optimization that doesn't require
a reference model. Often outperforms DPO on benchmarks.
"""

from dataclasses import dataclass
from typing import Any

from .base import BaseLoss, LossOutput


@dataclass
class SimPOConfig:
    """SimPO configuration."""
    beta: float = 2.0  # Preference strength
    gamma: float = 0.5  # Target margin between chosen/rejected


class SimPOLoss(BaseLoss):
    """
    Simple Preference Optimization loss.
    
    SimPO improves on DPO by:
    - Length normalization prevents length bias
    - No reference model needed (faster training)
    - Target margin (gamma) improves separation
    
    Key innovations:
    - Length-normalized rewards: logp / length
    - Margin loss: encourages chosen > rejected + gamma
    - Simpler than DPO, often better results
    """
    
    def __init__(self, config: SimPOConfig):
        """
        Initialize SimPO loss.
        
        Args:
            config: SimPO configuration
        """
        self.config = config
    
    @property
    def requires_reference_model(self) -> bool:
        """SimPO does not require reference model."""
        return False
    
    def compute(
        self,
        policy_chosen_logps: Any,
        policy_rejected_logps: Any,
        chosen_lengths: Any,
        rejected_lengths: Any,
        **kwargs
    ) -> LossOutput:
        """
        Compute SimPO loss.
        
        Args:
            policy_chosen_logps: Policy model log probs for chosen
            policy_rejected_logps: Policy model log probs for rejected
            chosen_lengths: Sequence lengths for chosen responses
            rejected_lengths: Sequence lengths for rejected responses
        
        Returns:
            LossOutput with SimPO loss and metrics
        """
        # Length-normalized rewards
        chosen_rewards = policy_chosen_logps / chosen_lengths
        rejected_rewards = policy_rejected_logps / rejected_lengths
        
        # SimPO loss with margin
        # Loss = -log(sigmoid(beta * (chosen_reward - rejected_reward - gamma)))
        logits = self.config.beta * (
            chosen_rewards - rejected_rewards - self.config.gamma
        )
        
        # Compute loss
        import numpy as np
        if hasattr(logits, 'item'):  # MLX array
            loss = -np.logaddexp(0, -logits).mean()
            reward_margin = (chosen_rewards - rejected_rewards).mean().item()
            accuracy = (chosen_rewards > rejected_rewards).mean().item()
            chosen_reward_mean = chosen_rewards.mean().item()
            rejected_reward_mean = rejected_rewards.mean().item()
        else:  # PyTorch tensor
            import torch
            loss = -torch.nn.functional.logsigmoid(logits).mean()
            reward_margin = (chosen_rewards - rejected_rewards).mean().item()
            accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
            chosen_reward_mean = chosen_rewards.mean().item()
            rejected_reward_mean = rejected_rewards.mean().item()
        
        return LossOutput(
            loss=loss,
            metrics={
                "simpo_loss": loss.item() if hasattr(loss, 'item') else float(loss),
                "reward_margin": reward_margin,
                "accuracy": accuracy,
                "chosen_reward": chosen_reward_mean,
                "rejected_reward": rejected_reward_mean,
                "length_ratio": (chosen_lengths.mean() / rejected_lengths.mean()).item() if hasattr(chosen_lengths, 'item') else float(chosen_lengths.mean() / rejected_lengths.mean()),
            }
        )
