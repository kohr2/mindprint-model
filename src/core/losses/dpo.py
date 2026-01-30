"""
DPO: Direct Preference Optimization.

Paper: https://arxiv.org/abs/2305.18290

Standard DPO loss using Bradley-Terry model. Requires reference model
for KL regularization.
"""

from dataclasses import dataclass
from typing import Any

from .base import BaseLoss, LossOutput


@dataclass
class DPOConfig:
    """DPO configuration."""
    beta: float = 0.1  # KL penalty coefficient


class DPOLoss(BaseLoss):
    """
    Direct Preference Optimization loss.
    
    DPO optimizes a policy to prefer chosen over rejected responses
    while staying close to a reference model via KL regularization.
    
    Key features:
    - Requires reference model (slower training)
    - KL regularization prevents mode collapse
    - Standard in RLHF pipelines
    """
    
    def __init__(self, config: DPOConfig):
        """
        Initialize DPO loss.
        
        Args:
            config: DPO configuration
        """
        self.config = config
    
    @property
    def requires_reference_model(self) -> bool:
        """DPO requires reference model log probabilities."""
        return True
    
    def compute(
        self,
        policy_chosen_logps: Any,
        policy_rejected_logps: Any,
        ref_chosen_logps: Any,
        ref_rejected_logps: Any,
        **kwargs
    ) -> LossOutput:
        """
        Compute DPO loss.
        
        Args:
            policy_chosen_logps: Policy model log probs for chosen
            policy_rejected_logps: Policy model log probs for rejected
            ref_chosen_logps: Reference model log probs for chosen
            ref_rejected_logps: Reference model log probs for rejected
        
        Returns:
            LossOutput with DPO loss and metrics
        """
        # Compute log ratios
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        # DPO loss: -log(sigmoid(beta * (policy_logratios - ref_logratios)))
        logits = self.config.beta * (policy_logratios - ref_logratios)
        
        # Use log-sigmoid for numerical stability
        # -log(sigmoid(x)) = log(1 + exp(-x))
        import numpy as np
        if hasattr(logits, 'item'):  # MLX array
            loss = -np.logaddexp(0, -logits).mean()
            accuracy = (policy_logratios > ref_logratios).mean().item()
            reward_margin = (policy_logratios - ref_logratios).mean().item()
        else:  # PyTorch tensor
            import torch
            loss = -torch.nn.functional.logsigmoid(logits).mean()
            accuracy = (policy_logratios > ref_logratios).float().mean().item()
            reward_margin = (policy_logratios - ref_logratios).mean().item()
        
        return LossOutput(
            loss=loss,
            metrics={
                "dpo_loss": loss.item() if hasattr(loss, 'item') else float(loss),
                "accuracy": accuracy,
                "reward_margin": reward_margin,
                "policy_chosen_logp": policy_chosen_logps.mean().item() if hasattr(policy_chosen_logps, 'item') else float(policy_chosen_logps.mean()),
                "policy_rejected_logp": policy_rejected_logps.mean().item() if hasattr(policy_rejected_logps, 'item') else float(policy_rejected_logps.mean()),
            }
        )
