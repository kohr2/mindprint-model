"""
ORPO: Odds Ratio Preference Optimization.

Paper: https://arxiv.org/abs/2403.07691

Combines SFT and preference alignment in a single training stage.
More efficient than two-stage SFT+DPO pipelines.
"""

from dataclasses import dataclass
from typing import Any

from .base import BaseLoss, LossOutput


@dataclass
class ORPOConfig:
    """ORPO configuration."""
    lambda_orpo: float = 0.1  # Weight for preference term


class ORPOLoss(BaseLoss):
    """
    Odds Ratio Preference Optimization loss.
    
    ORPO combines supervised fine-tuning and preference alignment
    in a single loss function, eliminating the need for separate
    SFT and DPO stages.
    
    Key innovations:
    - Combined NLL loss (SFT) + odds ratio loss (preference)
    - Single-stage training (2x faster than SFT+DPO)
    - No reference model needed
    - Often better instruction following than DPO
    """
    
    def __init__(self, config: ORPOConfig):
        """
        Initialize ORPO loss.
        
        Args:
            config: ORPO configuration
        """
        self.config = config
    
    @property
    def requires_reference_model(self) -> bool:
        """ORPO does not require reference model."""
        return False
    
    def compute(
        self,
        logits: Any,
        chosen_ids: Any,
        rejected_ids: Any,
        rejected_logits: Any = None,
        **kwargs
    ) -> LossOutput:
        """
        Compute ORPO loss.
        
        Args:
            logits: Model logits for chosen responses [batch, seq_len, vocab_size]
            chosen_ids: Token IDs for chosen responses
            rejected_ids: Token IDs for rejected responses
            rejected_logits: Model logits for rejected responses [batch, seq_len, vocab_size]
                If None, uses logits (for backward compatibility, but not recommended)
        
        Returns:
            LossOutput with ORPO loss and metrics
        """
        # Compute log probabilities for chosen and rejected
        chosen_logps = self._compute_log_probs(logits, chosen_ids)
        # Use rejected_logits if provided, otherwise use chosen logits (not ideal but backward compatible)
        rejected_logits_to_use = rejected_logits if rejected_logits is not None else logits
        rejected_logps = self._compute_log_probs(rejected_logits_to_use, rejected_ids)
        
        # NLL loss (SFT component) - only on chosen
        nll_loss = -chosen_logps.mean()
        
        # Odds ratio loss (preference component)
        log_odds = chosen_logps - rejected_logps
        or_loss = -self._log_sigmoid(log_odds).mean()
        
        # Combined loss
        total_loss = nll_loss + self.config.lambda_orpo * or_loss
        
        # Compute metrics
        import numpy as np
        if hasattr(log_odds, 'item'):
            accuracy = (log_odds > 0).mean().item()
            odds_margin = log_odds.mean().item()
        else:
            accuracy = (log_odds > 0).float().mean().item()
            odds_margin = log_odds.mean().item()
        
        return LossOutput(
            loss=total_loss,
            metrics={
                "orpo_loss": total_loss.item() if hasattr(total_loss, 'item') else float(total_loss),
                "nll_loss": nll_loss.item() if hasattr(nll_loss, 'item') else float(nll_loss),
                "or_loss": or_loss.item() if hasattr(or_loss, 'item') else float(or_loss),
                "accuracy": accuracy,
                "odds_margin": odds_margin,
            }
        )
    
    def _compute_log_probs(self, logits: Any, labels: Any) -> Any:
        """Compute log probabilities for labels given logits."""
        if hasattr(logits, 'item'):  # MLX
            import mlx.nn as nn
            import mlx.core as mx

            log_probs = nn.log_softmax(logits, axis=-1)
            batch_size, seq_len, vocab_size = log_probs.shape
            flat_log_probs = log_probs.reshape(-1, vocab_size)
            flat_labels = labels.reshape(-1)

            # Use one-hot encoding to gather values (MLX-compatible)
            one_hot = mx.eye(vocab_size)[flat_labels.astype(mx.int32)]
            selected = (flat_log_probs * one_hot).sum(axis=1)

            return selected.reshape(batch_size, seq_len).sum(axis=-1)
        else:  # PyTorch
            import torch
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
    
    def _log_sigmoid(self, x: Any) -> Any:
        """Compute log(sigmoid(x)) numerically stable."""
        import numpy as np
        if hasattr(x, 'item'):  # MLX
            return np.logaddexp(0, x) - x  # log(sigmoid(x))
        else:  # PyTorch
            import torch
            return torch.nn.functional.logsigmoid(x)
