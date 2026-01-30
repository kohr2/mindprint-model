"""
Base loss function interface.

All preference learning loss functions inherit from BaseLoss,
providing a consistent API and enabling easy swapping between methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class LossOutput:
    """
    Standardized loss output.
    
    Attributes:
        loss: The computed loss value (tensor/array)
        metrics: Dictionary of metrics to log (e.g., accuracy, reward margin)
    """
    loss: Any  # Tensor/array from ML framework
    metrics: Dict[str, float]  # Logged metrics


class BaseLoss(ABC):
    """
    Abstract base class for preference learning losses.
    
    All loss functions must implement:
    - compute(): Compute loss and metrics
    - requires_reference_model: Whether a reference model is needed
    
    This abstraction allows swapping between DPO, SimPO, ORPO, etc.
    without changing training code.
    """
    
    @abstractmethod
    def compute(
        self,
        policy_chosen_logps: Any,
        policy_rejected_logps: Any,
        **kwargs
    ) -> LossOutput:
        """
        Compute loss and metrics.
        
        Args:
            policy_chosen_logps: Log probabilities of chosen responses from policy model
            policy_rejected_logps: Log probabilities of rejected responses from policy model
            **kwargs: Additional arguments (e.g., ref_logps, lengths, etc.)
        
        Returns:
            LossOutput with loss value and metrics dictionary
        """
        pass
    
    @property
    @abstractmethod
    def requires_reference_model(self) -> bool:
        """
        Whether this loss needs a reference model.
        
        Returns:
            True if reference model log probabilities are required
        """
        pass
