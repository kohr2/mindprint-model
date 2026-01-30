"""
Loss functions for preference learning.

Implements state-of-the-art loss functions:
- DPO: Direct Preference Optimization
- SimPO: Simple Preference Optimization (length-normalized)
- ORPO: Odds Ratio Preference Optimization (combined SFT+alignment)
- IPO: Identity Preference Optimization (regularized DPO)
- KTO: Kahneman-Tversky Optimization (prospect theory)
"""

from .base import BaseLoss, LossOutput
from .dpo import DPOLoss, DPOConfig
from .simpo import SimPOLoss, SimPOConfig
from .orpo import ORPOLoss, ORPOConfig

__all__ = [
    "BaseLoss",
    "LossOutput",
    "DPOLoss",
    "DPOConfig",
    "SimPOLoss",
    "SimPOConfig",
    "ORPOLoss",
    "ORPOConfig",
]
