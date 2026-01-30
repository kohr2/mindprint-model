"""
LoRA configuration with state-of-the-art defaults.

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method.
"""

from dataclasses import dataclass, field
from typing import List
from enum import Enum


class LoRAVariant(Enum):
    """LoRA variants."""
    STANDARD = "standard"  # Original LoRA
    DORA = "dora"  # Weight-Decomposed LoRA
    QLORA = "qlora"  # Quantized LoRA


@dataclass
class LoRAConfig:
    """
    State-of-the-art LoRA configuration.
    
    Defaults optimized for 7B instruction-tuned models.
    """
    rank: int = 32  # Increased from 8 for better quality
    alpha: int = 64  # Typically 2x rank
    dropout: float = 0.05
    variant: LoRAVariant = LoRAVariant.STANDARD
    
    # Target ALL linear layers for best quality
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",  # MLP
    ])
    
    # Modules to never apply LoRA to
    modules_to_save: List[str] = field(default_factory=lambda: [
        "embed_tokens", "lm_head"
    ])
    
    def validate(self) -> None:
        """Validate configuration."""
        assert self.rank > 0, "rank must be positive"
        assert self.alpha > 0, "alpha must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert len(self.target_modules) > 0, "must target at least one module"
