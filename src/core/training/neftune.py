"""
NEFTune: Noisy Embeddings Improve Instruction Finetuning.

Paper: https://arxiv.org/abs/2310.05914

Simple technique that adds uniform noise to embeddings during training.
Improves instruction following by 10-15% on benchmarks.
"""

from dataclasses import dataclass
from typing import Any
import math


@dataclass
class NEFTuneConfig:
    """NEFTune configuration."""
    noise_alpha: float = 5.0  # Noise scaling factor
    enabled: bool = True  # Enable/disable NEFTune


def apply_neftune(
    embeddings: Any,
    seq_length: int,
    config: NEFTuneConfig,
    training: bool = True,
) -> Any:
    """
    Apply NEFTune noise to embeddings.
    
    NEFTune adds uniform noise scaled by sequence length to embeddings
    during training. This simple technique significantly improves
    instruction following performance.
    
    Args:
        embeddings: Input embeddings [batch, seq, hidden]
        seq_length: Sequence length for normalization
        config: NEFTune configuration
        training: Whether in training mode
    
    Returns:
        Noisy embeddings (or original if not training/disabled)
    """
    if not training or not config.enabled:
        return embeddings
    
    # Generate uniform noise
    import numpy as np
    
    if hasattr(embeddings, 'shape'):  # MLX or PyTorch
        if hasattr(embeddings, '__array__'):  # MLX
            import mlx.core as mx
            noise = mx.random.uniform(
                -1.0, 1.0, embeddings.shape
            )
            # Scale by sequence length
            noise = noise * config.noise_alpha / math.sqrt(seq_length)
            return embeddings + noise
        else:  # PyTorch
            import torch
            noise = torch.empty_like(embeddings).uniform_(
                -1.0, 1.0
            )
            noise = noise * config.noise_alpha / math.sqrt(seq_length)
            return embeddings + noise
    else:
        # NumPy fallback
        noise = np.random.uniform(
            -1.0, 1.0, embeddings.shape
        )
        noise = noise * config.noise_alpha / math.sqrt(seq_length)
        return embeddings + noise
