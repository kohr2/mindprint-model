"""
MPS utilities for Mac Studio M2 Ultra training.

Handles MPS-specific operations, memory management, and CPU fallbacks
for operations not supported on MPS.

Optimized for:
- Mac Studio M2 Ultra (64GB unified memory)
- PyTorch MPS backend
- float16 precision (no quantization needed)
"""

from dataclasses import dataclass
from typing import Union, Set
import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Operations known to be supported on MPS
SUPPORTED_OPS: Set[str] = {
    "matmul",
    "softmax",
    "attention",
    "linear",
    "layernorm",
    "embedding",
    "gelu",
    "relu",
    "dropout",
    "cross_entropy",
    "mse",
}

# Operations known to have issues on MPS
UNSUPPORTED_OPS: Set[str] = {
    "scatter_reduce",  # Some indexing operations
    "complex_gather",  # Complex tensor indexing
    "sparse_matmul",  # Sparse operations
}


@dataclass
class MPSConfig:
    """Configuration for MPS backend.

    Attributes:
        device: Target device ("mps" or "cpu")
        dtype: Tensor dtype (float16 recommended for MPS)
        fallback_to_cpu: Whether to fall back to CPU on failures
        gradient_checkpointing: Enable gradient checkpointing for memory
    """

    device: str = "mps"
    dtype: torch.dtype = torch.float16
    fallback_to_cpu: bool = True
    gradient_checkpointing: bool = True


def get_mps_device() -> torch.device:
    """
    Get MPS device with validation.

    Returns MPS device if available, CPU otherwise.

    Returns:
        torch.device: MPS device if available, CPU otherwise
    """
    if torch.backends.mps.is_available():
        logger.debug("MPS device available, using MPS")
        return torch.device("mps")
    else:
        logger.warning("MPS not available, falling back to CPU")
        return torch.device("cpu")


def mps_empty_cache() -> None:
    """
    Clear MPS memory cache.

    Equivalent to torch.cuda.empty_cache() for MPS.
    Should be called between training phases to prevent memory buildup.
    """
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            logger.debug("MPS cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear MPS cache: {e}")


def move_to_device(
    tensor_or_model: Union[Tensor, torch.nn.Module],
    device: torch.device,
    fallback_to_cpu: bool = True,
) -> Union[Tensor, torch.nn.Module]:
    """
    Move tensor or model to device with fallback.

    Some operations fail on MPS and need CPU fallback.

    Args:
        tensor_or_model: Tensor or model to move
        device: Target device
        fallback_to_cpu: If True, fall back to CPU on failure

    Returns:
        Tensor or model on target device
    """
    try:
        return tensor_or_model.to(device)
    except Exception as e:
        if fallback_to_cpu and device.type != "cpu":
            logger.warning(f"Failed to move to {device}, falling back to CPU: {e}")
            return tensor_or_model.to("cpu")
        raise


def check_mps_operation_support(operation: str) -> bool:
    """
    Check if an operation is supported on MPS.

    Args:
        operation: Name of operation to check

    Returns:
        True if supported, False otherwise
    """
    operation_lower = operation.lower()

    # Check against known supported operations
    if operation_lower in SUPPORTED_OPS:
        return True

    # Check against known unsupported operations
    if operation_lower in UNSUPPORTED_OPS:
        return False

    # Unknown operation - assume unsupported for safety
    return False


class MPSTrainingContext:
    """
    Context manager for MPS training.

    Handles:
    - Device placement
    - Memory cleanup between batches
    - Automatic CPU fallback for unsupported ops

    Example:
        with MPSTrainingContext() as ctx:
            model = model.to(ctx.device)
            output = model(input)
    """

    def __init__(self, config: MPSConfig = None):
        """
        Initialize training context.

        Args:
            config: Optional MPSConfig. Uses defaults if not provided.
        """
        self.config = config or MPSConfig()
        self.device = get_mps_device() if self.config.device == "mps" else torch.device("cpu")

    def __enter__(self) -> "MPSTrainingContext":
        """Enter context and set up device."""
        logger.debug(f"Entering MPS training context with device: {self.device}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and clean up."""
        # Clear cache regardless of exception
        try:
            mps_empty_cache()
        except Exception as e:
            logger.warning(f"Error during cache cleanup: {e}")

        # Don't suppress exceptions
        logger.debug("Exiting MPS training context")
