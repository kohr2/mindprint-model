"""
MPS (Metal Performance Shaders) utilities for Mac Studio M2 Ultra.

Provides optimized training utilities for Apple Silicon:
- MPS device detection with CPU fallback
- Memory management (cache clearing)
- Safe tensor/model movement between devices
- Training context manager for clean resource handling
"""

from dataclasses import dataclass
from typing import Union, Optional
import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class MPSConfig:
    """Configuration for MPS training.

    Optimized defaults for Mac Studio M2 Ultra (64GB unified memory).
    """

    device: str = "mps"
    dtype: torch.dtype = torch.float16
    fallback_to_cpu: bool = True
    gradient_checkpointing: bool = True


def get_mps_device() -> torch.device:
    """
    Get the best available device, preferring MPS.

    Returns:
        torch.device: MPS if available, otherwise CPU
    """
    if torch.backends.mps.is_available():
        logger.info("MPS device available, using Apple Silicon GPU")
        return torch.device("mps")
    else:
        logger.warning("MPS not available, falling back to CPU")
        return torch.device("cpu")


def mps_empty_cache() -> None:
    """
    Clear MPS memory cache.

    Safe to call even when MPS is not available.
    """
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            logger.debug("Cleared MPS cache")
        except Exception as e:
            logger.warning(f"Failed to clear MPS cache: {e}")


def move_to_device(
    tensor_or_model: Union[torch.Tensor, nn.Module],
    device: torch.device,
    fallback_to_cpu: bool = True,
) -> Union[torch.Tensor, nn.Module]:
    """
    Safely move a tensor or model to the specified device.

    Args:
        tensor_or_model: Tensor or nn.Module to move
        device: Target device
        fallback_to_cpu: If True, fallback to CPU on failure

    Returns:
        Tensor or model on the target device
    """
    try:
        return tensor_or_model.to(device)
    except Exception as e:
        if fallback_to_cpu and device.type != "cpu":
            logger.warning(f"Failed to move to {device}, falling back to CPU: {e}")
            return tensor_or_model.to(torch.device("cpu"))
        raise


def check_mps_operation_support(operation: str) -> bool:
    """
    Check if a specific operation is supported on MPS.

    Args:
        operation: Name of the operation to check

    Returns:
        True if the operation is supported on MPS
    """
    if not torch.backends.mps.is_available():
        return False

    # Common operations that are well-supported on MPS
    supported_ops = {
        "matmul",
        "add",
        "mul",
        "div",
        "sub",
        "relu",
        "gelu",
        "softmax",
        "layer_norm",
        "linear",
        "conv2d",
        "embedding",
        "attention",
    }

    return operation.lower() in supported_ops


class MPSTrainingContext:
    """
    Context manager for MPS training.

    Handles:
    - Device selection
    - Memory cleanup on exit
    - Exception handling
    """

    def __init__(self, config: Optional[MPSConfig] = None):
        """
        Initialize the training context.

        Args:
            config: MPS configuration (uses defaults if None)
        """
        self.config = config or MPSConfig()
        self._device: Optional[torch.device] = None

    @property
    def device(self) -> torch.device:
        """Get the current device."""
        if self._device is None:
            if self.config.device == "mps":
                self._device = get_mps_device()
            else:
                self._device = torch.device(self.config.device)
        return self._device

    def __enter__(self) -> "MPSTrainingContext":
        """Enter the training context."""
        logger.info(f"Entering MPS training context with device: {self.device}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the training context and clean up."""
        mps_empty_cache()
        logger.info("Exited MPS training context, cleared cache")

        # Don't suppress exceptions
        return None
