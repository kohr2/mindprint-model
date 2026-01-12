"""
PyTorch Device Manager - Device management for PyTorch backend.

Wraps MPS utilities and provides unified device management across
CUDA, MPS, and CPU devices.
"""

from typing import Any, Optional
import torch
import logging

from ...training.mps_utils import (
    get_mps_device,
    mps_empty_cache,
    move_to_device as mps_move_to_device,
)

logger = logging.getLogger(__name__)


class PyTorchDeviceManager:
    """
    Device manager for PyTorch backend.

    Handles device detection, memory management, and tensor movement
    across CUDA, MPS, and CPU devices.
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize device manager.

        Args:
            device: Target device ("auto", "cuda", "mps", "cpu")
        """
        self._device = self._resolve_device(device)
        logger.info(f"PyTorch device manager initialized with device: {self._device}")

    def _resolve_device(self, device: str) -> torch.device:
        """
        Resolve device string to torch.device.

        Args:
            device: Device string ("auto", "cuda", "mps", "cpu")

        Returns:
            torch.device instance
        """
        if device == "auto":
            # Auto-select best available device
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return get_mps_device()
            else:
                return torch.device("cpu")

        elif device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                return torch.device("cpu")
            return torch.device("cuda")

        elif device == "mps":
            if not torch.backends.mps.is_available():
                logger.warning("MPS requested but not available, falling back to CPU")
                return torch.device("cpu")
            return get_mps_device()

        elif device == "cpu":
            return torch.device("cpu")

        else:
            logger.warning(f"Unknown device '{device}', falling back to auto")
            return self._resolve_device("auto")

    def get_device(self) -> torch.device:
        """
        Get the current device.

        Returns:
            torch.device instance
        """
        return self._device

    def empty_cache(self) -> None:
        """
        Clear device memory cache.

        Calls appropriate cache clearing function based on device type.
        """
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")
        elif self._device.type == "mps":
            mps_empty_cache()
            logger.debug("Cleared MPS cache")
        # CPU doesn't have cache to clear

    def move_to_device(self, obj: Any) -> Any:
        """
        Move object to device.

        Args:
            obj: Object to move (model, tensor, etc.)

        Returns:
            Object on device
        """
        if self._device.type == "mps":
            # Use MPS-specific move function for safety
            return mps_move_to_device(obj, self._device)
        else:
            # Standard PyTorch .to() for CUDA/CPU
            if hasattr(obj, 'to'):
                return obj.to(self._device)
            return obj

    def is_available(self) -> bool:
        """
        Check if accelerator is available.

        Returns:
            True if GPU/MPS is available (not CPU)
        """
        if self._device.type == "cuda":
            return torch.cuda.is_available()
        elif self._device.type == "mps":
            return torch.backends.mps.is_available()
        return False  # CPU always available but not an "accelerator"

    @property
    def device_type(self) -> str:
        """Get device type as string (cuda, mps, cpu)."""
        return self._device.type

    @property
    def is_mps(self) -> bool:
        """Check if using MPS device."""
        return self._device.type == "mps"

    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA device."""
        return self._device.type == "cuda"

    @property
    def is_cpu(self) -> bool:
        """Check if using CPU device."""
        return self._device.type == "cpu"
