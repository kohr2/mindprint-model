"""
MLX Device Manager - Device management for MLX backend.

MLX uses unified memory on Apple Silicon, so device management is simpler
than PyTorch. All operations happen on GPU with automatic memory management.
"""

from typing import Any
import logging

logger = logging.getLogger(__name__)


class MLXDeviceManager:
    """
    Device manager for MLX backend.

    MLX uses unified memory on Apple Silicon M-series chips, so there's
    no explicit device placement needed. All operations automatically
    use the GPU.
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize device manager.

        Args:
            device: Target device (MLX only supports "auto", "gpu", or "mps")
                   All resolve to GPU on Apple Silicon.
        """
        # MLX doesn't need explicit device selection
        # All operations use unified memory and GPU automatically
        self._device = self._resolve_device(device)
        logger.info(f"MLX device manager initialized (device: {self._device})")

    def _resolve_device(self, device: str) -> str:
        """
        Resolve device string for MLX.

        Args:
            device: Device string (all map to "gpu" for MLX)

        Returns:
            Device string ("gpu")
        """
        # MLX always uses GPU on Apple Silicon
        # Unified memory means no explicit device placement needed
        valid_devices = ["auto", "gpu", "mps"]

        if device not in valid_devices:
            logger.warning(
                f"MLX only supports 'auto', 'gpu', or 'mps'. "
                f"Got '{device}', using 'gpu'"
            )

        return "gpu"

    def get_device(self) -> str:
        """
        Get the current device.

        Returns:
            Device string ("gpu")
        """
        return self._device

    def empty_cache(self) -> None:
        """
        Clear device memory cache.

        For MLX, this is a no-op since MLX uses unified memory
        with automatic memory management.
        """
        # MLX handles memory automatically with unified memory
        # No manual cache clearing needed
        logger.debug("MLX empty_cache called (no-op due to unified memory)")

    def move_to_device(self, obj: Any) -> Any:
        """
        Move object to device.

        For MLX, this is a no-op since all arrays are already in
        unified memory accessible by both CPU and GPU.

        Args:
            obj: Object to move (MLX array, model, etc.)

        Returns:
            Object (unchanged)
        """
        # MLX uses unified memory - no device movement needed
        return obj

    def is_available(self) -> bool:
        """
        Check if GPU is available.

        Returns:
            True (MLX always has GPU access on Apple Silicon)
        """
        # MLX backend only runs on Apple Silicon with GPU
        # If we're here, GPU is available
        return True

    @property
    def device_type(self) -> str:
        """Get device type as string."""
        return self._device

    @property
    def is_gpu(self) -> bool:
        """Check if using GPU (always True for MLX)."""
        return True

    @property
    def supports_mixed_precision(self) -> bool:
        """Check if mixed precision is supported (True for MLX)."""
        return True

    def synchronize(self) -> None:
        """
        Synchronize device operations.

        For MLX, we can use mx.eval() to force evaluation of lazy computations.
        """
        try:
            import mlx.core as mx
            # Force evaluation of any pending lazy computations
            mx.eval([])
            logger.debug("MLX device synchronized")
        except ImportError:
            logger.warning("mlx not installed, cannot synchronize")
