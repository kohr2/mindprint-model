"""
Backend Protocol - Core abstraction for multiple ML frameworks.

Defines the interface that all backends (PyTorch, MLX, etc.) must implement
to support ORPO training with LoRA adapters.
"""

from typing import Protocol, Any, Dict, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BackendConfig:
    """Configuration for backend initialization."""

    backend_type: str  # "pytorch" or "mlx"
    device: str = "auto"  # "mps", "cuda", "cpu", "gpu", "auto"
    dtype: str = "float16"  # "float16", "float32", "bfloat16"
    quantization: Optional[str] = None  # "int4", "int8", None for no quantization
    seed: int = 42
    validate: bool = True  # Set to False to skip validation (for testing)

    def __post_init__(self):
        """Validate configuration."""
        if not self.validate:
            return

        # Import here to avoid circular dependency
        from .factory import BackendRegistry

        # Check if backend is registered
        if not BackendRegistry.is_registered(self.backend_type):
            available = BackendRegistry.list_backends()
            raise ValueError(
                f"Unknown backend_type: {self.backend_type}. "
                f"Registered backends: {available}"
            )

        valid_dtypes = ["float16", "float32", "bfloat16"]
        if self.dtype not in valid_dtypes:
            raise ValueError(
                f"Invalid dtype: {self.dtype}. "
                f"Must be one of {valid_dtypes}"
            )


class BackendProtocol(Protocol):
    """
    Protocol defining the interface all backends must implement.

    This protocol ensures that PyTorch and MLX backends provide
    equivalent functionality for model loading, training, and inference.
    """

    @property
    def name(self) -> str:
        """Backend name (e.g., 'pytorch', 'mlx')."""
        ...

    @property
    def config(self) -> BackendConfig:
        """Backend configuration."""
        ...

    def load_model(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
    ) -> "ModelInterface":
        """
        Load a model from path.

        Args:
            model_path: Path to model weights or HuggingFace model ID
            tokenizer_path: Optional path to tokenizer (defaults to model_path)
            adapter_path: Optional path to LoRA adapter to load

        Returns:
            ModelInterface instance wrapping the loaded model
        """
        ...

    def create_orpo_trainer(
        self,
        model: "ModelInterface",
        config: Dict[str, Any],
    ) -> "TrainerInterface":
        """
        Create an ORPO (Odds Ratio Preference Optimization) trainer.

        Args:
            model: Model to train
            config: ORPO configuration (learning rate, batch size, lambda_orpo, etc.)

        Returns:
            TrainerInterface for ORPO training
        """
        ...

    def get_device_manager(self) -> "DeviceManager":
        """
        Get device management utilities for this backend.

        Returns:
            DeviceManager for handling device operations
        """
        ...


class DeviceManager(Protocol):
    """
    Protocol for device management across backends.

    Handles device detection, memory management, and tensor movement.
    """

    def get_device(self) -> Any:
        """
        Get the current device.

        Returns:
            Device object (torch.device for PyTorch, string for MLX)
        """
        ...

    def empty_cache(self) -> None:
        """Clear device memory cache."""
        ...

    def move_to_device(self, obj: Any) -> Any:
        """
        Move object to device.

        Args:
            obj: Object to move (model, tensor, etc.)

        Returns:
            Object on device
        """
        ...

    def is_available(self) -> bool:
        """
        Check if accelerator is available.

        Returns:
            True if GPU/MPS is available
        """
        ...
