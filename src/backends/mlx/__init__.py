"""
MLX Backend Package - MLX implementation of backend protocol.

Provides MLX-based implementations of all backend interfaces optimized
for Apple Silicon M-series chips.
"""

from .mlx_backend import MLXBackend
from .mlx_model import MLXModel
from .mlx_device_manager import MLXDeviceManager
from .mlx_adapter_manager import MLXAdapterManager
from .mlx_sft_trainer import MLXSFTTrainer
from .mlx_dpo_trainer import MLXDPOTrainer

# Register MLX backend with the factory
from ..factory import BackendRegistry

BackendRegistry.register("mlx", MLXBackend)

__all__ = [
    "MLXBackend",
    "MLXModel",
    "MLXDeviceManager",
    "MLXAdapterManager",
    "MLXSFTTrainer",
    "MLXDPOTrainer",
]
