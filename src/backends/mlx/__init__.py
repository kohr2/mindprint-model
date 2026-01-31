"""
MLX Backend Package - MLX implementation of backend protocol.

Provides MLX-based implementations of all backend interfaces optimized
for Apple Silicon M-series chips using ORPO training.
"""

from .mlx_backend import MLXBackend
from .mlx_model import MLXModel
from .mlx_device_manager import MLXDeviceManager
from .mlx_adapter_manager import MLXAdapterManager
from .mlx_orpo_trainer import MLXORPOTrainer

# Register MLX backend with the factory
from ..factory import BackendRegistry

BackendRegistry.register("mlx", MLXBackend)

__all__ = [
    "MLXBackend",
    "MLXModel",
    "MLXDeviceManager",
    "MLXAdapterManager",
    "MLXORPOTrainer",
]
