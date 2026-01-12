"""
PyTorch Backend Package - PyTorch/transformers implementation of backend protocol.

Provides PyTorch-based implementations of all backend interfaces using
transformers, PEFT, and TRL libraries.
"""

from .pytorch_backend import PyTorchBackend
from .pytorch_model import PyTorchModel
from .pytorch_device_manager import PyTorchDeviceManager
from .pytorch_adapter_manager import PyTorchAdapterManager
from .pytorch_sft_trainer import PyTorchSFTTrainer
from .pytorch_dpo_trainer import PyTorchDPOTrainer

# Register PyTorch backend with the factory
from ..factory import BackendRegistry

BackendRegistry.register("pytorch", PyTorchBackend)

__all__ = [
    "PyTorchBackend",
    "PyTorchModel",
    "PyTorchDeviceManager",
    "PyTorchAdapterManager",
    "PyTorchSFTTrainer",
    "PyTorchDPOTrainer",
]
