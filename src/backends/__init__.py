"""
Backends Package - Multi-framework ML backend abstraction.

Provides a unified interface for training LLMs with different ML frameworks:
- PyTorch (with transformers, PEFT, TRL)
- MLX (Apple Silicon optimized)

Usage:
    from src.backends import create_backend

    # Create MLX backend for Mac Studio
    backend = create_backend("mlx", device="mps")
    model = backend.load_model("Qwen/Qwen2.5-7B-Instruct")
    trainer = backend.create_sft_trainer(model, config)

    # Or use PyTorch for cloud GPU
    backend = create_backend("pytorch", device="cuda")
"""

from .protocol import BackendProtocol, BackendConfig, DeviceManager
from .model_interface import ModelInterface
from .trainer_interface import (
    TrainerInterface,
    SFTTrainerInterface,
    DPOTrainerInterface,
    TrainingResult,
)
from .adapter_interface import AdapterManager, AdapterConfig
from .factory import (
    BackendRegistry,
    create_backend,
    register_backend,
)

__all__ = [
    # Core protocols
    "BackendProtocol",
    "BackendConfig",
    "DeviceManager",
    # Interfaces
    "ModelInterface",
    "TrainerInterface",
    "SFTTrainerInterface",
    "DPOTrainerInterface",
    "TrainingResult",
    "AdapterManager",
    "AdapterConfig",
    # Factory
    "BackendRegistry",
    "create_backend",
    "register_backend",
]

__version__ = "0.1.0"
