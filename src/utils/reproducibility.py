"""
Reproducibility utilities for deterministic training and experiment tracking.

Provides:
- set_seed(): Set random seeds for reproducibility
- hash_config(): Generate hash of config for experiment tracking
- get_reproducibility_info(): Collect environment info for logging
"""

import random
import hashlib
import json
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU, CUDA, MPS)

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def hash_config(config: Any) -> str:
    """
    Generate a SHA256 hash of a configuration for experiment tracking.

    Supports:
    - Dictionaries
    - Dataclasses
    - Nested structures

    Args:
        config: Configuration object (dict or dataclass)

    Returns:
        64-character hex string (SHA256 hash)
    """
    # Convert dataclass to dict if needed
    if is_dataclass(config) and not isinstance(config, type):
        config_dict = asdict(config)
    elif isinstance(config, dict):
        config_dict = config
    else:
        # Try to convert to dict
        config_dict = vars(config) if hasattr(config, "__dict__") else {"value": str(config)}

    # Sort keys for consistent ordering
    json_str = json.dumps(config_dict, sort_keys=True, default=str)

    # Generate SHA256 hash
    return hashlib.sha256(json_str.encode()).hexdigest()


def get_reproducibility_info() -> Dict[str, Any]:
    """
    Collect environment information for reproducibility logging.

    Returns:
        Dict with Python version, PyTorch version, device, platform, timestamp
    """
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        device_name = "Apple Silicon"
    else:
        device = "cpu"
        device_name = "CPU"

    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": device,
        "device_name": device_name,
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat(),
    }
