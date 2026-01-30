"""
Reproducibility utilities.

Seed management and determinism for reproducible training.
"""

from dataclasses import dataclass, asdict, is_dataclass
import random
import hashlib
import json
import platform
import sys
from datetime import datetime
from typing import Any, Dict, Union
import numpy as np

@dataclass
class SeedConfig:
    """Seed configuration for reproducibility."""
    seed: int = 42
    set_python: bool = True
    set_numpy: bool = True
    set_torch: bool = True
    set_mlx: bool = True


def set_seed(seed_or_config: Union[int, SeedConfig] = 42) -> None:
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed_or_config: Either an integer seed (default: 42) or a SeedConfig object
    """
    if isinstance(seed_or_config, int):
        config = SeedConfig(seed=seed_or_config)
    else:
        config = seed_or_config
    
    if config.set_python:
        random.seed(config.seed)
    
    if config.set_numpy:
        np.random.seed(config.seed)
    
    if config.set_torch:
        try:
            import torch
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
            # MPS (Apple Silicon)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.manual_seed(config.seed)
        except ImportError:
            pass
    
    if config.set_mlx:
        try:
            import mlx.core as mx
            mx.random.seed(config.seed)
        except ImportError:
            pass


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
    # Try to import torch for device detection
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            torch_version = torch.__version__
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            device_name = "Apple Silicon"
            torch_version = torch.__version__
        else:
            device = "cpu"
            device_name = "CPU"
            torch_version = torch.__version__
    except ImportError:
        device = "unknown"
        device_name = "Unknown"
        torch_version = "not installed"

    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch_version,
        "numpy_version": np.__version__,
        "device": device,
        "device_name": device_name,
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat(),
    }
