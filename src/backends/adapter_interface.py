"""
Adapter Interface - Abstract interface for LoRA adapter management.

Provides unified API for adding, saving, loading, and merging LoRA adapters
that works with both PyTorch (PEFT) and MLX (mlx-lm) implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class AdapterConfig:
    """
    Unified LoRA adapter configuration.

    Compatible with both PEFT (PyTorch) and mlx-lm (MLX) LoRA implementations.
    """

    r: int = 8  # LoRA rank
    alpha: float = 16.0  # LoRA alpha (scaling factor)
    dropout: float = 0.05  # Dropout probability
    target_modules: List[str] = field(default_factory=list)  # Modules to adapt
    bias: str = "none"  # Bias adaptation: "none", "all", "lora_only"
    task_type: str = "CAUSAL_LM"  # Task type

    def __post_init__(self):
        """Set default target modules if not provided."""
        if not self.target_modules:
            # Default to common attention projection layers
            self.target_modules = ["q_proj", "v_proj", "o_proj"]

    def to_peft_config(self) -> Dict:
        """
        Convert to PEFT LoraConfig format.

        Returns:
            Dictionary for PEFT LoraConfig initialization
        """
        return {
            "r": self.r,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
        }

    def to_mlx_config(self) -> Dict:
        """
        Convert to MLX LoRA config format.

        Returns:
            Dictionary for MLX LoRA initialization
        """
        return {
            "rank": self.r,
            "scale": self.alpha / self.r,  # MLX uses scale instead of alpha
            "dropout": self.dropout,
            "target_modules": self.target_modules,
        }


class AdapterManager(ABC):
    """
    Abstract interface for LoRA adapter management.

    Handles adding, saving, loading, and merging LoRA adapters across
    different ML frameworks (PyTorch PEFT, MLX).
    """

    @abstractmethod
    def add_adapter(
        self,
        model: "ModelInterface",
        config: AdapterConfig,
        adapter_name: str = "default",
    ) -> "ModelInterface":
        """
        Add LoRA adapter to model.

        Args:
            model: Model to add adapter to
            config: Adapter configuration
            adapter_name: Name for this adapter

        Returns:
            Model with adapter attached
        """
        ...

    @abstractmethod
    def save_adapter(
        self,
        model: "ModelInterface",
        path: Path,
        adapter_name: str = "default",
    ) -> None:
        """
        Save adapter weights to disk.

        Args:
            model: Model with adapter
            path: Directory to save adapter
            adapter_name: Name of adapter to save
        """
        ...

    @abstractmethod
    def load_adapter(
        self,
        model: "ModelInterface",
        path: Path,
        adapter_name: str = "default",
    ) -> "ModelInterface":
        """
        Load adapter weights from disk.

        Args:
            model: Model to load adapter into
            path: Directory containing adapter weights
            adapter_name: Name for loaded adapter

        Returns:
            Model with adapter loaded
        """
        ...

    @abstractmethod
    def merge_adapter(
        self,
        model: "ModelInterface",
        adapter_name: str = "default",
    ) -> "ModelInterface":
        """
        Merge adapter weights into base model.

        WARNING: On PyTorch MPS backend, this may cause model corruption
        due to non-contiguous tensor bugs. Prefer save/reload strategy.

        Args:
            model: Model with adapter
            adapter_name: Name of adapter to merge

        Returns:
            Model with adapter merged (no longer attached as adapter)
        """
        ...

    @abstractmethod
    def unload_adapter(
        self,
        model: "ModelInterface",
        adapter_name: str = "default",
    ) -> "ModelInterface":
        """
        Remove adapter from model without merging.

        WARNING: On PyTorch MPS backend, this may cause issues. Consider
        using merge or save/reload instead.

        Args:
            model: Model with adapter
            adapter_name: Name of adapter to unload

        Returns:
            Model with adapter removed
        """
        ...

    @abstractmethod
    def has_adapter(
        self,
        model: "ModelInterface",
        adapter_name: Optional[str] = None,
    ) -> bool:
        """
        Check if model has adapter(s).

        Args:
            model: Model to check
            adapter_name: Optional specific adapter name to check

        Returns:
            True if model has adapter(s)
        """
        ...

    @abstractmethod
    def list_adapters(
        self,
        model: "ModelInterface",
    ) -> List[str]:
        """
        List all adapters in model.

        Args:
            model: Model to inspect

        Returns:
            List of adapter names
        """
        ...
