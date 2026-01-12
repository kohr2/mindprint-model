"""
Model Interface - Abstract interface for models across backends.

Provides a unified API for model operations (forward pass, generation,
adapter management) that works with both PyTorch and MLX.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pathlib import Path


class ModelInterface(ABC):
    """
    Abstract interface for models across backends.

    This interface abstracts away framework-specific model implementations,
    allowing the training pipeline to work with PyTorch (transformers/PEFT)
    or MLX (mlx-lm) models transparently.
    """

    @abstractmethod
    def generate(
        self,
        input_ids: Any,
        attention_mask: Optional[Any] = None,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        **kwargs,
    ) -> Any:
        """
        Generate text from inputs.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            Generated token IDs
        """
        ...

    @abstractmethod
    def forward(
        self,
        input_ids: Any,
        attention_mask: Optional[Any] = None,
        labels: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation
            **kwargs: Additional forward pass parameters

        Returns:
            Dictionary containing:
                - loss: Loss value (if labels provided)
                - logits: Model output logits
                - Any other model outputs
        """
        ...

    @abstractmethod
    def save_pretrained(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Directory to save model
        """
        ...

    @abstractmethod
    def load_adapter(self, adapter_path: Path) -> None:
        """
        Load a LoRA adapter into the model.

        Args:
            adapter_path: Path to adapter weights
        """
        ...

    @abstractmethod
    def save_adapter(self, adapter_path: Path) -> None:
        """
        Save current LoRA adapter.

        Args:
            adapter_path: Path to save adapter
        """
        ...

    @abstractmethod
    def has_adapter(self) -> bool:
        """
        Check if model has an active adapter.

        Returns:
            True if adapter is attached
        """
        ...

    @abstractmethod
    def unload_adapter(self) -> None:
        """
        Unload current adapter from model.

        WARNING: This may cause corruption on PyTorch MPS backend.
        Use with caution or prefer save/reload strategy.
        """
        ...

    @property
    @abstractmethod
    def device(self) -> Any:
        """
        Get model device.

        Returns:
            Device object (backend-specific)
        """
        ...

    @property
    @abstractmethod
    def dtype(self) -> Any:
        """
        Get model dtype.

        Returns:
            Dtype object (backend-specific)
        """
        ...

    @property
    @abstractmethod
    def config(self) -> Any:
        """
        Get model configuration.

        Returns:
            Model config object (backend-specific)
        """
        ...

    @abstractmethod
    def get_tokenizer(self) -> Any:
        """
        Get the tokenizer associated with this model.

        Returns:
            Tokenizer object
        """
        ...

    @abstractmethod
    def parameters(self) -> Any:
        """
        Get model parameters.

        Returns:
            Iterator over parameters (backend-specific)
        """
        ...

    @abstractmethod
    def named_parameters(self) -> Any:
        """
        Get named model parameters.

        Returns:
            Iterator over (name, parameter) tuples
        """
        ...

    @abstractmethod
    def eval(self) -> "ModelInterface":
        """
        Set model to evaluation mode.

        Returns:
            Self for chaining
        """
        ...

    @abstractmethod
    def train(self) -> "ModelInterface":
        """
        Set model to training mode.

        Returns:
            Self for chaining
        """
        ...
