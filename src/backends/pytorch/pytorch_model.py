"""
PyTorch Model Wrapper - ModelInterface implementation for PyTorch.

Wraps transformers.PreTrainedModel and peft.PeftModel to provide
a unified interface for model operations.
"""

from typing import Any, Dict, Optional, List
from pathlib import Path
import logging
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel

from ..model_interface import ModelInterface

logger = logging.getLogger(__name__)


class PyTorchModel(ModelInterface):
    """
    PyTorch implementation of ModelInterface.

    Wraps both base transformers models and PEFT-adapted models,
    providing a unified API for training and inference.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        """
        Initialize PyTorch model wrapper.

        Args:
            model: Transformers model (base or PEFT-wrapped)
            tokenizer: Tokenizer for this model
        """
        self._model = model
        self._tokenizer = tokenizer
        logger.info(
            f"Initialized PyTorchModel wrapper for {model.config.model_type} "
            f"(has_adapter={self.has_adapter()})"
        )

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
            input_ids: Input token IDs (torch.Tensor)
            attention_mask: Optional attention mask
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            Generated token IDs (torch.Tensor)
        """
        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": temperature > 0.0,
        }

        if attention_mask is not None:
            generation_kwargs["attention_mask"] = attention_mask

        generation_kwargs.update(kwargs)

        with torch.no_grad():
            outputs = self._model.generate(**generation_kwargs)

        return outputs

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
            input_ids: Input token IDs (torch.Tensor)
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation
            **kwargs: Additional forward pass parameters

        Returns:
            Dictionary with keys: loss, logits, and other outputs
        """
        forward_kwargs = {
            "input_ids": input_ids,
        }

        if attention_mask is not None:
            forward_kwargs["attention_mask"] = attention_mask

        if labels is not None:
            forward_kwargs["labels"] = labels

        forward_kwargs.update(kwargs)

        outputs = self._model(**forward_kwargs)

        # Convert transformers outputs to dict
        result = {
            "logits": outputs.logits,
        }

        if hasattr(outputs, "loss") and outputs.loss is not None:
            result["loss"] = outputs.loss

        # Include any additional outputs
        for key in dir(outputs):
            if not key.startswith("_") and key not in ["logits", "loss"]:
                attr = getattr(outputs, key)
                if not callable(attr):
                    result[key] = attr

        return result

    def save_pretrained(self, path: Path) -> None:
        """
        Save model to disk.

        For PEFT models, saves only the adapter weights.
        For base models, saves full model.

        Args:
            path: Directory to save model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(str(path))
        self._tokenizer.save_pretrained(str(path))

        logger.info(f"Saved model to {path}")

    def load_adapter(self, adapter_path: Path) -> None:
        """
        Load a LoRA adapter into the model.

        Args:
            adapter_path: Path to adapter weights

        Raises:
            ValueError: If model already has an adapter
        """
        if self.has_adapter():
            raise ValueError(
                "Model already has an adapter. Use AdapterManager to manage multiple adapters."
            )

        # Load adapter using PEFT
        self._model = PeftModel.from_pretrained(
            self._model,
            str(adapter_path),
        )

        logger.info(f"Loaded adapter from {adapter_path}")

    def save_adapter(self, adapter_path: Path) -> None:
        """
        Save current LoRA adapter.

        Args:
            adapter_path: Path to save adapter

        Raises:
            ValueError: If model doesn't have an adapter
        """
        if not self.has_adapter():
            raise ValueError("Model doesn't have an adapter to save")

        adapter_path = Path(adapter_path)
        adapter_path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(str(adapter_path))

        logger.info(f"Saved adapter to {adapter_path}")

    def has_adapter(self) -> bool:
        """
        Check if model has a LoRA adapter attached.

        Returns:
            True if model is a PeftModel
        """
        return isinstance(self._model, PeftModel)

    def unload_adapter(self) -> None:
        """
        Remove adapter from model without merging.

        WARNING: On PyTorch MPS backend, this may cause issues.
        Consider using save/reload instead.

        Raises:
            ValueError: If model doesn't have an adapter
        """
        if not self.has_adapter():
            raise ValueError("Model doesn't have an adapter to unload")

        # Get base model from PEFT wrapper
        self._model = self._model.unload()

        logger.warning(
            "Unloaded adapter. Note: On MPS backend, this may cause corruption. "
            "Consider save/reload strategy instead."
        )

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self._model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get model dtype."""
        return next(self._model.parameters()).dtype

    @property
    def config(self) -> Any:
        """Get model config."""
        return self._model.config

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get tokenizer."""
        return self._tokenizer

    @property
    def parameters(self) -> List[torch.nn.Parameter]:
        """Get model parameters."""
        return list(self._model.parameters())

    @property
    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get trainable parameters only."""
        return [p for p in self._model.parameters() if p.requires_grad]

    @property
    def num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters)

    @property
    def num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.trainable_parameters)

    def get_underlying_model(self) -> PreTrainedModel:
        """
        Get the underlying transformers model.

        For advanced use cases that need direct access to the PyTorch model.

        Returns:
            PreTrainedModel (base or PEFT-wrapped)
        """
        return self._model

    def eval(self) -> None:
        """Set model to evaluation mode."""
        self._model.eval()

    def train(self, mode: bool = True) -> None:
        """
        Set model to training mode.

        Args:
            mode: True for training mode, False for eval mode
        """
        self._model.train(mode)

    def to(self, device: torch.device) -> "PyTorchModel":
        """
        Move model to device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        self._model = self._model.to(device)
        return self

    def get_tokenizer(self) -> Any:
        """
        Get the tokenizer for this model.

        Returns:
            The tokenizer instance
        """
        return self._tokenizer

    def named_parameters(self) -> Any:
        """
        Get named parameters iterator.

        Returns:
            Iterator over (name, parameter) pairs
        """
        return self._model.named_parameters()
