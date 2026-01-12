"""
MLX Model Wrapper - ModelInterface implementation for MLX.

Wraps mlx-lm models to provide a unified interface for model operations.
"""

from typing import Any, Dict, Optional, List
from pathlib import Path
import logging

from ..model_interface import ModelInterface

logger = logging.getLogger(__name__)


class MLXModel(ModelInterface):
    """
    MLX implementation of ModelInterface.

    Wraps mlx-lm models, providing a unified API for training and inference.
    """

    def __init__(
        self,
        model: Any,  # mlx-lm model
        tokenizer: Any,  # HuggingFace tokenizer
    ):
        """
        Initialize MLX model wrapper.

        Args:
            model: MLX model (from mlx-lm)
            tokenizer: Tokenizer for this model (HuggingFace compatible)
        """
        self._model = model
        self._tokenizer = tokenizer
        self._has_adapter = False  # Track if LoRA adapter is attached

        logger.info(f"Initialized MLXModel wrapper")

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
            input_ids: Input token IDs (MLX array)
            attention_mask: Optional attention mask (ignored in basic MLX)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            Generated token IDs (MLX array)
        """
        try:
            import mlx.core as mx
            from mlx_lm import generate as mlx_generate

            # MLX generate expects prompt as string or tokens
            # Convert input_ids to list if needed
            if hasattr(input_ids, 'tolist'):
                prompt_tokens = input_ids.tolist()
            else:
                prompt_tokens = input_ids

            # MLX generation
            output = mlx_generate(
                self._model,
                self._tokenizer,
                prompt=prompt_tokens,
                max_tokens=max_new_tokens,
                temp=temperature,
                top_p=top_p,
                **kwargs
            )

            return output

        except ImportError:
            raise RuntimeError(
                "mlx-lm not installed. Install with: pip install mlx-lm"
            )

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
            input_ids: Input token IDs (MLX array)
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation
            **kwargs: Additional forward pass parameters

        Returns:
            Dictionary with keys: loss, logits
        """
        try:
            import mlx.core as mx

            # Forward pass
            logits = self._model(input_ids)

            result = {"logits": logits}

            # Compute loss if labels provided
            if labels is not None:
                # Cross-entropy loss
                # Shift logits and labels for causal LM
                shift_logits = logits[..., :-1, :]
                shift_labels = labels[..., 1:]

                # Flatten for loss computation
                vocab_size = shift_logits.shape[-1]
                shift_logits_flat = shift_logits.reshape(-1, vocab_size)
                shift_labels_flat = shift_labels.reshape(-1)

                # Cross-entropy loss
                loss = mx.mean(
                    mx.losses.cross_entropy(
                        shift_logits_flat,
                        shift_labels_flat,
                        reduction='none'
                    )
                )

                result["loss"] = loss

            return result

        except ImportError:
            raise RuntimeError("mlx not installed. Install with: pip install mlx")

    def save_pretrained(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Directory to save model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        try:
            import mlx.core as mx
            from mlx_lm.utils import save_model

            # Save MLX model
            save_model(str(path), self._model)

            # Save tokenizer
            self._tokenizer.save_pretrained(str(path))

            logger.info(f"Saved MLX model to {path}")

        except ImportError:
            raise RuntimeError("mlx-lm not installed")

    def load_adapter(self, adapter_path: Path) -> None:
        """
        Load a LoRA adapter into the model.

        Args:
            adapter_path: Path to adapter weights
        """
        try:
            from mlx_lm.utils import load_adapter as mlx_load_adapter

            # Load adapter using mlx-lm
            mlx_load_adapter(self._model, str(adapter_path))
            self._has_adapter = True

            logger.info(f"Loaded adapter from {adapter_path}")

        except ImportError:
            raise RuntimeError("mlx-lm not installed")

    def save_adapter(self, adapter_path: Path) -> None:
        """
        Save current LoRA adapter.

        Args:
            adapter_path: Path to save adapter
        """
        if not self._has_adapter:
            raise ValueError("Model doesn't have an adapter to save")

        adapter_path = Path(adapter_path)
        adapter_path.mkdir(parents=True, exist_ok=True)

        try:
            import mlx.core as mx

            # Save adapter weights
            # MLX adapters are saved as part of the model state
            # Extract and save only adapter parameters
            adapter_weights = {}
            for name, param in self._model.named_parameters():
                if 'lora' in name.lower():
                    adapter_weights[name] = param

            # Save to file
            mx.save_safetensors(
                str(adapter_path / "adapter_model.safetensors"),
                adapter_weights
            )

            logger.info(f"Saved adapter to {adapter_path}")

        except ImportError:
            raise RuntimeError("mlx not installed")

    def has_adapter(self) -> bool:
        """
        Check if model has a LoRA adapter attached.

        Returns:
            True if model has an adapter
        """
        return self._has_adapter

    def unload_adapter(self) -> None:
        """
        Remove adapter from model without merging.

        For MLX, this removes LoRA layers from the model.
        """
        if not self._has_adapter:
            raise ValueError("Model doesn't have an adapter to unload")

        # MLX adapter removal would require rebuilding model without LoRA
        # For now, log warning that this is not fully implemented
        logger.warning(
            "MLX adapter unloading not fully implemented. "
            "Consider reloading base model instead."
        )
        self._has_adapter = False

    @property
    def device(self) -> str:
        """Get model device (always 'gpu' for MLX)."""
        return "gpu"

    @property
    def dtype(self) -> str:
        """Get model dtype."""
        # MLX typically uses float16 or bfloat16
        return "float16"

    @property
    def config(self) -> Any:
        """Get model config."""
        # Return tokenizer config as proxy
        return self._tokenizer

    @property
    def tokenizer(self) -> Any:
        """Get tokenizer."""
        return self._tokenizer

    @property
    def parameters(self) -> List[Any]:
        """Get model parameters."""
        try:
            return list(self._model.parameters().values())
        except Exception:
            return []

    @property
    def trainable_parameters(self) -> List[Any]:
        """Get trainable parameters only."""
        # In MLX, parameters are trainable by default
        # LoRA parameters would be the trainable ones when adapter is added
        return self.parameters

    @property
    def num_parameters(self) -> int:
        """Get total number of parameters."""
        try:
            import mlx.core as mx
            total = 0
            for param in self.parameters:
                total += param.size
            return total
        except Exception:
            return 0

    @property
    def num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        # For LoRA, this would be much smaller than total parameters
        return self.num_parameters

    def get_underlying_model(self) -> Any:
        """
        Get the underlying MLX model.

        Returns:
            MLX model
        """
        return self._model

    def eval(self) -> None:
        """Set model to evaluation mode."""
        # MLX models don't have train/eval modes like PyTorch
        # This is a no-op for compatibility
        pass

    def train(self, mode: bool = True) -> None:
        """
        Set model to training mode.

        Args:
            mode: True for training mode, False for eval mode
        """
        # MLX models don't have train/eval modes like PyTorch
        # This is a no-op for compatibility
        pass
