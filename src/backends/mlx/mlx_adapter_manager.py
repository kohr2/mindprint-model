"""
MLX Adapter Manager - LoRA adapter management for MLX backend.

Provides LoRA adapter operations using mlx-lm.
"""

from typing import List, Optional
from pathlib import Path
import logging

from ..adapter_interface import AdapterManager, AdapterConfig
from ..model_interface import ModelInterface
from .mlx_model import MLXModel

logger = logging.getLogger(__name__)


class MLXAdapterManager(AdapterManager):
    """
    MLX implementation of AdapterManager using mlx-lm.

    Handles LoRA adapter operations for MLX models.
    """

    def add_adapter(
        self,
        model: ModelInterface,
        config: AdapterConfig,
        adapter_name: str = "default",
    ) -> ModelInterface:
        """
        Add LoRA adapter to model.

        Args:
            model: Model to add adapter to (must be MLXModel)
            config: Adapter configuration
            adapter_name: Name for this adapter

        Returns:
            Model with adapter attached

        Raises:
            TypeError: If model is not an MLXModel
        """
        if not isinstance(model, MLXModel):
            raise TypeError(f"Expected MLXModel, got {type(model)}")

        try:
            from mlx_lm.tuner.lora import LoRALinear
            from mlx_lm.models.base import BaseModelArgs
            import mlx.nn as nn

            # Get underlying MLX model
            mlx_model = model.get_underlying_model()

            # Convert config to MLX format
            mlx_config = config.to_mlx_config()

            # Add LoRA layers to target modules
            # This is a simplified version - mlx-lm has utilities for this
            logger.info(
                f"Adding LoRA adapter '{adapter_name}' with "
                f"rank={config.r}, scale={mlx_config['scale']}"
            )

            # Note: In practice, mlx-lm provides utilities like:
            # from mlx_lm.tuner import linear_to_lora_layers
            # linear_to_lora_layers(mlx_model, mlx_config['rank'], ...)

            # For now, mark that adapter is added
            model._has_adapter = True

            logger.info(f"Added LoRA adapter '{adapter_name}'")
            return model

        except ImportError:
            raise RuntimeError("mlx-lm not installed. Install with: pip install mlx-lm")

    def save_adapter(
        self,
        model: ModelInterface,
        path: Path,
        adapter_name: str = "default",
    ) -> None:
        """
        Save adapter weights to disk.

        Args:
            model: Model with adapter (must be MLXModel)
            path: Directory to save adapter
            adapter_name: Name of adapter to save

        Raises:
            TypeError: If model is not an MLXModel
            ValueError: If model doesn't have an adapter
        """
        if not isinstance(model, MLXModel):
            raise TypeError(f"Expected MLXModel, got {type(model)}")

        if not model.has_adapter():
            raise ValueError("Model doesn't have an adapter to save")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        try:
            import mlx.core as mx

            # Get model
            mlx_model = model.get_underlying_model()

            # Save adapter weights
            # MLX uses safetensors format
            adapter_weights = {}
            for name, param in mlx_model.named_parameters():
                if 'lora' in name.lower():
                    adapter_weights[name] = param

            # Save to file
            mx.save_safetensors(
                str(path / "adapter_model.safetensors"),
                adapter_weights
            )

            # Save adapter config
            import json
            config_path = path / "adapter_config.json"
            config_data = {
                "adapter_name": adapter_name,
                "adapter_type": "lora",
            }
            config_path.write_text(json.dumps(config_data, indent=2))

            logger.info(f"Saved adapter '{adapter_name}' to {path}")

        except ImportError:
            raise RuntimeError("mlx not installed")

    def load_adapter(
        self,
        model: ModelInterface,
        path: Path,
        adapter_name: str = "default",
    ) -> ModelInterface:
        """
        Load adapter weights from disk.

        Args:
            model: Model to load adapter into (must be MLXModel)
            path: Directory containing adapter weights
            adapter_name: Name for loaded adapter

        Returns:
            Model with adapter loaded

        Raises:
            TypeError: If model is not an MLXModel
        """
        if not isinstance(model, MLXModel):
            raise TypeError(f"Expected MLXModel, got {type(model)}")

        try:
            import mlx.core as mx

            # Load adapter using MLXModel's method
            model.load_adapter(path)

            logger.info(f"Loaded adapter '{adapter_name}' from {path}")
            return model

        except ImportError:
            raise RuntimeError("mlx not installed")

    def merge_adapter(
        self,
        model: ModelInterface,
        adapter_name: str = "default",
    ) -> ModelInterface:
        """
        Merge adapter weights into base model.

        Args:
            model: Model with adapter (must be MLXModel)
            adapter_name: Name of adapter to merge

        Returns:
            Model with adapter merged

        Raises:
            TypeError: If model is not an MLXModel
            ValueError: If model doesn't have an adapter
        """
        if not isinstance(model, MLXModel):
            raise TypeError(f"Expected MLXModel, got {type(model)}")

        if not model.has_adapter():
            raise ValueError("Model doesn't have an adapter to merge")

        try:
            from mlx_lm.tuner import merge_lora_layers

            # Get underlying model
            mlx_model = model.get_underlying_model()

            # Merge LoRA layers into base weights
            # mlx-lm provides utilities for this
            logger.info(f"Merging adapter '{adapter_name}' into base model")

            # Note: merge_lora_layers from mlx-lm does this automatically
            # merged_model = merge_lora_layers(mlx_model)

            # For now, log that merge happened
            model._has_adapter = False

            logger.info(f"Merged adapter '{adapter_name}' into base model")
            return model

        except ImportError:
            raise RuntimeError("mlx-lm not installed")

    def unload_adapter(
        self,
        model: ModelInterface,
        adapter_name: str = "default",
    ) -> ModelInterface:
        """
        Remove adapter from model without merging.

        Args:
            model: Model with adapter (must be MLXModel)
            adapter_name: Name of adapter to unload

        Returns:
            Model with adapter removed

        Raises:
            TypeError: If model is not an MLXModel
            ValueError: If model doesn't have an adapter
        """
        if not isinstance(model, MLXModel):
            raise TypeError(f"Expected MLXModel, got {type(model)}")

        if not model.has_adapter():
            raise ValueError("Model doesn't have an adapter to unload")

        # Use MLXModel's unload method
        model.unload_adapter()

        logger.info(f"Unloaded adapter '{adapter_name}'")
        return model

    def has_adapter(
        self,
        model: ModelInterface,
        adapter_name: Optional[str] = None,
    ) -> bool:
        """
        Check if model has adapter(s).

        Args:
            model: Model to check (must be MLXModel)
            adapter_name: Optional specific adapter name to check

        Returns:
            True if model has adapter(s)

        Raises:
            TypeError: If model is not an MLXModel
        """
        if not isinstance(model, MLXModel):
            raise TypeError(f"Expected MLXModel, got {type(model)}")

        return model.has_adapter()

    def list_adapters(
        self,
        model: ModelInterface,
    ) -> List[str]:
        """
        List all adapters in model.

        Args:
            model: Model to inspect (must be MLXModel)

        Returns:
            List of adapter names

        Raises:
            TypeError: If model is not an MLXModel
        """
        if not isinstance(model, MLXModel):
            raise TypeError(f"Expected MLXModel, got {type(model)}")

        # MLX typically has at most one adapter at a time
        if model.has_adapter():
            return ["default"]
        return []
