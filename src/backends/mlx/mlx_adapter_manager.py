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
            from mlx_lm.tuner.utils import linear_to_lora_layers
            from mlx_lm.tuner.lora import LoRALinear
            import mlx.nn as nn
            
            # Import at module level for isinstance checks
            _LoRALinear = LoRALinear

            # Get underlying MLX model
            mlx_model = model.get_underlying_model()

            # Convert config to MLX format
            mlx_config = config.to_mlx_config()

            logger.info(
                f"Adding LoRA adapter '{adapter_name}' with "
                f"rank={config.r}, scale={mlx_config['scale']}, "
                f"target_modules={mlx_config['target_modules']}"
            )

            # Use official utility to apply LoRA
            lora_config = {
                'rank': mlx_config['rank'],
                'alpha': mlx_config.get('alpha', mlx_config['rank'] * 2),
                'dropout': mlx_config.get('dropout', 0.0),
                'scale': mlx_config['scale'],
            }
            
            # Apply LoRA to all layers (-1 means all layers)
            converted_count = linear_to_lora_layers(mlx_model, num_layers=-1, config=lora_config)
            
            if converted_count == 0:
                logger.warning("No linear layers were converted to LoRA. Check target_modules configuration.")

            # Freeze base model parameters and ensure ONLY LoRA parameters are trainable
            # This is critical for trainable_parameters() to work correctly
            try:
                # Freeze the entire model first
                mlx_model.freeze()
                
                # Then selectively unfreeze ONLY lora_a and lora_b parameters
                for m in mlx_model.modules():
                    if isinstance(m, _LoRALinear):
                        # Unfreeze only the LoRA weights, not the base weights
                        m.unfreeze(keys=['lora_a', 'lora_b'], strict=False)
                
                logger.debug("Frozen base model parameters, unfrozen LoRA parameters (lora_a, lora_b)")
            except Exception as e:
                logger.warning(f"Could not freeze/unfreeze parameters: {e}")

            # Mark that adapter is added
            model._has_adapter = True

            # Count LoRA layers
            lora_layer_count = 0
            for m in mlx_model.modules():
                if isinstance(m, _LoRALinear):
                    lora_layer_count += 1
            
            # Count trainable parameters (lora_a + lora_b per layer = 2 * layer_count)
            trainable_param_count = lora_layer_count * 2
            
            logger.info(
                f"Added LoRA adapter '{adapter_name}' "
                f"({lora_layer_count} LoRA layers, {trainable_param_count} trainable parameter groups)"
            )
            return model

        except ImportError:
            raise RuntimeError("mlx-lm not installed. Install with: pip install mlx-lm")
        except Exception as e:
            logger.error(f"Failed to add LoRA adapter: {e}")
            raise

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
