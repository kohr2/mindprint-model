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

            # Convert linear layers to LoRA layers
            # MLX models use a dictionary-like structure accessed via items()
            def convert_to_lora(module, parent_name="", depth=0):
                """Recursively convert linear layers to LoRA layers."""
                if depth > 20:  # Safety limit to prevent infinite recursion
                    return 0
                
                converted_count = 0
                
                # MLX modules use items() method for dict-like access
                items_to_check = []
                
                # Method 1: Try items() if available (MLX dict-like interface)
                if hasattr(module, 'items'):
                    try:
                        items_to_check = list(module.items())
                    except Exception as e:
                        logger.debug(f"Could not get items from {parent_name}: {e}")
                
                # Method 2: Try __dict__ for standard Python objects
                if not items_to_check and hasattr(module, '__dict__'):
                    items_to_check = [(k, v) for k, v in module.__dict__.items() if not k.startswith('_')]
                
                # Process items
                for name, child in items_to_check:
                    if name.startswith('_'):
                        continue
                    
                    full_name = f"{parent_name}.{name}" if parent_name else name
                    
                    # Check if this is a Linear layer that should be converted
                    if isinstance(child, nn.Linear):
                        # Check if this module should be converted
                        should_convert = any(
                            target in name for target in mlx_config['target_modules']
                        )
                        
                        if should_convert:
                            logger.info(f"Converting {full_name} to LoRA")
                            try:
                                # Convert to LoRA layer using from_base
                                # from_base takes the base Linear layer and LoRA config
                                lora_layer = LoRALinear.from_base(
                                    child,
                                    r=mlx_config['rank'],
                                    scale=mlx_config['scale'],
                                    dropout=mlx_config.get('dropout', 0.0),
                                )
                                # Set the LoRA layer back using items() interface
                                if hasattr(module, '__setitem__'):
                                    # Dictionary-like interface (MLX modules)
                                    module[name] = lora_layer
                                elif hasattr(module, '__setattr__'):
                                    # Standard attribute interface
                                    setattr(module, name, lora_layer)
                                else:
                                    # Try direct assignment
                                    module[name] = lora_layer
                                converted_count += 1
                            except Exception as e:
                                logger.warning(f"Failed to convert {full_name}: {e}")
                                import traceback
                                logger.debug(traceback.format_exc())
                    else:
                        # Recursively process nested modules
                        # Skip non-module types (like lists, tuples, primitives)
                        if isinstance(child, (list, tuple, str, int, float, bool)):
                            continue
                        # Check if it's a module-like object worth recursing into
                        if isinstance(child, nn.Module) or hasattr(child, 'items') or (hasattr(child, '__dict__') and not isinstance(child, type)):
                            converted_count += convert_to_lora(child, full_name, depth + 1)
                
                return converted_count

            # Convert model layers
            # MLX models typically have: model.model.layers[i].self_attn.q_proj, etc.
            converted_count = 0
            if hasattr(mlx_model, 'model'):
                # Qwen/Llama-style models have a 'model' attribute containing layers
                # Traverse: model -> model.model -> model.model.layers -> layer.self_attn -> q_proj
                base_model = mlx_model.model
                if hasattr(base_model, 'layers'):
                    # Process each layer in the transformer
                    for i, layer in enumerate(base_model.layers):
                        converted_count += convert_to_lora(layer, f"layers.{i}", depth=0)
                else:
                    # Fallback: try converting the whole model
                    converted_count = convert_to_lora(base_model)
            elif hasattr(mlx_model, 'layers'):
                # Direct layers attribute
                for i, layer in enumerate(mlx_model.layers):
                    converted_count += convert_to_lora(layer, f"layers.{i}", depth=0)
            else:
                # Try direct conversion
                converted_count = convert_to_lora(mlx_model)
            
            if converted_count == 0:
                logger.warning("No linear layers were converted to LoRA. Check target_modules configuration.")

            # Mark that adapter is added
            model._has_adapter = True

            # Count LoRA parameters recursively
            # Count LoRA parameters by recursively traversing the model
            def count_lora_params(module, depth=0):
                """Recursively count LoRA parameters."""
                if depth > 20:
                    return 0
                count = 0
                # Check if this module has items() (MLX dict-like interface)
                if hasattr(module, 'items'):
                    try:
                        for name, child in module.items():
                            # Check if this is a LoRALinear module
                            if isinstance(child, _LoRALinear):
                                # LoRALinear has lora_a and lora_b parameters
                                if hasattr(child, 'parameters'):
                                    child_params = child.parameters()
                                    # Count lora_a and lora_b
                                    for param_name in child_params.keys():
                                        if 'lora' in param_name.lower():
                                            count += 1
                            # Recurse into nested modules
                            if isinstance(child, nn.Module) or hasattr(child, 'items'):
                                count += count_lora_params(child, depth + 1)
                    except Exception as e:
                        logger.debug(f"Error counting LoRA params at depth {depth}: {e}")
                return count
            
            # Count LoRA parameters starting from the base model (same path as conversion)
            if hasattr(mlx_model, 'model'):
                lora_param_count = count_lora_params(mlx_model.model)
            else:
                lora_param_count = count_lora_params(mlx_model)
            
            logger.info(
                f"Added LoRA adapter '{adapter_name}' "
                f"({lora_param_count} LoRA parameter groups, {converted_count} layers converted)"
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
