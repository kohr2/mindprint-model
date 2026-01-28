"""
PyTorch Adapter Manager - PEFT adapter management for PyTorch backend.

Provides LoRA adapter operations using the PEFT library.
"""

from typing import List, Optional
from pathlib import Path
import logging
import torch
from peft import get_peft_model, LoraConfig, PeftModel

from ..adapter_interface import AdapterManager, AdapterConfig
from ..model_interface import ModelInterface
from .pytorch_model import PyTorchModel

logger = logging.getLogger(__name__)


class PyTorchAdapterManager(AdapterManager):
    """
    PyTorch implementation of AdapterManager using PEFT.

    Handles LoRA adapter operations for PyTorch/transformers models.
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
            model: Model to add adapter to (must be PyTorchModel)
            config: Adapter configuration
            adapter_name: Name for this adapter

        Returns:
            Model with adapter attached (new PyTorchModel wrapper)

        Raises:
            TypeError: If model is not a PyTorchModel
        """
        if not isinstance(model, PyTorchModel):
            raise TypeError(f"Expected PyTorchModel, got {type(model)}")

        # Get underlying transformers model
        base_model = model.get_underlying_model()

        # Check if already a PEFT model
        if isinstance(base_model, PeftModel):
            logger.info(f"Model already has adapters, adding new adapter: {adapter_name}")
            # Add adapter to existing PEFT model
            peft_config = LoraConfig(**config.to_peft_config())
            base_model.add_adapter(adapter_name, peft_config)
            base_model.set_adapter(adapter_name)
        else:
            # Create new PEFT model
            logger.info(f"Adding first adapter to model: {adapter_name}")
            peft_config = LoraConfig(**config.to_peft_config())
            base_model = get_peft_model(base_model, peft_config, adapter_name=adapter_name)

        # Return new wrapper with PEFT model
        return PyTorchModel(base_model, model.tokenizer)

    def save_adapter(
        self,
        model: ModelInterface,
        path: Path,
        adapter_name: str = "default",
    ) -> None:
        """
        Save adapter weights to disk.

        Args:
            model: Model with adapter (must be PyTorchModel)
            path: Directory to save adapter
            adapter_name: Name of adapter to save

        Raises:
            TypeError: If model is not a PyTorchModel
            ValueError: If model doesn't have the specified adapter
        """
        if not isinstance(model, PyTorchModel):
            raise TypeError(f"Expected PyTorchModel, got {type(model)}")

        if not model.has_adapter():
            raise ValueError("Model doesn't have any adapters to save")

        base_model = model.get_underlying_model()

        if not isinstance(base_model, PeftModel):
            raise ValueError("Model is not a PEFT model")

        # Check if adapter exists
        if adapter_name not in base_model.peft_config:
            available = list(base_model.peft_config.keys())
            raise ValueError(
                f"Adapter '{adapter_name}' not found. Available: {available}"
            )

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Set active adapter and save
        base_model.set_adapter(adapter_name)
        base_model.save_pretrained(str(path))

        logger.info(f"Saved adapter '{adapter_name}' to {path}")

    def load_adapter(
        self,
        model: ModelInterface,
        path: Path,
        adapter_name: str = "default",
    ) -> ModelInterface:
        """
        Load adapter weights from disk.

        Args:
            model: Model to load adapter into (must be PyTorchModel)
            path: Directory containing adapter weights
            adapter_name: Name for loaded adapter

        Returns:
            Model with adapter loaded

        Raises:
            TypeError: If model is not a PyTorchModel
        """
        if not isinstance(model, PyTorchModel):
            raise TypeError(f"Expected PyTorchModel, got {type(model)}")

        base_model = model.get_underlying_model()

        if isinstance(base_model, PeftModel):
            # Load into existing PEFT model
            logger.info(f"Loading adapter '{adapter_name}' into existing PEFT model")
            base_model.load_adapter(str(path), adapter_name=adapter_name)
            base_model.set_adapter(adapter_name)
        else:
            # Create PEFT model from adapter
            logger.info(f"Creating PEFT model with adapter '{adapter_name}'")
            base_model = PeftModel.from_pretrained(
                base_model,
                str(path),
                adapter_name=adapter_name,
            )

        return PyTorchModel(base_model, model.tokenizer)

    def merge_adapter(
        self,
        model: ModelInterface,
        adapter_name: str = "default",
    ) -> ModelInterface:
        """
        Merge adapter weights into base model.

        WARNING: On PyTorch MPS backend, this may cause model corruption
        due to non-contiguous tensor bugs. Prefer save/reload strategy.

        Args:
            model: Model with adapter (must be PyTorchModel)
            adapter_name: Name of adapter to merge

        Returns:
            Model with adapter merged (no longer attached as adapter)

        Raises:
            TypeError: If model is not a PyTorchModel
            ValueError: If model doesn't have adapters
        """
        if not isinstance(model, PyTorchModel):
            raise TypeError(f"Expected PyTorchModel, got {type(model)}")

        if not model.has_adapter():
            raise ValueError("Model doesn't have adapters to merge")

        base_model = model.get_underlying_model()

        if not isinstance(base_model, PeftModel):
            raise ValueError("Model is not a PEFT model")

        # Log MPS warning
        if model.device.type == "mps":
            logger.warning(
                "Merging adapter on MPS device! This is known to cause corruption "
                "due to PyTorch MPS non-contiguous tensor bugs. "
                "Consider using save/reload strategy instead."
            )

        # Set active adapter and merge
        base_model.set_adapter(adapter_name)
        merged_model = base_model.merge_and_unload()

        logger.info(f"Merged adapter '{adapter_name}' into base model")

        return PyTorchModel(merged_model, model.tokenizer)

    def unload_adapter(
        self,
        model: ModelInterface,
        adapter_name: str = "default",
    ) -> ModelInterface:
        """
        Remove adapter from model without merging.

        WARNING: On PyTorch MPS backend, this may cause issues. Consider
        using merge or save/reload instead.

        Args:
            model: Model with adapter (must be PyTorchModel)
            adapter_name: Name of adapter to unload

        Returns:
            Model with adapter removed

        Raises:
            TypeError: If model is not a PyTorchModel
            ValueError: If model doesn't have the specified adapter
        """
        if not isinstance(model, PyTorchModel):
            raise TypeError(f"Expected PyTorchModel, got {type(model)}")

        if not model.has_adapter():
            raise ValueError("Model doesn't have adapters to unload")

        base_model = model.get_underlying_model()

        if not isinstance(base_model, PeftModel):
            raise ValueError("Model is not a PEFT model")

        # Log MPS warning
        if model.device.type == "mps":
            logger.warning(
                "Unloading adapter on MPS device! This may cause issues. "
                "Consider save/reload strategy instead."
            )

        # Delete adapter
        if adapter_name in base_model.peft_config:
            base_model.delete_adapter(adapter_name)
            logger.info(f"Unloaded adapter '{adapter_name}'")

            # If no adapters left, unload PEFT wrapper
            if not base_model.peft_config:
                base_model = base_model.unload()
                logger.info("All adapters removed, unwrapped PEFT model")
        else:
            available = list(base_model.peft_config.keys())
            raise ValueError(
                f"Adapter '{adapter_name}' not found. Available: {available}"
            )

        return PyTorchModel(base_model, model.tokenizer)

    def has_adapter(
        self,
        model: ModelInterface,
        adapter_name: Optional[str] = None,
    ) -> bool:
        """
        Check if model has adapter(s).

        Args:
            model: Model to check (must be PyTorchModel)
            adapter_name: Optional specific adapter name to check

        Returns:
            True if model has adapter(s)

        Raises:
            TypeError: If model is not a PyTorchModel
        """
        if not isinstance(model, PyTorchModel):
            raise TypeError(f"Expected PyTorchModel, got {type(model)}")

        base_model = model.get_underlying_model()

        if not isinstance(base_model, PeftModel):
            return False

        if adapter_name is None:
            # Check if any adapters
            return bool(base_model.peft_config)

        # Check specific adapter
        return adapter_name in base_model.peft_config

    def list_adapters(
        self,
        model: ModelInterface,
    ) -> List[str]:
        """
        List all adapters in model.

        Args:
            model: Model to inspect (must be PyTorchModel)

        Returns:
            List of adapter names

        Raises:
            TypeError: If model is not a PyTorchModel
        """
        if not isinstance(model, PyTorchModel):
            raise TypeError(f"Expected PyTorchModel, got {type(model)}")

        base_model = model.get_underlying_model()

        if not isinstance(base_model, PeftModel):
            return []

        return list(base_model.peft_config.keys())
