"""
PyTorch Backend - Main backend implementation for PyTorch/transformers.

Implements the BackendProtocol for PyTorch-based training with
transformers, PEFT, and TRL libraries.
"""

from typing import Any, Dict, Optional
from pathlib import Path
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..protocol import BackendProtocol, BackendConfig
from ..model_interface import ModelInterface
from ..trainer_interface import TrainerInterface
from .pytorch_model import PyTorchModel
from .pytorch_device_manager import PyTorchDeviceManager
from .pytorch_adapter_manager import PyTorchAdapterManager

logger = logging.getLogger(__name__)


class PyTorchBackend(BackendProtocol):
    """
    PyTorch implementation of BackendProtocol.

    Uses transformers for model loading, PEFT for LoRA adapters,
    and TRL for DPO training.
    """

    def __init__(self, config: BackendConfig):
        """
        Initialize PyTorch backend.

        Args:
            config: Backend configuration
        """
        self._config = config
        self._device_manager = PyTorchDeviceManager(config.device)
        self._adapter_manager = PyTorchAdapterManager()

        # Set random seed
        if config.seed is not None:
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)

        logger.info(
            f"Initialized PyTorch backend "
            f"(device={self._device_manager.device_type}, "
            f"dtype={config.dtype}, "
            f"seed={config.seed})"
        )

    @property
    def name(self) -> str:
        """Backend name."""
        return "pytorch"

    @property
    def config(self) -> BackendConfig:
        """Backend configuration."""
        return self._config

    def load_model(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
    ) -> ModelInterface:
        """
        Load a model from path.

        Args:
            model_path: Path to model weights or HuggingFace model ID
            tokenizer_path: Optional path to tokenizer (defaults to model_path)
            adapter_path: Optional path to LoRA adapter to load

        Returns:
            PyTorchModel instance wrapping the loaded model

        Raises:
            Exception: If model loading fails
        """
        logger.info(f"Loading model from {model_path}")

        # Resolve tokenizer path
        if tokenizer_path is None:
            tokenizer_path = model_path

        # Map dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self._config.dtype, torch.float16)

        try:
            # Load tokenizer
            logger.info(f"Loading tokenizer from {tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
            )

            # Ensure padding token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")

            # Create quantization config if requested
            quantization_config = None
            if self._config.quantization is not None:
                logger.info(f"Setting up {self._config.quantization} quantization")
                if self._config.quantization == "int4":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,  # Nested quantization
                    )
                elif self._config.quantization == "int8":
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                else:
                    raise ValueError(
                        f"Unsupported quantization: {self._config.quantization}. "
                        f"Supported: 'int4', 'int8', None"
                    )

            # Determine device_map strategy
            # Use "auto" for quantization (distributes across GPUs automatically)
            # Use None for manual placement otherwise
            device_map = "auto" if quantization_config is not None else None

            # Load base model
            logger.info(f"Loading base model with dtype={torch_dtype}, device_map={device_map}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                device_map=device_map,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
            )

            # Move to device if not using auto device_map
            if device_map is None:
                device = self._device_manager.get_device()
                model = model.to(device)
                logger.info(f"Moved model to {device}")
            else:
                logger.info(f"Model loaded with automatic device placement")

            # Wrap in PyTorchModel
            wrapped_model = PyTorchModel(model, tokenizer)

            # Load adapter if provided
            if adapter_path is not None:
                logger.info(f"Loading adapter from {adapter_path}")
                wrapped_model = self._adapter_manager.load_adapter(
                    wrapped_model,
                    Path(adapter_path),
                )

            logger.info(
                f"Model loaded successfully "
                f"({wrapped_model.num_parameters:,} parameters, "
                f"{wrapped_model.num_trainable_parameters:,} trainable)"
            )

            return wrapped_model

        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def create_sft_trainer(
        self,
        model: ModelInterface,
        config: Dict[str, Any],
    ) -> TrainerInterface:
        """
        Create an SFT (Supervised Fine-Tuning) trainer.

        Args:
            model: Model to train (must be PyTorchModel)
            config: Training configuration dict

        Returns:
            PyTorchSFTTrainer instance

        Raises:
            TypeError: If model is not a PyTorchModel
        """
        from .pytorch_sft_trainer import PyTorchSFTTrainer

        if not isinstance(model, PyTorchModel):
            raise TypeError(f"Expected PyTorchModel, got {type(model)}")

        logger.info("Creating SFT trainer")
        return PyTorchSFTTrainer(model, config, self._device_manager, self._adapter_manager)

    def create_dpo_trainer(
        self,
        model: ModelInterface,
        config: Dict[str, Any],
        ref_model: Optional[ModelInterface] = None,
    ) -> TrainerInterface:
        """
        Create a DPO (Direct Preference Optimization) trainer.

        Args:
            model: Policy model to train (must be PyTorchModel)
            config: DPO configuration dict
            ref_model: Optional reference model (defaults to copy of model)

        Returns:
            PyTorchDPOTrainer instance

        Raises:
            TypeError: If model is not a PyTorchModel
        """
        from .pytorch_dpo_trainer import PyTorchDPOTrainer

        if not isinstance(model, PyTorchModel):
            raise TypeError(f"Expected PyTorchModel, got {type(model)}")

        if ref_model is not None and not isinstance(ref_model, PyTorchModel):
            raise TypeError(f"Expected PyTorchModel for ref_model, got {type(ref_model)}")

        logger.info("Creating DPO trainer")
        return PyTorchDPOTrainer(
            model, config, self._device_manager, self._adapter_manager, ref_model
        )

    def get_device_manager(self) -> PyTorchDeviceManager:
        """
        Get device management utilities for this backend.

        Returns:
            PyTorchDeviceManager instance
        """
        return self._device_manager

    def get_adapter_manager(self) -> PyTorchAdapterManager:
        """
        Get adapter management utilities for this backend.

        Returns:
            PyTorchAdapterManager instance
        """
        return self._adapter_manager
