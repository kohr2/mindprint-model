"""
MLX Backend - Main backend implementation for MLX.

Implements the BackendProtocol for MLX-based training with mlx-lm.
"""

from typing import Any, Dict, Optional
from pathlib import Path
import logging

from ..protocol import BackendProtocol, BackendConfig
from ..model_interface import ModelInterface
from ..trainer_interface import TrainerInterface
from .mlx_model import MLXModel
from .mlx_device_manager import MLXDeviceManager
from .mlx_adapter_manager import MLXAdapterManager

logger = logging.getLogger(__name__)


class MLXBackend(BackendProtocol):
    """
    MLX implementation of BackendProtocol.

    Uses mlx-lm for model loading and training on Apple Silicon.
    """

    def __init__(self, config: BackendConfig):
        """
        Initialize MLX backend.

        Args:
            config: Backend configuration
        """
        self._config = config
        self._device_manager = MLXDeviceManager(config.device)
        self._adapter_manager = MLXAdapterManager()

        # Set random seed
        if config.seed is not None:
            try:
                import mlx.core as mx
                mx.random.seed(config.seed)
                logger.info(f"Set MLX random seed to {config.seed}")
            except ImportError:
                logger.warning("mlx not installed, cannot set seed")

        logger.info(
            f"Initialized MLX backend "
            f"(device={self._device_manager.device_type}, "
            f"dtype={config.dtype}, "
            f"seed={config.seed})"
        )

    @property
    def name(self) -> str:
        """Backend name."""
        return "mlx"

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
            MLXModel instance wrapping the loaded model

        Raises:
            Exception: If model loading fails
        """
        logger.info(f"Loading model from {model_path}")

        # Resolve tokenizer path
        if tokenizer_path is None:
            tokenizer_path = model_path

        try:
            from mlx_lm import load as mlx_load
            from transformers import AutoTokenizer

            # Load tokenizer (use HuggingFace)
            logger.info(f"Loading tokenizer from {tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
            )

            # Ensure padding token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")

            # Load MLX model
            logger.info(f"Loading MLX model")
            model, _ = mlx_load(model_path)

            # Wrap in MLXModel (store model_path for config lookup)
            wrapped_model = MLXModel(model, tokenizer, model_path=model_path)

            # Load adapter if provided
            if adapter_path is not None:
                logger.info(f"Loading adapter from {adapter_path}")
                wrapped_model = self._adapter_manager.load_adapter(
                    wrapped_model,
                    Path(adapter_path),
                )

            logger.info(
                f"Model loaded successfully "
                f"({wrapped_model.num_parameters:,} parameters)"
            )

            return wrapped_model

        except ImportError as e:
            logger.error(f"MLX dependencies not installed: {e}")
            raise RuntimeError(
                "mlx-lm not installed. Install with: "
                "pip install mlx mlx-lm"
            )
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def create_orpo_trainer(
        self,
        model: ModelInterface,
        config: Dict[str, Any],
    ) -> TrainerInterface:
        """
        Create an ORPO (Odds Ratio Preference Optimization) trainer.

        Args:
            model: Policy model to train (must be MLXModel)
            config: ORPO configuration dict

        Returns:
            MLXORPOTrainer instance

        Raises:
            TypeError: If model is not an MLXModel
        """
        from .mlx_orpo_trainer import MLXORPOTrainer

        if not isinstance(model, MLXModel):
            raise TypeError(f"Expected MLXModel, got {type(model)}")

        logger.info("Creating MLX ORPO trainer")
        return MLXORPOTrainer(
            model, config, self._device_manager, self._adapter_manager
        )

    def get_device_manager(self) -> MLXDeviceManager:
        """
        Get device management utilities for this backend.

        Returns:
            MLXDeviceManager instance
        """
        return self._device_manager

    def get_adapter_manager(self) -> MLXAdapterManager:
        """
        Get adapter management utilities for this backend.

        Returns:
            MLXAdapterManager instance
        """
        return self._adapter_manager
