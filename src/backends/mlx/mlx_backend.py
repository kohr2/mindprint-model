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

    def create_sft_trainer(
        self,
        model: ModelInterface,
        config: Dict[str, Any],
    ) -> TrainerInterface:
        """
        Create an SFT (Supervised Fine-Tuning) trainer.

        Args:
            model: Model to train (must be MLXModel)
            config: Training configuration dict

        Returns:
            MLXSFTTrainer instance

        Raises:
            TypeError: If model is not an MLXModel
        """
        from .mlx_sft_trainer import MLXSFTTrainer
        from ..adapter_interface import AdapterConfig
        from src.models.config import get_model_config

        if not isinstance(model, MLXModel):
            raise TypeError(f"Expected MLXModel, got {type(model)}")

        # Add LoRA adapter if not already present
        if not model.has_adapter():
            # Get model name from model_path or config
            model_name = getattr(model, '_model_path', None) or config.get('model_name', 'qwen-7b')
            
            # Map HuggingFace model name to config name
            model_name = self._map_model_name(model_name)
            
            try:
                model_cfg = get_model_config(model_name)
                
                adapter_config = AdapterConfig(
                    r=model_cfg.lora.r,
                    alpha=model_cfg.lora.alpha,
                    dropout=model_cfg.lora.dropout,
                    target_modules=model_cfg.lora.target_modules,
                )
                
                logger.info(f"Adding LoRA adapter: rank={adapter_config.r}, alpha={adapter_config.alpha}")
                model = self._adapter_manager.add_adapter(model, adapter_config)
            except (KeyError, AttributeError) as e:
                logger.warning(f"Could not load model config for {model_name}: {e}")
                logger.warning("Using default LoRA config")
                # Fallback to default config
                adapter_config = AdapterConfig(
                    r=8,
                    alpha=16.0,
                    dropout=0.05,
                    target_modules=["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
                )
                model = self._adapter_manager.add_adapter(model, adapter_config)

        logger.info("Creating MLX SFT trainer")
        return MLXSFTTrainer(
            model, config, self._device_manager, self._adapter_manager
        )

    def _map_model_name(self, model_path: str) -> str:
        """
        Map HuggingFace model path to internal config name.
        
        Args:
            model_path: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-7B-Instruct")
            
        Returns:
            Internal model config name (e.g., "qwen-7b")
        """
        model_lower = model_path.lower()
        
        # Map common patterns
        if "qwen" in model_lower:
            if "72" in model_lower or "72b" in model_lower:
                return "qwen-72b"
            else:
                return "qwen-7b"
        elif "gemma" in model_lower:
            return "gemma-12b"
        else:
            # Default fallback
            return "qwen-7b"

    def create_dpo_trainer(
        self,
        model: ModelInterface,
        config: Dict[str, Any],
        ref_model: Optional[ModelInterface] = None,
    ) -> TrainerInterface:
        """
        Create a DPO (Direct Preference Optimization) trainer.

        Args:
            model: Policy model to train (must be MLXModel)
            config: DPO configuration dict
            ref_model: Optional reference model (defaults to copy of model)

        Returns:
            MLXDPOTrainer instance

        Raises:
            TypeError: If model is not an MLXModel
        """
        from .mlx_dpo_trainer import MLXDPOTrainer

        if not isinstance(model, MLXModel):
            raise TypeError(f"Expected MLXModel, got {type(model)}")

        if ref_model is not None and not isinstance(ref_model, MLXModel):
            raise TypeError(f"Expected MLXModel for ref_model, got {type(ref_model)}")

        logger.info("Creating MLX DPO trainer")
        return MLXDPOTrainer(
            model, config, self._device_manager, self._adapter_manager, ref_model
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
