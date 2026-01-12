"""
PyTorch SFT Trainer Wrapper - Wraps existing SFTTrainer for backend interface.

Adapts the src.training.sft_trainer.SFTTrainer to implement the
SFTTrainerInterface, providing backend abstraction.
"""

from typing import List, Dict, Any
from pathlib import Path
import logging

from ..trainer_interface import SFTTrainerInterface, TrainingResult
from ..model_interface import ModelInterface
from .pytorch_model import PyTorchModel
from .pytorch_device_manager import PyTorchDeviceManager
from .pytorch_adapter_manager import PyTorchAdapterManager
from ...training.sft_trainer import SFTTrainer, SFTConfig, SFTResult

logger = logging.getLogger(__name__)


class PyTorchSFTTrainer(SFTTrainerInterface):
    """
    PyTorch implementation of SFTTrainerInterface.

    Wraps the existing SFTTrainer class to provide backend abstraction.
    """

    def __init__(
        self,
        model: ModelInterface,
        config: Dict[str, Any],
        device_manager: PyTorchDeviceManager,
        adapter_manager: PyTorchAdapterManager,
    ):
        """
        Initialize PyTorch SFT trainer.

        Args:
            model: Model to train (must be PyTorchModel)
            config: Training configuration dict
            device_manager: Device manager for this backend
            adapter_manager: Adapter manager for this backend

        Raises:
            TypeError: If model is not a PyTorchModel
        """
        if not isinstance(model, PyTorchModel):
            raise TypeError(f"Expected PyTorchModel, got {type(model)}")

        self._pytorch_model = model
        self._device_manager = device_manager
        self._adapter_manager = adapter_manager

        # Convert dict config to SFTConfig
        self._config = self._convert_config(config)

        # Create internal SFTTrainer
        self._trainer = SFTTrainer(
            model=model.get_underlying_model(),
            tokenizer=model.tokenizer,
            config=self._config,
        )

        logger.info("Initialized PyTorchSFTTrainer")

    def _convert_config(self, config: Dict[str, Any]) -> SFTConfig:
        """
        Convert generic config dict to SFTConfig.

        Args:
            config: Configuration dictionary

        Returns:
            SFTConfig instance
        """
        # Extract SFT-specific parameters
        sft_config = SFTConfig(
            learning_rate=config.get("learning_rate", 3e-4),
            num_epochs=config.get("num_epochs", 3),
            per_device_batch_size=config.get("per_device_batch_size", 4),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            warmup_ratio=config.get("warmup_ratio", 0.1),
            max_seq_length=config.get("max_seq_length", 2048),
            lora_r=config.get("lora_r", 8),
            lora_alpha=config.get("lora_alpha", 16),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=config.get(
                "target_modules",
                ["q_proj", "v_proj", "o_proj"],
            ),
            device=self._device_manager.device_type,
            dtype=config.get("dtype", "float16"),
            output_dir=config.get("output_dir"),
            save_adapters=config.get("save_adapters", True),
        )

        logger.debug(f"Converted config: {sft_config}")
        return sft_config

    def _convert_result(self, sft_result: SFTResult) -> TrainingResult:
        """
        Convert SFTResult to TrainingResult.

        Args:
            sft_result: Result from SFTTrainer

        Returns:
            TrainingResult instance
        """
        return TrainingResult(
            success=sft_result.success,
            final_loss=sft_result.final_loss,
            training_time_seconds=sft_result.training_time_seconds,
            samples_trained=sft_result.samples_trained,
            adapter_path=sft_result.adapter_path,
            error_message=sft_result.error_message,
            metrics={},  # SFTResult doesn't have additional metrics
        )

    def train(self, train_data: List[Dict[str, Any]]) -> TrainingResult:
        """
        Train on Q&A data.

        Args:
            train_data: List of dicts with 'question'/'answer' or 'instruction'/'output' keys

        Returns:
            TrainingResult with training outcome
        """
        logger.info(f"Starting SFT training on {len(train_data)} samples")

        # Delegate to internal trainer
        sft_result = self._trainer.train(train_data)

        # Convert result
        result = self._convert_result(sft_result)

        if result.success:
            logger.info(
                f"SFT training completed: loss={result.final_loss:.4f}, "
                f"time={result.training_time_seconds:.1f}s"
            )
            # Update wrapped model with trained PEFT model
            trained_model = self._trainer.get_model()
            self._pytorch_model._model = trained_model
        else:
            logger.error(f"SFT training failed: {result.error_message}")

        return result

    def train_on_topic(
        self,
        topic_data: List[Dict[str, Any]],
        topic_id: str,
    ) -> TrainingResult:
        """
        Train on a single topic's data.

        Args:
            topic_data: Q&A pairs for this topic
            topic_id: Unique identifier for the topic (e.g., "unit-01/chapter-01/topic-01")

        Returns:
            TrainingResult with training outcome
        """
        logger.info(f"Training on topic: {topic_id}")

        # Delegate to internal trainer
        sft_result = self._trainer.train_on_topic(topic_data, topic_id)

        # Convert result
        result = self._convert_result(sft_result)

        if result.success:
            logger.info(
                f"Topic {topic_id} SFT complete: loss={result.final_loss:.4f}"
            )
            # Update wrapped model
            trained_model = self._trainer.get_model()
            self._pytorch_model._model = trained_model
        else:
            logger.error(f"Topic {topic_id} SFT failed: {result.error_message}")

        return result

    def save_adapter(self, path: Path) -> Path:
        """
        Save the LoRA adapter.

        Args:
            path: Directory to save adapter

        Returns:
            Path to saved adapter
        """
        logger.info(f"Saving SFT adapter to {path}")
        return self._trainer.save_adapter(path)

    def get_model(self) -> ModelInterface:
        """
        Get the trained model.

        Returns:
            PyTorchModel with trained adapter
        """
        return self._pytorch_model
