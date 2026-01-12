"""
PyTorch DPO Trainer Wrapper - Wraps existing Rank1DPOTrainer for backend interface.

Adapts the src.training.dpo_trainer.Rank1DPOTrainer to implement the
DPOTrainerInterface, providing backend abstraction.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from ..trainer_interface import DPOTrainerInterface, TrainingResult
from ..model_interface import ModelInterface
from .pytorch_model import PyTorchModel
from .pytorch_device_manager import PyTorchDeviceManager
from .pytorch_adapter_manager import PyTorchAdapterManager
from ...training.dpo_trainer import Rank1DPOTrainer, Rank1DPOConfig, DPOResult

logger = logging.getLogger(__name__)


class PyTorchDPOTrainer(DPOTrainerInterface):
    """
    PyTorch implementation of DPOTrainerInterface.

    Wraps the existing Rank1DPOTrainer class to provide backend abstraction.
    """

    def __init__(
        self,
        model: ModelInterface,
        config: Dict[str, Any],
        device_manager: PyTorchDeviceManager,
        adapter_manager: PyTorchAdapterManager,
        ref_model: Optional[ModelInterface] = None,
    ):
        """
        Initialize PyTorch DPO trainer.

        Args:
            model: Policy model to train (must be PyTorchModel)
            config: Training configuration dict
            device_manager: Device manager for this backend
            adapter_manager: Adapter manager for this backend
            ref_model: Optional reference model (must be PyTorchModel if provided)

        Raises:
            TypeError: If model or ref_model is not a PyTorchModel
        """
        if not isinstance(model, PyTorchModel):
            raise TypeError(f"Expected PyTorchModel, got {type(model)}")

        if ref_model is not None and not isinstance(ref_model, PyTorchModel):
            raise TypeError(f"Expected PyTorchModel for ref_model, got {type(ref_model)}")

        self._pytorch_model = model
        self._pytorch_ref_model = ref_model
        self._device_manager = device_manager
        self._adapter_manager = adapter_manager

        # Convert dict config to Rank1DPOConfig
        self._config = self._convert_config(config)

        # Get underlying transformers models
        underlying_model = model.get_underlying_model()
        underlying_ref_model = ref_model.get_underlying_model() if ref_model else None

        # Create internal Rank1DPOTrainer
        self._trainer = Rank1DPOTrainer(
            model=underlying_model,
            tokenizer=model.tokenizer,
            config=self._config,
            ref_model=underlying_ref_model,
        )

        logger.info("Initialized PyTorchDPOTrainer")

    def _convert_config(self, config: Dict[str, Any]) -> Rank1DPOConfig:
        """
        Convert generic config dict to Rank1DPOConfig.

        Args:
            config: Configuration dictionary

        Returns:
            Rank1DPOConfig instance
        """
        # Determine if using MPS
        use_mps = self._device_manager.is_mps

        # Extract DPO-specific parameters
        dpo_config = Rank1DPOConfig(
            beta=config.get("beta", 0.1),
            learning_rate=config.get("learning_rate", 5e-7),
            max_steps=config.get("max_steps", 100),
            per_device_batch_size=config.get("per_device_batch_size", 2),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            max_prompt_length=config.get("max_prompt_length", 512),
            max_length=config.get("max_length", 1024),
            lora_r=config.get("lora_r", 1),
            lora_alpha=config.get("lora_alpha", 1.0),
            lora_dropout=config.get("lora_dropout", 0.0),
            target_modules=config.get(
                "target_modules",
                ["o_proj", "v_proj", "up_proj", "down_proj"],
            ),
            use_mps=use_mps,
            fp16=config.get("fp16", False),  # MPS doesn't support fp16 flag
            bf16=config.get("bf16", False),
            ref_model_strategy=config.get("ref_model_strategy", "shared"),
            output_dir=config.get("output_dir", "./dpo_output"),
            logging_steps=config.get("logging_steps", 10),
        )

        logger.debug(f"Converted DPO config: {dpo_config}")
        return dpo_config

    def _convert_result(self, dpo_result: DPOResult) -> TrainingResult:
        """
        Convert DPOResult to TrainingResult.

        Args:
            dpo_result: Result from Rank1DPOTrainer

        Returns:
            TrainingResult instance
        """
        return TrainingResult(
            success=dpo_result.success,
            final_loss=dpo_result.final_loss,
            training_time_seconds=dpo_result.training_time_seconds,
            samples_trained=dpo_result.steps_completed,  # Steps as proxy for samples
            adapter_path=dpo_result.adapter_path,
            error_message=dpo_result.error_message,
            metrics={
                "chosen_rewards_mean": dpo_result.chosen_rewards_mean,
                "rejected_rewards_mean": dpo_result.rejected_rewards_mean,
                "reward_margin": dpo_result.reward_margin,
                "steps_completed": dpo_result.steps_completed,
            },
        )

    def train(self, train_data: List[Dict[str, Any]]) -> TrainingResult:
        """
        Train with DPO on preference pairs.

        Args:
            train_data: List of dicts with "prompt", "chosen", "rejected" keys

        Returns:
            TrainingResult with training outcome
        """
        logger.info(f"Starting DPO training on {len(train_data)} preference pairs")

        # Delegate to internal trainer
        dpo_result = self._trainer.train(train_data)

        # Convert result
        result = self._convert_result(dpo_result)

        if result.success:
            logger.info(
                f"DPO training completed: loss={result.final_loss:.4f}, "
                f"margin={result.metrics['reward_margin']:.4f}, "
                f"time={result.training_time_seconds:.1f}s"
            )
            # Update wrapped model with trained PEFT model
            trained_model = self._trainer.get_model()
            self._pytorch_model._model = trained_model
        else:
            logger.error(f"DPO training failed: {result.error_message}")

        return result

    def train_on_topic(
        self,
        topic_data: List[Dict[str, Any]],
        topic_id: str,
    ) -> TrainingResult:
        """
        Train on a single topic's preference pairs.

        Args:
            topic_data: Preference pairs for this topic
            topic_id: Unique identifier for the topic (e.g., "unit-01/chapter-01/topic-01")

        Returns:
            TrainingResult with training outcome
        """
        logger.info(f"DPO training on topic: {topic_id}")

        # Delegate to internal trainer
        dpo_result = self._trainer.train_on_topic(topic_data, topic_id)

        # Convert result
        result = self._convert_result(dpo_result)

        if result.success:
            logger.info(
                f"Topic {topic_id} DPO complete: loss={result.final_loss:.4f}, "
                f"margin={result.metrics['reward_margin']:.4f}"
            )
            # Update wrapped model
            trained_model = self._trainer.get_model()
            self._pytorch_model._model = trained_model
        else:
            logger.error(f"Topic {topic_id} DPO failed: {result.error_message}")

        return result

    def save_adapter(self, path: Path) -> Path:
        """
        Save the DPO LoRA adapter.

        Args:
            path: Directory to save adapter

        Returns:
            Path to saved adapter
        """
        logger.info(f"Saving DPO adapter to {path}")
        return self._trainer.save_adapter(str(path))

    def get_model(self) -> ModelInterface:
        """
        Get the trained policy model.

        Returns:
            PyTorchModel with trained DPO adapter
        """
        return self._pytorch_model

    def get_ref_model(self) -> Optional[ModelInterface]:
        """
        Get the reference model.

        Returns:
            PyTorchModel reference model, or None if not provided
        """
        return self._pytorch_ref_model
