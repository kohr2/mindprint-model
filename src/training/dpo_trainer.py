"""
Rank1DPOTrainer - Direct Preference Optimization with Rank-1 LoRA.

Refines voice fidelity using preference pairs after SFT.
Optimized for Mac Studio M2 Ultra (MPS backend, fp16).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import time

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

from .mps_utils import get_mps_device, mps_empty_cache

logger = logging.getLogger(__name__)


@dataclass
class Rank1DPOConfig:
    """Configuration for Rank-1 DPO training."""

    # DPO hyperparameters
    beta: float = 0.1  # KL penalty coefficient
    learning_rate: float = 5e-7  # Much lower than SFT
    max_steps: int = 100  # Steps per topic
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_prompt_length: int = 512
    max_length: int = 1024

    # Rank-1 LoRA configuration
    lora_r: int = 1
    lora_alpha: float = 1.0
    lora_dropout: float = 0.0  # No dropout for Rank-1
    target_modules: List[str] = field(
        default_factory=lambda: ["o_proj", "v_proj", "up_proj", "down_proj"]
    )

    # MPS-specific
    use_mps: bool = True
    fp16: bool = False  # MPS doesn't support fp16 training flag
    bf16: bool = False

    # Reference model handling
    ref_model_strategy: str = "shared"  # "shared", "copy", or "none"

    # Output
    output_dir: str = "./dpo_output"
    logging_steps: int = 10

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert self.beta > 0, "Beta must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.max_steps > 0, "Max steps must be positive"
        assert self.lora_r >= 1, "LoRA rank must be at least 1"


@dataclass
class DPOResult:
    """Result from DPO training."""

    success: bool
    adapter_path: str
    final_loss: float
    steps_completed: int
    training_time_seconds: float
    chosen_rewards_mean: float
    rejected_rewards_mean: float
    reward_margin: float
    error_message: Optional[str] = None


class Rank1DPOTrainer:
    """
    Rank-1 LoRA trainer with DPO.

    Features:
    - Minimal rank (r=1) for efficient preference refinement
    - Targets voice/style layers (o_proj, v_proj, up_proj, down_proj)
    - Uses TRL's DPOTrainer under the hood
    - MPS-optimized
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Rank1DPOConfig,
        ref_model: Optional[PreTrainedModel] = None,
    ):
        """
        Initialize DPO trainer.

        Args:
            model: Policy model (may already have SFT LoRA)
            tokenizer: Tokenizer for the model
            config: DPO configuration
            ref_model: Reference model (if None, uses model before LoRA)
        """
        self.base_model = model
        self.tokenizer = tokenizer
        self.config = config
        self.ref_model = ref_model
        self.peft_model: Optional[PreTrainedModel] = None

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_model(self) -> Tuple[PreTrainedModel, Optional[PreTrainedModel]]:
        """
        Prepare policy and reference models.

        For MPS:
        - Both models stay in fp16
        - Reference model is frozen
        - Policy model gets Rank-1 LoRA

        Returns:
            Tuple of (policy_model, reference_model)
        """
        # Check if model already has adapters (reuse SFT adapter for DPO)
        from peft import PeftModel
        if isinstance(self.base_model, PeftModel):
            logger.info(
                "Model already has SFT adapter. "
                "Continuing to train the same adapter with DPO objective."
            )
            # Use existing adapter instead of creating new one
            self.peft_model = self.base_model
        else:
            # Create new adapter if none exists
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                task_type=TaskType.CAUSAL_LM,
            )
            self.peft_model = get_peft_model(self.base_model, lora_config)

        logger.info(
            f"Prepared Rank-1 DPO model with r={self.config.lora_r}, "
            f"alpha={self.config.lora_alpha}"
        )

        return self.peft_model, self.ref_model

    def train(
        self,
        preference_pairs: List[Dict],
    ) -> DPOResult:
        """
        Train with DPO on preference pairs.

        Args:
            preference_pairs: List of dicts with "prompt", "chosen", "rejected"

        Returns:
            DPOResult with training outcome
        """
        start_time = time.time()

        try:
            # Prepare model if not already done
            if self.peft_model is None:
                self.prepare_model()

            # Create dataset
            train_dataset = self._create_dataset(preference_pairs)

            # Create DPO config
            dpo_config = DPOConfig(
                beta=self.config.beta,
                learning_rate=self.config.learning_rate,
                max_steps=self.config.max_steps,
                per_device_train_batch_size=self.config.per_device_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                max_prompt_length=self.config.max_prompt_length,
                max_length=self.config.max_length,
                output_dir=self.config.output_dir,
                logging_steps=self.config.logging_steps,
                fp16=self.config.fp16,
                bf16=self.config.bf16,
                remove_unused_columns=False,
                report_to="none",
            )

            # Create DPO trainer
            trainer = DPOTrainer(
                model=self.peft_model,
                ref_model=self.ref_model,
                args=dpo_config,
                train_dataset=train_dataset,
                processing_class=self.tokenizer,
            )

            # Train
            train_result = trainer.train()

            # Get metrics
            metrics = train_result.metrics if hasattr(train_result, "metrics") else {}
            chosen_rewards = metrics.get("train_rewards/chosen", 0.0)
            rejected_rewards = metrics.get("train_rewards/rejected", 0.0)

            # Clear cache after training
            mps_empty_cache()

            training_time = time.time() - start_time

            return DPOResult(
                success=True,
                adapter_path=self.config.output_dir,
                final_loss=getattr(train_result, "training_loss", 0.0),
                steps_completed=self.config.max_steps,
                training_time_seconds=training_time,
                chosen_rewards_mean=chosen_rewards,
                rejected_rewards_mean=rejected_rewards,
                reward_margin=chosen_rewards - rejected_rewards,
            )

        except Exception as e:
            logger.error(f"DPO training failed: {e}")
            return DPOResult(
                success=False,
                adapter_path="",
                final_loss=0.0,
                steps_completed=0,
                training_time_seconds=time.time() - start_time,
                chosen_rewards_mean=0.0,
                rejected_rewards_mean=0.0,
                reward_margin=0.0,
                error_message=str(e),
            )

    def train_on_topic(
        self,
        topic_preference_pairs: List[Dict],
        topic_identifier: str,
    ) -> DPOResult:
        """
        Train on a single topic's preference pairs.

        Args:
            topic_preference_pairs: Preference pairs for this topic
            topic_identifier: e.g., "unit-01/chapter-01/topic-01"

        Returns:
            DPOResult for this topic
        """
        logger.info(
            f"DPO training on topic {topic_identifier} "
            f"with {len(topic_preference_pairs)} pairs"
        )

        return self.train(topic_preference_pairs)

    def _create_dataset(self, pairs: List[Dict]) -> Dataset:
        """
        Create HuggingFace dataset from preference pairs.

        Args:
            pairs: List of preference dicts with prompt, chosen, rejected

        Returns:
            Dataset ready for DPOTrainer
        """
        return Dataset.from_list(pairs)

    def save_adapter(self, path: str) -> Path:
        """
        Save the DPO LoRA adapter.

        Args:
            path: Output path for adapter

        Returns:
            Path to saved adapter
        """
        if self.peft_model is None:
            raise ValueError("Model not prepared. Call prepare_model() first.")

        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.peft_model.save_pretrained(output_path)
        logger.info(f"Saved DPO adapter to {output_path}")

        return output_path

    def merge_adapter(self) -> PreTrainedModel:
        """
        Merge LoRA adapter into base model.

        Returns:
            Merged model
        """
        if self.peft_model is None:
            raise ValueError("Model not prepared. Call prepare_model() first.")

        merged_model = self.peft_model.merge_and_unload()
        logger.info("Merged DPO adapter into base model")

        return merged_model

    def get_model(self) -> PreTrainedModel:
        """Get the current policy model."""
        if self.peft_model is None:
            raise ValueError("Model not prepared. Call prepare_model() first.")
        return self.peft_model
