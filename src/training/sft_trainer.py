"""
SFTTrainer - Supervised Fine-Tuning for Bob Loukas mindprint.

Trains on Q&A pairs from the textbook before DPO refinement.
Optimized for Mac Studio M2 Ultra (MPS backend, fp16).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
import time

import torch
from torch.utils.data import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

from .mps_utils import get_mps_device, mps_empty_cache

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Configuration for SFT training."""

    # Training hyperparameters
    learning_rate: float = 3e-4
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 2048

    # LoRA configuration
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "o_proj"]
    )

    # MPS-specific
    use_mps: bool = True
    gradient_checkpointing: bool = True
    fp16: bool = True
    bf16: bool = False  # MPS prefers fp16 over bf16

    # Output
    output_dir: str = "./sft_output"
    save_steps: int = 100
    logging_steps: int = 10

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.num_epochs > 0, "Number of epochs must be positive"
        assert self.per_device_batch_size > 0, "Batch size must be positive"
        assert self.lora_r > 0, "LoRA rank must be positive"


@dataclass
class SFTResult:
    """Result from SFT training."""

    success: bool
    adapter_path: str
    final_loss: float
    training_time_seconds: float
    samples_trained: int
    error_message: Optional[str] = None


class SFTDataset(Dataset):
    """Dataset for SFT training from Q&A pairs."""

    GEMMA_USER = "<start_of_turn>user\n"
    GEMMA_MODEL = "<start_of_turn>model\n"
    GEMMA_END = "<end_of_turn>\n"

    CHATML_SYSTEM = "<|im_start|>system\n"
    CHATML_USER = "<|im_start|>user\n"
    CHATML_ASSISTANT = "<|im_start|>assistant\n"
    CHATML_END = "<|im_end|>\n"

    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        prompt_format: str = "gemma",
    ):
        """
        Initialize SFT dataset.

        Args:
            data: List of dicts with "instruction", "input", "output" keys
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length
            prompt_format: "gemma" or "chatml"
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_format = prompt_format

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")

        # Format the prompt
        formatted = self._format_prompt(instruction, input_text, output)

        # Tokenize
        encoding = self.tokenizer(
            formatted,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Squeeze to remove batch dimension
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Labels are same as input_ids for causal LM
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _format_prompt(self, instruction: str, input_text: str, output: str) -> str:
        """Format as model-specific prompt."""
        if self.prompt_format == "gemma":
            return self._format_gemma(instruction, input_text, output)
        elif self.prompt_format == "chatml":
            return self._format_chatml(instruction, input_text, output)
        else:
            # Simple format
            if input_text:
                return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    def _format_gemma(self, instruction: str, input_text: str, output: str) -> str:
        """Format for Gemma models."""
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction

        return (
            f"{self.GEMMA_USER}{user_content}{self.GEMMA_END}"
            f"{self.GEMMA_MODEL}{output}{self.GEMMA_END}"
        )

    def _format_chatml(self, instruction: str, input_text: str, output: str) -> str:
        """Format for ChatML-style models."""
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction

        return (
            f"{self.CHATML_USER}{user_content}{self.CHATML_END}"
            f"{self.CHATML_ASSISTANT}{output}{self.CHATML_END}"
        )


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer for mindprint.

    Features:
    - LoRA fine-tuning (not full model)
    - MPS-optimized training loop
    - fp16 mixed precision
    - Gradient checkpointing for memory efficiency
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: SFTConfig,
    ):
        """
        Initialize SFT trainer.

        Args:
            model: Base model (will be wrapped with LoRA)
            tokenizer: Tokenizer for the model
            config: SFT configuration
        """
        self.base_model = model
        self.tokenizer = tokenizer
        self.config = config
        self.peft_model: Optional[PreTrainedModel] = None

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_model(self) -> PreTrainedModel:
        """
        Prepare model for training with LoRA.

        Returns:
            Model with LoRA adapters attached
        """
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
        )

        self.peft_model = get_peft_model(self.base_model, lora_config)

        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            self.peft_model.enable_input_require_grads()

        logger.info(
            f"Prepared LoRA model with r={self.config.lora_r}, "
            f"alpha={self.config.lora_alpha}"
        )

        return self.peft_model

    def train(
        self,
        train_data: List[Dict],
        eval_data: Optional[List[Dict]] = None,
    ) -> SFTResult:
        """
        Train on SFT data.

        Args:
            train_data: Training Q&A pairs with instruction/input/output
            eval_data: Optional evaluation Q&A pairs

        Returns:
            SFTResult with training outcome
        """
        start_time = time.time()

        try:
            # Prepare model if not already done
            if self.peft_model is None:
                self.prepare_model()

            # Create datasets
            train_dataset = SFTDataset(
                train_data,
                self.tokenizer,
                max_length=self.config.max_seq_length,
            )

            eval_dataset = None
            if eval_data:
                eval_dataset = SFTDataset(
                    eval_data,
                    self.tokenizer,
                    max_length=self.config.max_seq_length,
                )

            # Create training arguments
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.per_device_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                warmup_ratio=self.config.warmup_ratio,
                weight_decay=self.config.weight_decay,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                fp16=self.config.fp16,
                bf16=self.config.bf16,
                gradient_checkpointing=self.config.gradient_checkpointing,
                remove_unused_columns=False,
                report_to="none",  # Disable wandb etc for now
            )

            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # Causal LM, not masked LM
            )

            # Create trainer
            trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )

            # Train
            train_result = trainer.train()

            # Clear cache after training
            mps_empty_cache()

            training_time = time.time() - start_time

            return SFTResult(
                success=True,
                adapter_path=self.config.output_dir,
                final_loss=train_result.training_loss,
                training_time_seconds=training_time,
                samples_trained=len(train_data),
            )

        except Exception as e:
            logger.error(f"SFT training failed: {e}")
            return SFTResult(
                success=False,
                adapter_path="",
                final_loss=0.0,
                training_time_seconds=time.time() - start_time,
                samples_trained=0,
                error_message=str(e),
            )

    def train_on_topic(
        self,
        topic_data: Dict,
        topic_identifier: str,
    ) -> SFTResult:
        """
        Train on a single topic's data.

        Args:
            topic_data: Dict with topic questions and answers
            topic_identifier: e.g., "unit-01/chapter-01/topic-01"

        Returns:
            SFTResult for this topic
        """
        # Convert topic data to SFT format
        sft_data = []
        questions = topic_data.get("questions", [])

        for q in questions:
            sft_data.append(
                {
                    "instruction": q.get("question", ""),
                    "input": "",
                    "output": q.get("reference_answer", ""),
                }
            )

        logger.info(f"Training on topic {topic_identifier} with {len(sft_data)} examples")

        return self.train(sft_data)

    def save_adapter(self, path: str) -> Path:
        """
        Save LoRA adapter to disk.

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
        logger.info(f"Saved adapter to {output_path}")

        return output_path

    def get_model(self) -> PreTrainedModel:
        """Get the current model (with LoRA)."""
        if self.peft_model is None:
            raise ValueError("Model not prepared. Call prepare_model() first.")
        return self.peft_model
