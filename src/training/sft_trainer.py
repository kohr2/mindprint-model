"""
SFT (Supervised Fine-Tuning) Trainer for Bob Loukas Mindprint.

Implements LoRA-based supervised fine-tuning on Q&A pairs.
Optimized for Mac Studio M2 Ultra (MPS backend, fp16).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any
import time
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from .mps_utils import mps_empty_cache

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Configuration for SFT training."""

    # Training hyperparameters
    learning_rate: float = 3e-4
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048

    # LoRA configuration
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "o_proj"]
    )

    # Device settings
    device: str = "mps"
    dtype: str = "float16"

    # Output settings
    output_dir: Optional[str] = None  # If None, don't save adapters
    save_adapters: bool = True  # Auto-save after training

    def __post_init__(self):
        """Validate configuration."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.per_device_batch_size <= 0:
            raise ValueError("per_device_batch_size must be positive")


@dataclass
class SFTResult:
    """Result of SFT training."""

    success: bool
    adapter_path: str
    final_loss: float
    training_time_seconds: float
    samples_trained: int
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "adapter_path": self.adapter_path,
            "final_loss": self.final_loss,
            "training_time_seconds": self.training_time_seconds,
            "samples_trained": self.samples_trained,
            "error_message": self.error_message,
        }


class SFTDataset(Dataset):
    """Dataset for SFT training on Q&A pairs."""

    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
    ):
        """
        Initialize the dataset.

        Args:
            data: List of dicts with 'question' and 'answer' keys
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item."""
        item = self.data[idx]

        # Format as instruction-following prompt (Gemma-3 format)
        prompt = self._format_prompt(item["question"], item["answer"])

        # Tokenize
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": encoded["input_ids"].squeeze(0),
        }

    def _format_prompt(self, question: str, answer: str) -> str:
        """Format as Gemma-3 instruction prompt."""
        return f"""<start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
{answer}<end_of_turn>"""


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer with LoRA.

    Features:
    - LoRA adapter training (rank-8 default)
    - Topic-level training with isolated adapters
    - MPS-optimized for Mac Studio M2 Ultra
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: SFTConfig,
    ):
        """
        Initialize the trainer.

        Args:
            model: Base model to fine-tune
            tokenizer: Tokenizer for the model
            config: Training configuration
        """
        self.base_model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model: Optional[Any] = None  # PEFT model after preparation

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _prepare_model(self) -> None:
        """Prepare model with LoRA adapter."""
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

        self.model = get_peft_model(self.base_model, lora_config)
        logger.info(
            f"Created LoRA adapter: r={self.config.lora_r}, "
            f"alpha={self.config.lora_alpha}"
        )

    def train(self, train_data: List[Dict]) -> SFTResult:
        """
        Train on Q&A data.

        Args:
            train_data: List of dicts with 'question' and 'answer' keys

        Returns:
            SFTResult with training outcome
        """
        start_time = time.time()

        try:
            if not train_data:
                return SFTResult(
                    success=False,
                    adapter_path="",
                    final_loss=0.0,
                    training_time_seconds=0.0,
                    samples_trained=0,
                    error_message="No training data provided",
                )

            # Prepare model with LoRA
            self._prepare_model()

            # Create dataset and dataloader
            dataset = SFTDataset(
                train_data, self.tokenizer, self.config.max_seq_length
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.per_device_batch_size,
                shuffle=True,
            )

            # Setup optimizer
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
            )

            # Training loop
            self.model.train()
            total_loss = 0.0
            num_steps = 0
            max_grad_norm = 1.0  # Gradient clipping for stability

            for epoch in range(self.config.num_epochs):
                epoch_loss = 0.0
                valid_batches = 0
                for batch in dataloader:
                    # Move to device
                    batch = {k: v.to(self.model.device) for k, v in batch.items()}

                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss

                    # Skip if loss is NaN (MPS fp16 stability issue)
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"NaN/Inf loss detected, skipping batch")
                        optimizer.zero_grad()
                        continue

                    # Backward pass
                    loss.backward()

                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )

                    # Gradient accumulation
                    if (num_steps + 1) % self.config.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    epoch_loss += loss.item()
                    total_loss += loss.item()
                    num_steps += 1
                    valid_batches += 1

                avg_epoch_loss = epoch_loss / valid_batches if valid_batches > 0 else float('nan')
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.num_epochs}, "
                    f"Loss: {avg_epoch_loss:.4f}"
                )

            final_loss = total_loss / num_steps if num_steps > 0 else 0.0
            training_time = time.time() - start_time

            return SFTResult(
                success=True,
                adapter_path="",
                final_loss=final_loss,
                training_time_seconds=training_time,
                samples_trained=len(train_data) * self.config.num_epochs,
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

        finally:
            mps_empty_cache()

    def train_on_topic(
        self, topic_data: List[Dict], topic_id: str
    ) -> SFTResult:
        """
        Train on a single topic's Q&A data.

        Args:
            topic_data: Q&A pairs for this topic
            topic_id: Unique identifier for the topic

        Returns:
            SFTResult with training outcome
        """
        logger.info(f"Training on topic: {topic_id}")
        result = self.train(topic_data)

        if result.success:
            logger.info(
                f"Topic {topic_id} complete: loss={result.final_loss:.4f}, "
                f"time={result.training_time_seconds:.1f}s"
            )

            # Auto-save adapter if configured
            if self.config.save_adapters and self.config.output_dir:
                try:
                    from .adapter_utils import get_adapter_paths, parse_topic_id

                    # Parse topic_id to get components
                    unit_id, chapter_id, topic_name = parse_topic_id(topic_id)

                    # Get adapter path
                    sft_path, _, _ = get_adapter_paths(
                        self.config.output_dir,
                        unit_id,
                        chapter_id,
                        topic_name
                    )

                    # Save adapter
                    saved_path = self.save_adapter(sft_path)
                    result.adapter_path = str(saved_path)
                    logger.info(f"Auto-saved SFT adapter to {saved_path}")

                except Exception as e:
                    logger.warning(f"Failed to auto-save SFT adapter: {e}")

        return result

    def save_adapter(self, path: Path) -> Path:
        """
        Save the LoRA adapter.

        Args:
            path: Directory to save adapter

        Returns:
            Path to saved adapter
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        logger.info(f"Saved adapter to {path}")
        return path

    def get_model(self) -> PreTrainedModel:
        """
        Get the trained model with LoRA adapter.

        Returns:
            PEFT model with trained adapter
        """
        if self.model is None:
            raise ValueError("No trained model. Train first.")
        return self.model
