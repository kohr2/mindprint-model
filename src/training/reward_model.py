"""
Reward Model for PPO Training.

Learns to score responses based on:
- Voice fidelity to Bob Loukas's style
- Factual accuracy
- Correct halving/cycle distinction

The reward model is trained on preference pairs and used by PPO
to provide training signal.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import time
import logging

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from .mps_utils import mps_empty_cache

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for reward model training."""

    # Training hyperparameters
    learning_rate: float = 1e-5
    num_epochs: int = 3  # Increased from 1
    per_device_batch_size: int = 8  # Increased from 4
    max_length: int = 1024

    # Validation settings
    validation_split: float = 0.2  # 80/20 train/val split
    early_stopping_patience: int = 3  # Stop if no improvement for N epochs
    early_stopping_delta: float = 0.01  # Minimum improvement threshold

    # Learning rate scheduling
    use_lr_scheduler: bool = True
    scheduler_type: str = "cosine"  # cosine, linear, constant
    warmup_ratio: float = 0.1

    # Voice penalty settings
    use_voice_penalty: bool = False
    voice_penalty_weight: float = 0.1

    # Device settings
    device: str = "mps"

    def __post_init__(self):
        """Validate configuration."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.validation_split < 1:
            raise ValueError("validation_split must be in [0, 1)")
        if self.scheduler_type not in ["cosine", "linear", "constant"]:
            raise ValueError("scheduler_type must be cosine, linear, or constant")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")


@dataclass
class RewardResult:
    """Result of reward model training."""

    success: bool
    model_path: str
    final_loss: float
    final_accuracy: float
    training_time_seconds: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "model_path": self.model_path,
            "final_loss": self.final_loss,
            "final_accuracy": self.final_accuracy,
            "training_time_seconds": self.training_time_seconds,
            "error_message": self.error_message,
        }


class RewardModel(nn.Module):
    """
    Reward model that scores responses.

    Architecture:
    - Base transformer model for encoding
    - Score head that produces scalar reward
    """

    def __init__(self, base_model: PreTrainedModel):
        """
        Initialize the reward model.

        Args:
            base_model: Pretrained transformer model
        """
        super().__init__()
        self.base_model = base_model
        # Support both standard hidden_size and Gemma3 text_config.hidden_size
        if hasattr(base_model.config, "hidden_size"):
            hidden_size = base_model.config.hidden_size
        elif hasattr(base_model.config, "text_config"):
            hidden_size = base_model.config.text_config.hidden_size
        else:
            raise ValueError("Cannot determine hidden_size from model config")

        # Score head: hidden_size -> 1
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        )

        # Move score head to same device and dtype as base model
        self.score_head.to(base_model.device)
        if hasattr(base_model, 'dtype'):
            self.score_head.to(dtype=base_model.dtype)
        elif base_model.parameters().__next__() is not None:
            # Infer dtype from first parameter
            first_param = next(base_model.parameters())
            self.score_head.to(dtype=first_param.dtype)

        self._tokenizer: Optional[PreTrainedTokenizer] = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass to compute reward.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Reward scores [batch_size, 1]
        """
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use last hidden state at the last token position
        hidden_states = outputs.hidden_states[-1]

        # Find last non-padded position for each sample
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1

        # Gather last token hidden states
        last_hidden = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            sequence_lengths,
        ]

        # Ensure dtype consistency for MPS compatibility
        # Cast to score_head's expected dtype
        score_head_dtype = next(self.score_head.parameters()).dtype
        if last_hidden.dtype != score_head_dtype:
            last_hidden = last_hidden.to(dtype=score_head_dtype)

        # Compute reward score
        rewards = self.score_head(last_hidden)
        return rewards

    def get_reward(
        self,
        prompt: str,
        response: str,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> float:
        """
        Get reward for a prompt-response pair.

        Args:
            prompt: Input prompt
            response: Model response
            tokenizer: Tokenizer (uses stored if None)

        Returns:
            Scalar reward value
        """
        if tokenizer is None:
            tokenizer = self._tokenizer
        if tokenizer is None:
            raise ValueError("No tokenizer provided")

        # Format and tokenize
        text = f"{prompt}\n{response}"
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        # Move to device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get reward
        with torch.no_grad():
            reward = self.forward(
                inputs["input_ids"],
                inputs["attention_mask"],
            )

        return reward.item()

    def set_tokenizer(self, tokenizer: PreTrainedTokenizer) -> None:
        """Set the tokenizer for inference."""
        self._tokenizer = tokenizer


class PreferenceDataset(Dataset):
    """Dataset for preference pair training."""

    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
    ):
        """
        Initialize the dataset.

        Args:
            data: List of dicts with 'prompt', 'chosen', 'rejected'
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
        """Get a single preference pair."""
        item = self.data[idx]

        # Tokenize chosen
        chosen_text = f"{item['prompt']}\n{item['chosen']}"
        chosen = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize rejected
        rejected_text = f"{item['prompt']}\n{item['rejected']}"
        rejected = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected["attention_mask"].squeeze(0),
        }


class RewardModelTrainer:
    """
    Trainer for the reward model.

    Uses pairwise ranking loss to train the model to prefer
    chosen responses over rejected ones.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: RewardConfig,
    ):
        """
        Initialize the trainer.

        Args:
            model: Base model for reward model
            tokenizer: Tokenizer for the model
            config: Training configuration
        """
        self.base_model = model
        self.tokenizer = tokenizer
        self.config = config
        self._reward_model: Optional[RewardModel] = None

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _compute_loss(
        self,
        chosen_scores: torch.Tensor,
        rejected_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise ranking loss.

        Loss = -log(sigmoid(chosen - rejected))

        Args:
            chosen_scores: Scores for chosen responses
            rejected_scores: Scores for rejected responses

        Returns:
            Scalar loss value
        """
        # Pairwise ranking loss (Bradley-Terry)
        loss = -torch.log(torch.sigmoid(chosen_scores - rejected_scores)).mean()
        return loss

    def _split_data(
        self, data: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Split data into train and validation sets.

        Args:
            data: Full preference pair dataset

        Returns:
            Tuple of (train_data, val_data)
        """
        if self.config.validation_split <= 0:
            return data, []

        import random
        random.seed(42)  # Reproducibility
        shuffled = data.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1 - self.config.validation_split))
        train_data = shuffled[:split_idx]
        val_data = shuffled[split_idx:]

        logger.info(
            f"Split data: {len(train_data)} train, {len(val_data)} validation"
        )

        return train_data, val_data

    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int
    ) -> Optional[Any]:
        """Create learning rate scheduler."""
        if not self.config.use_lr_scheduler:
            return None

        warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        if self.config.scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps - warmup_steps
            )
        elif self.config.scheduler_type == "linear":
            from torch.optim.lr_scheduler import LinearLR
            return LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=num_training_steps
            )
        else:
            return None

    def _evaluate_on_validation(
        self,
        val_dataloader: DataLoader
    ) -> Tuple[float, float]:
        """
        Evaluate on validation set.

        Returns:
            Tuple of (val_loss, val_accuracy)
        """
        self._reward_model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_dataloader:
                device = self.base_model.device
                batch = {k: v.to(device) for k, v in batch.items()}

                chosen_scores = self._reward_model(
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                )

                rejected_scores = self._reward_model(
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                )

                loss = self._compute_loss(chosen_scores, rejected_scores)
                total_loss += loss.item()

                correct = (chosen_scores > rejected_scores).sum().item()
                total_correct += correct
                total_samples += len(chosen_scores)

        self._reward_model.train()

        avg_loss = total_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return avg_loss, accuracy

    def train(self, preference_data: List[Dict]) -> RewardResult:
        """
        Train the reward model on preference data with validation and early stopping.

        Now includes:
        - Train/validation split
        - Learning rate scheduling
        - Early stopping
        - Per-epoch and per-batch logging

        Args:
            preference_data: List of preference pairs

        Returns:
            RewardResult with training outcome
        """
        start_time = time.time()

        try:
            if not preference_data:
                return RewardResult(
                    success=False,
                    model_path="",
                    final_loss=0.0,
                    final_accuracy=0.0,
                    training_time_seconds=0.0,
                    error_message="No training data provided",
                )

            # Split data into train/val
            train_data, val_data = self._split_data(preference_data)

            # Create reward model
            self._reward_model = RewardModel(self.base_model)
            self._reward_model.set_tokenizer(self.tokenizer)

            # Create dataloaders
            train_dataset = PreferenceDataset(
                train_data, self.tokenizer, self.config.max_length
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.config.per_device_batch_size,
                shuffle=True,
            )

            val_dataloader = None
            if val_data:
                val_dataset = PreferenceDataset(
                    val_data, self.tokenizer, self.config.max_length
                )
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=self.config.per_device_batch_size,
                    shuffle=False,
                )

            # Setup optimizer
            optimizer = torch.optim.AdamW(
                self._reward_model.parameters(),
                lr=self.config.learning_rate,
            )

            # Setup scheduler
            num_training_steps = len(train_dataloader) * self.config.num_epochs
            scheduler = self._create_scheduler(optimizer, num_training_steps)

            # Training loop with early stopping
            self._reward_model.train()
            best_val_accuracy = 0.0
            epochs_without_improvement = 0

            epoch_metrics = []  # Track all epoch metrics

            for epoch in range(self.config.num_epochs):
                epoch_start = time.time()
                epoch_loss = 0.0
                epoch_correct = 0
                batch_count = 0

                for batch_idx, batch in enumerate(train_dataloader):
                    # Move to device
                    device = self.base_model.device
                    batch = {k: v.to(device) for k, v in batch.items()}

                    # Forward pass
                    chosen_scores = self._reward_model(
                        batch["chosen_input_ids"],
                        batch["chosen_attention_mask"],
                    )

                    rejected_scores = self._reward_model(
                        batch["rejected_input_ids"],
                        batch["rejected_attention_mask"],
                    )

                    # Compute loss
                    loss = self._compute_loss(chosen_scores, rejected_scores)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()

                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(
                        self._reward_model.parameters(),
                        max_norm=1.0
                    )

                    optimizer.step()

                    if scheduler:
                        scheduler.step()

                    # Track metrics
                    epoch_loss += loss.item()
                    correct = (chosen_scores > rejected_scores).sum().item()
                    epoch_correct += correct
                    batch_count += 1

                    # Log per-batch progress every 5 batches
                    if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_dataloader):
                        batch_acc = correct / len(chosen_scores)
                        logger.info(
                            f"Epoch {epoch+1}/{self.config.num_epochs}, "
                            f"Batch {batch_idx+1}/{len(train_dataloader)}: "
                            f"loss={loss.item():.4f}, acc={batch_acc:.2%}"
                        )

                # Epoch summary
                train_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
                train_acc = epoch_correct / len(train_dataset)
                epoch_time = time.time() - epoch_start

                # Validation
                val_loss = 0.0
                val_acc = 0.0
                if val_dataloader:
                    val_loss, val_acc = self._evaluate_on_validation(val_dataloader)

                logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs} complete "
                    f"({epoch_time:.1f}s):\n"
                    f"  Train: loss={train_loss:.4f}, acc={train_acc:.2%}\n"
                    f"  Val:   loss={val_loss:.4f}, acc={val_acc:.2%}"
                )

                epoch_metrics.append({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "time_seconds": epoch_time,
                })

                # Early stopping check
                if val_dataloader:
                    if val_acc > best_val_accuracy + self.config.early_stopping_delta:
                        best_val_accuracy = val_acc
                        epochs_without_improvement = 0
                        logger.info(f"New best validation accuracy: {best_val_accuracy:.2%}")
                    else:
                        epochs_without_improvement += 1
                        if epochs_without_improvement >= self.config.early_stopping_patience:
                            logger.info(
                                f"Early stopping after {epoch+1} epochs "
                                f"(no improvement for {epochs_without_improvement} epochs)"
                            )
                            break

            # Final metrics
            final_metrics = epoch_metrics[-1] if epoch_metrics else {}
            final_loss = final_metrics.get("train_loss", 0.0)
            final_accuracy = final_metrics.get("val_acc", final_metrics.get("train_acc", 0.0))
            training_time = time.time() - start_time

            logger.info(f"Reward model training complete:")
            logger.info(f"  Final loss: {final_loss:.4f}")
            logger.info(f"  Final accuracy: {final_accuracy:.2%}")
            logger.info(f"  Training time: {training_time:.1f}s")

            return RewardResult(
                success=True,
                model_path="",
                final_loss=final_loss,
                final_accuracy=final_accuracy,
                training_time_seconds=training_time,
            )

        except Exception as e:
            logger.error(f"Reward model training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return RewardResult(
                success=False,
                model_path="",
                final_loss=0.0,
                final_accuracy=0.0,
                training_time_seconds=time.time() - start_time,
                error_message=str(e),
            )

        finally:
            mps_empty_cache()

    def save_model(self, path: Path) -> Path:
        """
        Save the reward model.

        Args:
            path: Directory to save model

        Returns:
            Path to saved model
        """
        if self._reward_model is None:
            raise ValueError("No model to save. Train first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save reward model state dict
        torch.save(
            self._reward_model.state_dict(),
            path / "reward_model.pt",
        )

        logger.info(f"Saved reward model to {path}")
        return path

    @classmethod
    def load_model(
        cls,
        path: Path,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[RewardConfig] = None,
    ) -> "RewardModelTrainer":
        """
        Load a saved reward model.

        Args:
            path: Path to saved model
            base_model: Base model for reward model
            tokenizer: Tokenizer
            config: Configuration

        Returns:
            RewardModelTrainer with loaded model
        """
        config = config or RewardConfig()
        trainer = cls(base_model, tokenizer, config)

        # Create reward model and load weights
        trainer._reward_model = RewardModel(base_model)
        trainer._reward_model.load_state_dict(
            torch.load(path / "reward_model.pt")
        )
        trainer._reward_model.set_tokenizer(tokenizer)

        logger.info(f"Loaded reward model from {path}")
        return trainer

    def get_reward_model(self) -> RewardModel:
        """
        Get the trained reward model.

        Returns:
            Trained RewardModel
        """
        if self._reward_model is None:
            raise ValueError("No trained model. Train first.")
        return self._reward_model
