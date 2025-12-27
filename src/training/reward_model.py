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
from typing import List, Dict, Optional, Any
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
    num_epochs: int = 1
    per_device_batch_size: int = 4
    max_length: int = 1024

    # Voice penalty settings
    use_voice_penalty: bool = False
    voice_penalty_weight: float = 0.1

    # Device settings
    device: str = "mps"

    def __post_init__(self):
        """Validate configuration."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
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
        hidden_size = base_model.config.hidden_size

        # Score head: hidden_size -> 1
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        )

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

    def train(self, preference_data: List[Dict]) -> RewardResult:
        """
        Train the reward model on preference data.

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

            # Create reward model
            self._reward_model = RewardModel(self.base_model)
            self._reward_model.set_tokenizer(self.tokenizer)

            # Create dataset and dataloader
            dataset = PreferenceDataset(
                preference_data, self.tokenizer, self.config.max_length
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.per_device_batch_size,
                shuffle=True,
            )

            # Setup optimizer
            optimizer = torch.optim.AdamW(
                self._reward_model.parameters(),
                lr=self.config.learning_rate,
            )

            # Training loop
            self._reward_model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for epoch in range(self.config.num_epochs):
                epoch_loss = 0.0
                epoch_correct = 0

                for batch in dataloader:
                    # Move to device
                    device = self.base_model.device
                    batch = {k: v.to(device) for k, v in batch.items()}

                    # Forward pass for chosen
                    chosen_scores = self._reward_model(
                        batch["chosen_input_ids"],
                        batch["chosen_attention_mask"],
                    )

                    # Forward pass for rejected
                    rejected_scores = self._reward_model(
                        batch["rejected_input_ids"],
                        batch["rejected_attention_mask"],
                    )

                    # Compute loss
                    loss = self._compute_loss(chosen_scores, rejected_scores)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Track metrics
                    epoch_loss += loss.item()
                    total_loss += loss.item()

                    # Accuracy: how often chosen > rejected
                    correct = (chosen_scores > rejected_scores).sum().item()
                    epoch_correct += correct
                    total_correct += correct
                    total_samples += len(chosen_scores)

                logger.info(
                    f"Epoch {epoch + 1}/{self.config.num_epochs}, "
                    f"Loss: {epoch_loss / len(dataloader):.4f}, "
                    f"Accuracy: {epoch_correct / len(dataset):.2%}"
                )

            final_loss = total_loss / (len(dataloader) * self.config.num_epochs)
            final_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
            training_time = time.time() - start_time

            return RewardResult(
                success=True,
                model_path="",
                final_loss=final_loss,
                final_accuracy=final_accuracy,
                training_time_seconds=training_time,
            )

        except Exception as e:
            logger.error(f"Reward model training failed: {e}")
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
