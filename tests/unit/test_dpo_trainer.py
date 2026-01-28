"""
Tests for Rank1DPOTrainer - Direct Preference Optimization.

Tests cover:
- Rank-1 LoRA configuration
- Preference pair dataset creation
- DPO training loop (mocked)
- Reward computation
- Topic-level training
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import shutil

import torch

from src.training.dpo_trainer import (
    Rank1DPOConfig,
    DPOResult,
    Rank1DPOTrainer,
)


class TestRank1DPOConfig:
    """Test Rank1DPOConfig dataclass."""

    def test_default_beta(self) -> None:
        """Default beta is 0.1."""
        config = Rank1DPOConfig()
        assert config.beta == 0.1

    def test_default_learning_rate(self) -> None:
        """Default learning rate is 5e-7."""
        config = Rank1DPOConfig()
        assert config.learning_rate == 5e-7

    def test_default_max_steps(self) -> None:
        """Default max steps is 100."""
        config = Rank1DPOConfig()
        assert config.max_steps == 100

    def test_default_batch_size(self) -> None:
        """Default batch size is 2."""
        config = Rank1DPOConfig()
        assert config.per_device_batch_size == 2

    def test_default_lora_rank_is_one(self) -> None:
        """Default LoRA rank is 1 (Rank-1 LoRA)."""
        config = Rank1DPOConfig()
        assert config.lora_r == 1

    def test_default_lora_alpha(self) -> None:
        """Default LoRA alpha is 1.0."""
        config = Rank1DPOConfig()
        assert config.lora_alpha == 1.0

    def test_target_modules_for_voice(self) -> None:
        """Default target modules target voice layers."""
        config = Rank1DPOConfig()
        assert "o_proj" in config.target_modules
        assert "v_proj" in config.target_modules
        assert "up_proj" in config.target_modules
        assert "down_proj" in config.target_modules

    def test_validates_positive_beta(self) -> None:
        """Raises error for non-positive beta."""
        with pytest.raises(AssertionError):
            Rank1DPOConfig(beta=0)

    def test_validates_positive_learning_rate(self) -> None:
        """Raises error for non-positive learning rate."""
        with pytest.raises(AssertionError):
            Rank1DPOConfig(learning_rate=-1e-7)

    def test_validates_positive_max_steps(self) -> None:
        """Raises error for non-positive max steps."""
        with pytest.raises(AssertionError):
            Rank1DPOConfig(max_steps=0)

    def test_validates_positive_lora_rank(self) -> None:
        """Raises error for non-positive LoRA rank."""
        with pytest.raises(AssertionError):
            Rank1DPOConfig(lora_r=0)

    def test_custom_config(self) -> None:
        """Custom configuration values are set correctly."""
        config = Rank1DPOConfig(
            beta=0.2,
            learning_rate=1e-6,
            max_steps=50,
            lora_r=2,
        )
        assert config.beta == 0.2
        assert config.learning_rate == 1e-6
        assert config.max_steps == 50
        assert config.lora_r == 2


class TestDPOResult:
    """Test DPOResult dataclass."""

    def test_stores_success_status(self) -> None:
        """Result stores success status."""
        result = DPOResult(
            success=True,
            adapter_path="/path/to/adapter",
            final_loss=0.5,
            steps_completed=100,
            training_time_seconds=600.0,
            chosen_rewards_mean=0.8,
            rejected_rewards_mean=0.3,
            reward_margin=0.5,
        )
        assert result.success is True

    def test_stores_failure_with_error(self) -> None:
        """Result stores failure with error message."""
        result = DPOResult(
            success=False,
            adapter_path="",
            final_loss=0.0,
            steps_completed=0,
            training_time_seconds=10.0,
            chosen_rewards_mean=0.0,
            rejected_rewards_mean=0.0,
            reward_margin=0.0,
            error_message="Training failed",
        )
        assert result.success is False
        assert result.error_message == "Training failed"

    def test_stores_reward_metrics(self) -> None:
        """Result stores reward metrics."""
        result = DPOResult(
            success=True,
            adapter_path="/path",
            final_loss=0.3,
            steps_completed=100,
            training_time_seconds=300.0,
            chosen_rewards_mean=0.9,
            rejected_rewards_mean=0.2,
            reward_margin=0.7,
        )
        assert result.chosen_rewards_mean == 0.9
        assert result.rejected_rewards_mean == 0.2
        assert result.reward_margin == 0.7


class TestRank1DPOTrainer:
    """Test Rank1DPOTrainer class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = MagicMock()
        model.config = MagicMock()
        model.config.model_type = "gemma"
        model.device = torch.device("cpu")
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "</s>"
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        tokenizer.model_max_length = 2048
        return tokenizer

    @pytest.fixture
    def sample_preference_pairs(self):
        """Sample preference pairs."""
        return [
            {
                "prompt": "What drives the 4-year cycle?",
                "chosen": "Market psychology drives the cycle, not the halving.",
                "rejected": "The halving causes the 4-year cycle.",
            },
            {
                "prompt": "What is accumulation?",
                "chosen": "Accumulation is when smart money quietly builds positions.",
                "rejected": "Accumulation is when prices go up.",
            },
        ]

    def test_initializes_with_config(self, mock_model, mock_tokenizer) -> None:
        """Trainer initializes with configuration."""
        config = Rank1DPOConfig()
        trainer = Rank1DPOTrainer(mock_model, mock_tokenizer, config)
        assert trainer.config == config

    def test_has_prepare_model_method(self, mock_model, mock_tokenizer) -> None:
        """Trainer has prepare_model method."""
        trainer = Rank1DPOTrainer(mock_model, mock_tokenizer, Rank1DPOConfig())
        assert hasattr(trainer, "prepare_model")
        assert callable(trainer.prepare_model)

    def test_has_train_method(self, mock_model, mock_tokenizer) -> None:
        """Trainer has train method."""
        trainer = Rank1DPOTrainer(mock_model, mock_tokenizer, Rank1DPOConfig())
        assert hasattr(trainer, "train")
        assert callable(trainer.train)

    def test_has_train_on_topic_method(self, mock_model, mock_tokenizer) -> None:
        """Trainer has train_on_topic method."""
        trainer = Rank1DPOTrainer(mock_model, mock_tokenizer, Rank1DPOConfig())
        assert hasattr(trainer, "train_on_topic")
        assert callable(trainer.train_on_topic)

    def test_has_save_adapter_method(self, mock_model, mock_tokenizer) -> None:
        """Trainer has save_adapter method."""
        trainer = Rank1DPOTrainer(mock_model, mock_tokenizer, Rank1DPOConfig())
        assert hasattr(trainer, "save_adapter")
        assert callable(trainer.save_adapter)

    def test_has_merge_adapter_method(self, mock_model, mock_tokenizer) -> None:
        """Trainer has merge_adapter method."""
        trainer = Rank1DPOTrainer(mock_model, mock_tokenizer, Rank1DPOConfig())
        assert hasattr(trainer, "merge_adapter")
        assert callable(trainer.merge_adapter)

    def test_has_get_model_method(self, mock_model, mock_tokenizer) -> None:
        """Trainer has get_model method."""
        trainer = Rank1DPOTrainer(mock_model, mock_tokenizer, Rank1DPOConfig())
        assert hasattr(trainer, "get_model")
        assert callable(trainer.get_model)

    @patch("src.training.dpo_trainer.get_peft_model")
    def test_prepare_model_creates_rank1_lora(
        self, mock_get_peft: MagicMock, mock_model, mock_tokenizer
    ) -> None:
        """prepare_model creates LoRA with rank=1."""
        mock_peft_model = MagicMock()
        mock_get_peft.return_value = mock_peft_model

        config = Rank1DPOConfig()
        trainer = Rank1DPOTrainer(mock_model, mock_tokenizer, config)
        result = trainer.prepare_model()

        # Verify LoRA config has r=1
        call_args = mock_get_peft.call_args
        lora_config = call_args[0][1]  # Second positional arg
        assert lora_config.r == 1

    @patch("src.training.dpo_trainer.DPOTrainer")
    @patch("src.training.dpo_trainer.get_peft_model")
    def test_train_returns_dpo_result(
        self,
        mock_get_peft: MagicMock,
        mock_dpo_trainer_cls: MagicMock,
        mock_model,
        mock_tokenizer,
        sample_preference_pairs,
    ) -> None:
        """train() returns DPOResult."""
        mock_peft_model = MagicMock()
        mock_get_peft.return_value = mock_peft_model

        mock_dpo_trainer = MagicMock()
        mock_dpo_trainer.train.return_value = MagicMock()
        mock_dpo_trainer_cls.return_value = mock_dpo_trainer

        trainer = Rank1DPOTrainer(mock_model, mock_tokenizer, Rank1DPOConfig())
        result = trainer.train(sample_preference_pairs)

        assert isinstance(result, DPOResult)

    @patch("src.training.dpo_trainer.DPOTrainer")
    @patch("src.training.dpo_trainer.get_peft_model")
    def test_train_on_topic_returns_dpo_result(
        self,
        mock_get_peft: MagicMock,
        mock_dpo_trainer_cls: MagicMock,
        mock_model,
        mock_tokenizer,
    ) -> None:
        """train_on_topic() returns DPOResult."""
        mock_peft_model = MagicMock()
        mock_get_peft.return_value = mock_peft_model

        mock_dpo_trainer = MagicMock()
        mock_dpo_trainer.train.return_value = MagicMock()
        mock_dpo_trainer_cls.return_value = mock_dpo_trainer

        trainer = Rank1DPOTrainer(mock_model, mock_tokenizer, Rank1DPOConfig())

        topic_pairs = [
            {
                "prompt": "Question?",
                "chosen": "Good answer.",
                "rejected": "Bad answer.",
            }
        ]
        result = trainer.train_on_topic(topic_pairs, "unit-01/chapter-01/topic-01")

        assert isinstance(result, DPOResult)

    @patch("src.training.dpo_trainer.get_peft_model")
    def test_save_adapter_creates_files(
        self,
        mock_get_peft: MagicMock,
        mock_model,
        mock_tokenizer,
        temp_dir,
    ) -> None:
        """save_adapter creates adapter files."""
        mock_peft_model = MagicMock()
        mock_get_peft.return_value = mock_peft_model

        trainer = Rank1DPOTrainer(mock_model, mock_tokenizer, Rank1DPOConfig())
        trainer.prepare_model()

        output_path = Path(temp_dir) / "adapter"
        result = trainer.save_adapter(str(output_path))

        mock_peft_model.save_pretrained.assert_called_once()
        assert isinstance(result, Path)

    @patch("src.training.dpo_trainer.get_peft_model")
    def test_merge_adapter_returns_model(
        self, mock_get_peft: MagicMock, mock_model, mock_tokenizer
    ) -> None:
        """merge_adapter returns merged model."""
        mock_peft_model = MagicMock()
        mock_merged_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_merged_model
        mock_get_peft.return_value = mock_peft_model

        trainer = Rank1DPOTrainer(mock_model, mock_tokenizer, Rank1DPOConfig())
        trainer.prepare_model()
        result = trainer.merge_adapter()

        mock_peft_model.merge_and_unload.assert_called_once()
        assert result == mock_merged_model

    @patch("src.training.dpo_trainer.get_peft_model")
    def test_get_model_returns_peft_model(
        self, mock_get_peft: MagicMock, mock_model, mock_tokenizer
    ) -> None:
        """get_model returns the PEFT model after preparation."""
        mock_peft_model = MagicMock()
        mock_get_peft.return_value = mock_peft_model

        trainer = Rank1DPOTrainer(mock_model, mock_tokenizer, Rank1DPOConfig())
        trainer.prepare_model()
        result = trainer.get_model()

        assert result == mock_peft_model


class TestDPODataset:
    """Test DPO preference pair dataset creation."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = MagicMock()
        model.device = torch.device("cpu")
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "</s>"
        return tokenizer

    @patch("src.training.dpo_trainer.get_peft_model")
    def test_creates_dataset_from_pairs(
        self, mock_get_peft: MagicMock, mock_model, mock_tokenizer
    ) -> None:
        """Creates HuggingFace dataset from preference pairs."""
        mock_peft_model = MagicMock()
        mock_get_peft.return_value = mock_peft_model

        trainer = Rank1DPOTrainer(mock_model, mock_tokenizer, Rank1DPOConfig())
        trainer.prepare_model()

        pairs = [
            {
                "prompt": "What is X?",
                "chosen": "X is Y.",
                "rejected": "X is Z.",
            }
        ]
        dataset = trainer._create_dataset(pairs)

        # Should return a Dataset-like object
        assert hasattr(dataset, "__len__")
        assert len(dataset) == 1


class TestDPOWithRefModel:
    """Test DPO with reference model handling."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = MagicMock()
        model.device = torch.device("cpu")
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        return tokenizer

    def test_accepts_ref_model_parameter(
        self, mock_model, mock_tokenizer
    ) -> None:
        """Trainer accepts reference model parameter."""
        ref_model = MagicMock()
        trainer = Rank1DPOTrainer(
            mock_model, mock_tokenizer, Rank1DPOConfig(), ref_model=ref_model
        )
        assert trainer.ref_model == ref_model

    def test_uses_base_model_as_ref_when_none(
        self, mock_model, mock_tokenizer
    ) -> None:
        """Uses base model as reference when ref_model is None."""
        trainer = Rank1DPOTrainer(mock_model, mock_tokenizer, Rank1DPOConfig())
        # ref_model should be None, will be set to base_model during training
        assert trainer.ref_model is None
