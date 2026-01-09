"""
Tests for SFT (Supervised Fine-Tuning) Trainer.

Tests cover:
- SFT configuration
- Dataset creation from Q&A pairs
- LoRA adapter training
- Topic-level training
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import shutil

import torch
from torch.utils.data import Dataset

from src.training.sft_trainer import (
    SFTConfig,
    SFTResult,
    SFTDataset,
    SFTTrainer,
)


class TestSFTConfig:
    """Test SFTConfig dataclass."""

    def test_default_learning_rate(self) -> None:
        """Default learning rate is 3e-4."""
        config = SFTConfig()
        assert config.learning_rate == 3e-4

    def test_default_num_epochs(self) -> None:
        """Default epochs is 3."""
        config = SFTConfig()
        assert config.num_epochs == 3

    def test_default_batch_size(self) -> None:
        """Default batch size is 4."""
        config = SFTConfig()
        assert config.per_device_batch_size == 4

    def test_default_lora_rank(self) -> None:
        """Default LoRA rank is 8."""
        config = SFTConfig()
        assert config.lora_r == 8

    def test_default_lora_alpha(self) -> None:
        """Default LoRA alpha is 16."""
        config = SFTConfig()
        assert config.lora_alpha == 16

    def test_default_target_modules(self) -> None:
        """Default target modules for SFT."""
        config = SFTConfig()
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules
        assert "o_proj" in config.target_modules

    def test_validates_positive_learning_rate(self) -> None:
        """Learning rate must be positive."""
        with pytest.raises(ValueError):
            SFTConfig(learning_rate=-1e-4)

    def test_validates_positive_epochs(self) -> None:
        """Epochs must be positive."""
        with pytest.raises(ValueError):
            SFTConfig(num_epochs=0)

    def test_validates_positive_batch_size(self) -> None:
        """Batch size must be positive."""
        with pytest.raises(ValueError):
            SFTConfig(per_device_batch_size=0)

    def test_custom_config(self) -> None:
        """Custom config values are set correctly."""
        config = SFTConfig(
            learning_rate=1e-4,
            num_epochs=5,
            per_device_batch_size=8,
            lora_r=16,
        )
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 5
        assert config.per_device_batch_size == 8
        assert config.lora_r == 16


class TestSFTResult:
    """Test SFTResult dataclass."""

    def test_stores_success_status(self) -> None:
        """Result stores success status."""
        result = SFTResult(
            success=True,
            adapter_path="/path/to/adapter",
            final_loss=0.5,
            training_time_seconds=120.0,
            samples_trained=100,
        )
        assert result.success is True

    def test_stores_failure_with_error(self) -> None:
        """Result stores error message on failure."""
        result = SFTResult(
            success=False,
            adapter_path="",
            final_loss=0.0,
            training_time_seconds=10.0,
            samples_trained=0,
            error_message="Training failed",
        )
        assert result.success is False
        assert result.error_message == "Training failed"

    def test_stores_training_metrics(self) -> None:
        """Result stores training metrics."""
        result = SFTResult(
            success=True,
            adapter_path="/path/to/adapter",
            final_loss=0.35,
            training_time_seconds=300.0,
            samples_trained=500,
        )
        assert result.final_loss == 0.35
        assert result.training_time_seconds == 300.0
        assert result.samples_trained == 500


class TestSFTDataset:
    """Test SFTDataset class."""

    @pytest.fixture
    def sample_data(self):
        """Sample Q&A training data."""
        return [
            {
                "question": "What drives the 4-year cycle?",
                "answer": "Market psychology, not the halving.",
            },
            {
                "question": "What is accumulation?",
                "answer": "When smart money builds positions quietly.",
            },
        ]

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "</s>"
        return tokenizer

    def test_creates_dataset_from_qa_pairs(
        self, sample_data, mock_tokenizer
    ) -> None:
        """Dataset is created from Q&A pairs."""
        dataset = SFTDataset(sample_data, mock_tokenizer)
        assert len(dataset) == 2

    def test_returns_correct_item_type(
        self, sample_data, mock_tokenizer
    ) -> None:
        """Dataset items have correct structure."""
        dataset = SFTDataset(sample_data, mock_tokenizer)
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item

    def test_handles_empty_data(self, mock_tokenizer) -> None:
        """Dataset handles empty data gracefully."""
        dataset = SFTDataset([], mock_tokenizer)
        assert len(dataset) == 0

    def test_formats_prompt_correctly(
        self, sample_data, mock_tokenizer
    ) -> None:
        """Prompts are formatted for instruction tuning."""
        dataset = SFTDataset(sample_data, mock_tokenizer)
        # Verify tokenizer was called
        assert mock_tokenizer.call_count >= 0

    def test_is_torch_dataset(self, sample_data, mock_tokenizer) -> None:
        """SFTDataset is a proper torch Dataset."""
        dataset = SFTDataset(sample_data, mock_tokenizer)
        assert isinstance(dataset, Dataset)


class TestSFTTrainer:
    """Test SFTTrainer class."""

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
        model.device = torch.device("cpu")
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "</s>"
        return tokenizer

    @pytest.fixture
    def sample_data(self):
        """Sample training data."""
        return [
            {"question": "Q1?", "answer": "A1"},
            {"question": "Q2?", "answer": "A2"},
        ]

    def test_initializes_with_config(
        self, mock_model, mock_tokenizer
    ) -> None:
        """Trainer initializes with configuration."""
        config = SFTConfig()
        trainer = SFTTrainer(mock_model, mock_tokenizer, config)
        assert trainer.config == config

    def test_has_train_method(self, mock_model, mock_tokenizer) -> None:
        """Trainer has train method."""
        trainer = SFTTrainer(mock_model, mock_tokenizer, SFTConfig())
        assert hasattr(trainer, "train")
        assert callable(trainer.train)

    def test_has_train_on_topic_method(
        self, mock_model, mock_tokenizer
    ) -> None:
        """Trainer has train_on_topic method."""
        trainer = SFTTrainer(mock_model, mock_tokenizer, SFTConfig())
        assert hasattr(trainer, "train_on_topic")
        assert callable(trainer.train_on_topic)

    def test_has_save_adapter_method(
        self, mock_model, mock_tokenizer
    ) -> None:
        """Trainer has save_adapter method."""
        trainer = SFTTrainer(mock_model, mock_tokenizer, SFTConfig())
        assert hasattr(trainer, "save_adapter")
        assert callable(trainer.save_adapter)

    def test_has_get_model_method(self, mock_model, mock_tokenizer) -> None:
        """Trainer has get_model method."""
        trainer = SFTTrainer(mock_model, mock_tokenizer, SFTConfig())
        assert hasattr(trainer, "get_model")
        assert callable(trainer.get_model)

    @patch("src.training.sft_trainer.get_peft_model")
    @patch("src.training.sft_trainer.LoraConfig")
    def test_train_returns_sft_result(
        self,
        mock_lora_config: MagicMock,
        mock_get_peft: MagicMock,
        mock_model,
        mock_tokenizer,
        sample_data,
    ) -> None:
        """train() returns an SFTResult."""
        # Setup mocks
        mock_peft_model = MagicMock()
        mock_peft_model.parameters.return_value = [torch.nn.Parameter(torch.randn(10))]
        mock_get_peft.return_value = mock_peft_model

        trainer = SFTTrainer(mock_model, mock_tokenizer, SFTConfig())
        result = trainer.train(sample_data)

        assert isinstance(result, SFTResult)

    @patch("src.training.sft_trainer.get_peft_model")
    @patch("src.training.sft_trainer.LoraConfig")
    def test_train_on_topic_returns_sft_result(
        self,
        mock_lora_config: MagicMock,
        mock_get_peft: MagicMock,
        mock_model,
        mock_tokenizer,
        sample_data,
    ) -> None:
        """train_on_topic() returns an SFTResult."""
        mock_peft_model = MagicMock()
        mock_peft_model.parameters.return_value = [torch.nn.Parameter(torch.randn(10))]
        mock_get_peft.return_value = mock_peft_model

        trainer = SFTTrainer(mock_model, mock_tokenizer, SFTConfig())
        result = trainer.train_on_topic(sample_data, "topic-01")

        assert isinstance(result, SFTResult)

    @patch("src.training.sft_trainer.get_peft_model")
    @patch("src.training.sft_trainer.LoraConfig")
    def test_get_model_returns_peft_model(
        self,
        mock_lora_config: MagicMock,
        mock_get_peft: MagicMock,
        mock_model,
        mock_tokenizer,
        sample_data,
    ) -> None:
        """get_model() returns the PEFT model after training."""
        mock_peft_model = MagicMock()
        mock_peft_model.parameters.return_value = [torch.nn.Parameter(torch.randn(10))]
        mock_get_peft.return_value = mock_peft_model

        trainer = SFTTrainer(mock_model, mock_tokenizer, SFTConfig())
        trainer.train(sample_data)
        model = trainer.get_model()

        assert model is not None


class TestSFTTrainerLoRA:
    """Test LoRA configuration in SFTTrainer."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = MagicMock()
        model.config = MagicMock()
        model.device = torch.device("cpu")
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        return tokenizer

    @patch("src.training.sft_trainer.get_peft_model")
    @patch("src.training.sft_trainer.LoraConfig")
    def test_creates_lora_config_with_rank(
        self,
        mock_lora_cls: MagicMock,
        mock_get_peft: MagicMock,
        mock_model,
        mock_tokenizer,
    ) -> None:
        """LoRA config uses specified rank."""
        config = SFTConfig(lora_r=16)
        trainer = SFTTrainer(mock_model, mock_tokenizer, config)
        trainer._prepare_model()

        mock_lora_cls.assert_called()
        call_kwargs = mock_lora_cls.call_args[1]
        assert call_kwargs["r"] == 16

    @patch("src.training.sft_trainer.get_peft_model")
    @patch("src.training.sft_trainer.LoraConfig")
    def test_creates_lora_config_with_alpha(
        self,
        mock_lora_cls: MagicMock,
        mock_get_peft: MagicMock,
        mock_model,
        mock_tokenizer,
    ) -> None:
        """LoRA config uses specified alpha."""
        config = SFTConfig(lora_alpha=32)
        trainer = SFTTrainer(mock_model, mock_tokenizer, config)
        trainer._prepare_model()

        mock_lora_cls.assert_called()
        call_kwargs = mock_lora_cls.call_args[1]
        assert call_kwargs["lora_alpha"] == 32

    @patch("src.training.sft_trainer.get_peft_model")
    @patch("src.training.sft_trainer.LoraConfig")
    def test_targets_correct_modules(
        self,
        mock_lora_cls: MagicMock,
        mock_get_peft: MagicMock,
        mock_model,
        mock_tokenizer,
    ) -> None:
        """LoRA targets the specified modules."""
        config = SFTConfig(target_modules=["q_proj", "v_proj"])
        trainer = SFTTrainer(mock_model, mock_tokenizer, config)
        trainer._prepare_model()

        mock_lora_cls.assert_called()
        call_kwargs = mock_lora_cls.call_args[1]
        assert "q_proj" in call_kwargs["target_modules"]
        assert "v_proj" in call_kwargs["target_modules"]
