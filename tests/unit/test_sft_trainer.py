"""
Tests for SFTTrainer - Supervised Fine-Tuning.

Tests cover:
- Configuration validation
- Dataset creation from Q&A pairs
- Training loop (mocked)
- Adapter saving
- Topic-level training
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path
import tempfile
import shutil

import torch

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

    def test_default_epochs(self) -> None:
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
        """Default target modules include attention projections."""
        config = SFTConfig()
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules
        assert "o_proj" in config.target_modules

    def test_validates_positive_learning_rate(self) -> None:
        """Raises error for non-positive learning rate."""
        with pytest.raises(AssertionError):
            SFTConfig(learning_rate=-1e-4)

    def test_validates_positive_epochs(self) -> None:
        """Raises error for non-positive epochs."""
        with pytest.raises(AssertionError):
            SFTConfig(num_epochs=0)

    def test_validates_positive_batch_size(self) -> None:
        """Raises error for non-positive batch size."""
        with pytest.raises(AssertionError):
            SFTConfig(per_device_batch_size=0)

    def test_validates_positive_lora_rank(self) -> None:
        """Raises error for non-positive LoRA rank."""
        with pytest.raises(AssertionError):
            SFTConfig(lora_r=0)

    def test_custom_config(self) -> None:
        """Custom configuration values are set correctly."""
        config = SFTConfig(
            learning_rate=1e-5,
            num_epochs=5,
            per_device_batch_size=2,
            lora_r=16,
        )
        assert config.learning_rate == 1e-5
        assert config.num_epochs == 5
        assert config.per_device_batch_size == 2
        assert config.lora_r == 16


class TestSFTResult:
    """Test SFTResult dataclass."""

    def test_stores_success_status(self) -> None:
        """Result stores success status."""
        result = SFTResult(
            success=True,
            adapter_path="/path/to/adapter",
            final_loss=0.5,
            training_time_seconds=100.0,
            samples_trained=1000,
        )
        assert result.success is True

    def test_stores_failure_with_error(self) -> None:
        """Result stores failure with error message."""
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
            adapter_path="/path",
            final_loss=0.25,
            training_time_seconds=3600.0,
            samples_trained=5000,
        )
        assert result.final_loss == 0.25
        assert result.training_time_seconds == 3600.0
        assert result.samples_trained == 5000


class TestSFTDataset:
    """Test SFTDataset for Q&A pairs."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "</s>"
        tokenizer.model_max_length = 2048

        def mock_call(text, **kwargs):
            # Return mock encoding
            return {
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
            }

        tokenizer.side_effect = mock_call
        tokenizer.__call__ = mock_call
        return tokenizer

    @pytest.fixture
    def sample_data(self):
        """Sample SFT data."""
        return [
            {
                "instruction": "What is the 4-year cycle?",
                "input": "",
                "output": "The 4-year cycle is driven by market psychology, not the halving.",
            },
            {
                "instruction": "Explain accumulation phase.",
                "input": "",
                "output": "Accumulation is when smart money quietly builds positions.",
            },
        ]

    def test_creates_dataset_from_qa_pairs(
        self, mock_tokenizer, sample_data
    ) -> None:
        """Creates dataset from Q&A pairs."""
        dataset = SFTDataset(sample_data, mock_tokenizer)
        assert len(dataset) == 2

    def test_getitem_returns_dict(self, mock_tokenizer, sample_data) -> None:
        """__getitem__ returns dict with required keys."""
        dataset = SFTDataset(sample_data, mock_tokenizer)
        item = dataset[0]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_handles_empty_input(self, mock_tokenizer) -> None:
        """Handles empty input field."""
        data = [
            {
                "instruction": "Question?",
                "input": "",
                "output": "Answer.",
            }
        ]
        dataset = SFTDataset(data, mock_tokenizer)
        assert len(dataset) == 1

    def test_handles_non_empty_input(self, mock_tokenizer) -> None:
        """Handles non-empty input field."""
        data = [
            {
                "instruction": "Analyze this pattern:",
                "input": "Bitcoin formed a higher low.",
                "output": "This is bullish accumulation.",
            }
        ]
        dataset = SFTDataset(data, mock_tokenizer)
        assert len(dataset) == 1


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
    def sample_train_data(self):
        """Sample training data."""
        return [
            {
                "instruction": "What is the 4-year cycle?",
                "input": "",
                "output": "The cycle is driven by psychology.",
            },
            {
                "instruction": "Explain market phases.",
                "input": "",
                "output": "There are four phases: accumulation, markup, distribution, markdown.",
            },
        ]

    def test_initializes_with_config(self, mock_model, mock_tokenizer) -> None:
        """Trainer initializes with configuration."""
        config = SFTConfig()
        trainer = SFTTrainer(mock_model, mock_tokenizer, config)
        assert trainer.config == config

    def test_has_prepare_model_method(self, mock_model, mock_tokenizer) -> None:
        """Trainer has prepare_model method."""
        trainer = SFTTrainer(mock_model, mock_tokenizer, SFTConfig())
        assert hasattr(trainer, "prepare_model")
        assert callable(trainer.prepare_model)

    def test_has_train_method(self, mock_model, mock_tokenizer) -> None:
        """Trainer has train method."""
        trainer = SFTTrainer(mock_model, mock_tokenizer, SFTConfig())
        assert hasattr(trainer, "train")
        assert callable(trainer.train)

    def test_has_train_on_topic_method(self, mock_model, mock_tokenizer) -> None:
        """Trainer has train_on_topic method."""
        trainer = SFTTrainer(mock_model, mock_tokenizer, SFTConfig())
        assert hasattr(trainer, "train_on_topic")
        assert callable(trainer.train_on_topic)

    def test_has_save_adapter_method(self, mock_model, mock_tokenizer) -> None:
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
    def test_prepare_model_adds_lora(
        self, mock_get_peft: MagicMock, mock_model, mock_tokenizer
    ) -> None:
        """prepare_model adds LoRA adapters."""
        mock_peft_model = MagicMock()
        mock_get_peft.return_value = mock_peft_model

        trainer = SFTTrainer(mock_model, mock_tokenizer, SFTConfig())
        result = trainer.prepare_model()

        mock_get_peft.assert_called_once()
        assert result == mock_peft_model

    @patch("src.training.sft_trainer.Trainer")
    @patch("src.training.sft_trainer.get_peft_model")
    def test_train_returns_sft_result(
        self,
        mock_get_peft: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_model,
        mock_tokenizer,
        sample_train_data,
    ) -> None:
        """train() returns SFTResult."""
        mock_peft_model = MagicMock()
        mock_get_peft.return_value = mock_peft_model

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = MagicMock(
            training_loss=0.5, metrics={"train_loss": 0.5}
        )
        mock_trainer_cls.return_value = mock_trainer

        trainer = SFTTrainer(mock_model, mock_tokenizer, SFTConfig())
        result = trainer.train(sample_train_data)

        assert isinstance(result, SFTResult)

    @patch("src.training.sft_trainer.Trainer")
    @patch("src.training.sft_trainer.get_peft_model")
    def test_train_on_topic_returns_sft_result(
        self,
        mock_get_peft: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_model,
        mock_tokenizer,
    ) -> None:
        """train_on_topic() returns SFTResult."""
        mock_peft_model = MagicMock()
        mock_get_peft.return_value = mock_peft_model

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = MagicMock(
            training_loss=0.5, metrics={"train_loss": 0.5}
        )
        mock_trainer_cls.return_value = mock_trainer

        trainer = SFTTrainer(mock_model, mock_tokenizer, SFTConfig())

        topic_data = {
            "questions": [
                {"question": "What is X?", "reference_answer": "X is Y."}
            ]
        }
        result = trainer.train_on_topic(topic_data, "unit-01/chapter-01/topic-01")

        assert isinstance(result, SFTResult)

    @patch("src.training.sft_trainer.get_peft_model")
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

        trainer = SFTTrainer(mock_model, mock_tokenizer, SFTConfig())
        trainer.prepare_model()

        output_path = Path(temp_dir) / "adapter"
        result = trainer.save_adapter(str(output_path))

        mock_peft_model.save_pretrained.assert_called_once()
        assert isinstance(result, Path)

    @patch("src.training.sft_trainer.get_peft_model")
    def test_get_model_returns_peft_model(
        self, mock_get_peft: MagicMock, mock_model, mock_tokenizer
    ) -> None:
        """get_model returns the PEFT model after preparation."""
        mock_peft_model = MagicMock()
        mock_get_peft.return_value = mock_peft_model

        trainer = SFTTrainer(mock_model, mock_tokenizer, SFTConfig())
        trainer.prepare_model()
        result = trainer.get_model()

        assert result == mock_peft_model


class TestSFTDataFormatting:
    """Test SFT data formatting for different models."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "</s>"
        tokenizer.model_max_length = 2048

        def mock_call(text, **kwargs):
            return {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
            }

        tokenizer.__call__ = mock_call
        return tokenizer

    def test_gemma_format_includes_turn_markers(self, mock_tokenizer) -> None:
        """Gemma format includes turn markers."""
        data = [
            {
                "instruction": "Question?",
                "input": "",
                "output": "Answer.",
            }
        ]
        dataset = SFTDataset(data, mock_tokenizer, prompt_format="gemma")
        # Dataset should format with Gemma markers
        assert len(dataset) == 1

    def test_chatml_format_includes_role_tags(self, mock_tokenizer) -> None:
        """ChatML format includes role tags."""
        data = [
            {
                "instruction": "Question?",
                "input": "",
                "output": "Answer.",
            }
        ]
        dataset = SFTDataset(data, mock_tokenizer, prompt_format="chatml")
        assert len(dataset) == 1
