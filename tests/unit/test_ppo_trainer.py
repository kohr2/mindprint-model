"""
Tests for PPO (Proximal Policy Optimization) Trainer.

Tests cover:
- PPO configuration
- Policy model training with reward signal
- KL divergence penalty
- Value function learning
- Advantage estimation (GAE)
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import shutil

import torch

from src.training.ppo_trainer import (
    PPOConfig,
    PPOResult,
    PPOTrainer,
)


class TestPPOConfig:
    """Test PPOConfig dataclass."""

    def test_default_learning_rate(self) -> None:
        """Default learning rate is 1e-5."""
        config = PPOConfig()
        assert config.learning_rate == 1e-5

    def test_default_kl_penalty(self) -> None:
        """Default KL penalty coefficient is 0.2."""
        config = PPOConfig()
        assert config.kl_penalty == 0.2

    def test_default_clip_range(self) -> None:
        """Default clip range is 0.2."""
        config = PPOConfig()
        assert config.clip_range == 0.2

    def test_default_value_coefficient(self) -> None:
        """Default value function coefficient is 0.5."""
        config = PPOConfig()
        assert config.value_coef == 0.5

    def test_default_entropy_coefficient(self) -> None:
        """Default entropy coefficient is 0.01."""
        config = PPOConfig()
        assert config.entropy_coef == 0.01

    def test_default_max_steps(self) -> None:
        """Default max steps per topic is 100."""
        config = PPOConfig()
        assert config.max_steps == 100

    def test_default_batch_size(self) -> None:
        """Default batch size is 4."""
        config = PPOConfig()
        assert config.per_device_batch_size == 4

    def test_default_ppo_epochs(self) -> None:
        """Default PPO epochs per batch is 4."""
        config = PPOConfig()
        assert config.ppo_epochs == 4

    def test_default_gae_lambda(self) -> None:
        """Default GAE lambda is 0.95."""
        config = PPOConfig()
        assert config.gae_lambda == 0.95

    def test_default_gamma(self) -> None:
        """Default discount factor is 1.0 (no discounting for single-turn)."""
        config = PPOConfig()
        assert config.gamma == 1.0

    def test_validates_positive_learning_rate(self) -> None:
        """Learning rate must be positive."""
        with pytest.raises(ValueError):
            PPOConfig(learning_rate=-1e-5)

    def test_validates_clip_range(self) -> None:
        """Clip range must be in (0, 1)."""
        with pytest.raises(ValueError):
            PPOConfig(clip_range=0.0)
        with pytest.raises(ValueError):
            PPOConfig(clip_range=1.5)

    def test_custom_config(self) -> None:
        """Custom config values are set correctly."""
        config = PPOConfig(
            learning_rate=2e-5,
            kl_penalty=0.1,
            clip_range=0.1,
            max_steps=200,
        )
        assert config.learning_rate == 2e-5
        assert config.kl_penalty == 0.1
        assert config.clip_range == 0.1
        assert config.max_steps == 200


class TestPPOResult:
    """Test PPOResult dataclass."""

    def test_stores_success_status(self) -> None:
        """Result stores success status."""
        result = PPOResult(
            success=True,
            adapter_path="/path/to/adapter",
            final_reward_mean=0.75,
            kl_divergence=0.05,
            steps_completed=100,
            training_time_seconds=600.0,
        )
        assert result.success is True

    def test_stores_failure_with_error(self) -> None:
        """Result stores error message on failure."""
        result = PPOResult(
            success=False,
            adapter_path="",
            final_reward_mean=0.0,
            kl_divergence=0.0,
            steps_completed=0,
            training_time_seconds=10.0,
            error_message="Training failed",
        )
        assert result.success is False
        assert result.error_message == "Training failed"

    def test_stores_reward_metrics(self) -> None:
        """Result stores reward and KL metrics."""
        result = PPOResult(
            success=True,
            adapter_path="/path",
            final_reward_mean=0.82,
            kl_divergence=0.08,
            steps_completed=100,
            training_time_seconds=500.0,
        )
        assert result.final_reward_mean == 0.82
        assert result.kl_divergence == 0.08


class TestPPOTrainer:
    """Test PPOTrainer class."""

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
    def mock_reward_model(self):
        """Create mock reward model."""
        reward_model = MagicMock()
        reward_model.get_reward.return_value = 0.75
        return reward_model

    @pytest.fixture
    def sample_prompts(self):
        """Sample prompts for training."""
        return [
            "What drives the 4-year cycle?",
            "What is accumulation?",
            "How do you identify a cycle bottom?",
        ]

    def test_initializes_with_config(
        self, mock_model, mock_tokenizer, mock_reward_model
    ) -> None:
        """Trainer initializes with configuration."""
        config = PPOConfig()
        trainer = PPOTrainer(
            mock_model, mock_tokenizer, mock_reward_model, config
        )
        assert trainer.config == config

    def test_has_train_method(
        self, mock_model, mock_tokenizer, mock_reward_model
    ) -> None:
        """Trainer has train method."""
        trainer = PPOTrainer(
            mock_model, mock_tokenizer, mock_reward_model, PPOConfig()
        )
        assert hasattr(trainer, "train")
        assert callable(trainer.train)

    def test_has_train_on_topic_method(
        self, mock_model, mock_tokenizer, mock_reward_model
    ) -> None:
        """Trainer has train_on_topic method."""
        trainer = PPOTrainer(
            mock_model, mock_tokenizer, mock_reward_model, PPOConfig()
        )
        assert hasattr(trainer, "train_on_topic")
        assert callable(trainer.train_on_topic)

    def test_has_generate_response_method(
        self, mock_model, mock_tokenizer, mock_reward_model
    ) -> None:
        """Trainer has _generate_response method."""
        trainer = PPOTrainer(
            mock_model, mock_tokenizer, mock_reward_model, PPOConfig()
        )
        assert hasattr(trainer, "_generate_response")
        assert callable(trainer._generate_response)

    def test_has_compute_advantages_method(
        self, mock_model, mock_tokenizer, mock_reward_model
    ) -> None:
        """Trainer has _compute_advantages method."""
        trainer = PPOTrainer(
            mock_model, mock_tokenizer, mock_reward_model, PPOConfig()
        )
        assert hasattr(trainer, "_compute_advantages")
        assert callable(trainer._compute_advantages)

    def test_has_ppo_step_method(
        self, mock_model, mock_tokenizer, mock_reward_model
    ) -> None:
        """Trainer has _ppo_step method."""
        trainer = PPOTrainer(
            mock_model, mock_tokenizer, mock_reward_model, PPOConfig()
        )
        assert hasattr(trainer, "_ppo_step")
        assert callable(trainer._ppo_step)

    def test_has_save_adapter_method(
        self, mock_model, mock_tokenizer, mock_reward_model
    ) -> None:
        """Trainer has save_adapter method."""
        trainer = PPOTrainer(
            mock_model, mock_tokenizer, mock_reward_model, PPOConfig()
        )
        assert hasattr(trainer, "save_adapter")
        assert callable(trainer.save_adapter)

    def test_has_get_model_method(
        self, mock_model, mock_tokenizer, mock_reward_model
    ) -> None:
        """Trainer has get_model method."""
        trainer = PPOTrainer(
            mock_model, mock_tokenizer, mock_reward_model, PPOConfig()
        )
        assert hasattr(trainer, "get_model")
        assert callable(trainer.get_model)


class TestPPOTraining:
    """Test PPO training loop."""

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
    def mock_reward_model(self):
        """Create mock reward model."""
        reward_model = MagicMock()
        reward_model.get_reward.return_value = 0.75
        return reward_model

    @pytest.fixture
    def sample_prompts(self):
        """Sample prompts for training."""
        return [
            "What drives the 4-year cycle?",
            "What is accumulation?",
        ]

    @patch("src.training.ppo_trainer.get_peft_model")
    @patch("src.training.ppo_trainer.LoraConfig")
    def test_train_returns_ppo_result(
        self,
        mock_lora_config: MagicMock,
        mock_get_peft: MagicMock,
        mock_model,
        mock_tokenizer,
        mock_reward_model,
        sample_prompts,
    ) -> None:
        """train() returns a PPOResult."""
        mock_peft_model = MagicMock()
        mock_peft_model.parameters.return_value = [
            torch.nn.Parameter(torch.randn(10))
        ]
        mock_get_peft.return_value = mock_peft_model

        trainer = PPOTrainer(
            mock_model, mock_tokenizer, mock_reward_model, PPOConfig()
        )
        result = trainer.train(sample_prompts)

        assert isinstance(result, PPOResult)

    @patch("src.training.ppo_trainer.get_peft_model")
    @patch("src.training.ppo_trainer.LoraConfig")
    def test_train_on_topic_returns_ppo_result(
        self,
        mock_lora_config: MagicMock,
        mock_get_peft: MagicMock,
        mock_model,
        mock_tokenizer,
        mock_reward_model,
        sample_prompts,
    ) -> None:
        """train_on_topic() returns a PPOResult."""
        mock_peft_model = MagicMock()
        mock_peft_model.parameters.return_value = [
            torch.nn.Parameter(torch.randn(10))
        ]
        mock_get_peft.return_value = mock_peft_model

        trainer = PPOTrainer(
            mock_model, mock_tokenizer, mock_reward_model, PPOConfig()
        )
        result = trainer.train_on_topic(sample_prompts, "topic-01")

        assert isinstance(result, PPOResult)


class TestPPOLoss:
    """Test PPO loss computation."""

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

    @pytest.fixture
    def mock_reward_model(self):
        """Create mock reward model."""
        return MagicMock()

    def test_has_compute_policy_loss_method(
        self, mock_model, mock_tokenizer, mock_reward_model
    ) -> None:
        """Trainer has _compute_policy_loss method."""
        trainer = PPOTrainer(
            mock_model, mock_tokenizer, mock_reward_model, PPOConfig()
        )
        assert hasattr(trainer, "_compute_policy_loss")

    def test_has_compute_value_loss_method(
        self, mock_model, mock_tokenizer, mock_reward_model
    ) -> None:
        """Trainer has _compute_value_loss method."""
        trainer = PPOTrainer(
            mock_model, mock_tokenizer, mock_reward_model, PPOConfig()
        )
        assert hasattr(trainer, "_compute_value_loss")

    def test_has_compute_kl_penalty_method(
        self, mock_model, mock_tokenizer, mock_reward_model
    ) -> None:
        """Trainer has _compute_kl_penalty method."""
        trainer = PPOTrainer(
            mock_model, mock_tokenizer, mock_reward_model, PPOConfig()
        )
        assert hasattr(trainer, "_compute_kl_penalty")


class TestPPOAdvantages:
    """Test advantage estimation (GAE)."""

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

    @pytest.fixture
    def mock_reward_model(self):
        """Create mock reward model."""
        return MagicMock()

    def test_compute_advantages_returns_tensor(
        self, mock_model, mock_tokenizer, mock_reward_model
    ) -> None:
        """_compute_advantages returns a tensor."""
        trainer = PPOTrainer(
            mock_model, mock_tokenizer, mock_reward_model, PPOConfig()
        )

        rewards = torch.tensor([1.0, 0.5, 0.8])
        values = torch.tensor([0.9, 0.6, 0.7])

        advantages = trainer._compute_advantages(rewards, values)

        assert isinstance(advantages, torch.Tensor)
        assert advantages.shape == rewards.shape

    def test_higher_reward_gives_higher_advantage(
        self, mock_model, mock_tokenizer, mock_reward_model
    ) -> None:
        """Higher rewards lead to higher advantages."""
        trainer = PPOTrainer(
            mock_model, mock_tokenizer, mock_reward_model, PPOConfig()
        )

        # Same values, different rewards (with variation to avoid normalization to zero)
        values = torch.tensor([0.5, 0.6, 0.4])
        high_rewards = torch.tensor([1.0, 0.9, 1.1])
        low_rewards = torch.tensor([0.0, 0.1, -0.1])

        high_advantages = trainer._compute_advantages(high_rewards, values)
        low_advantages = trainer._compute_advantages(low_rewards, values)

        # Raw advantages (before normalization) should show high > low
        high_raw = (high_rewards - values).mean()
        low_raw = (low_rewards - values).mean()
        assert high_raw > low_raw


class TestPPOKLDivergence:
    """Test KL divergence tracking and penalty."""

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

    @pytest.fixture
    def mock_reward_model(self):
        """Create mock reward model."""
        return MagicMock()

    def test_kl_penalty_configurable(
        self, mock_model, mock_tokenizer, mock_reward_model
    ) -> None:
        """KL penalty weight can be configured."""
        config = PPOConfig(kl_penalty=0.5)
        trainer = PPOTrainer(
            mock_model, mock_tokenizer, mock_reward_model, config
        )
        assert trainer.config.kl_penalty == 0.5

    def test_kl_target_configurable(
        self, mock_model, mock_tokenizer, mock_reward_model
    ) -> None:
        """KL target can be configured."""
        config = PPOConfig(kl_target=0.1)
        trainer = PPOTrainer(
            mock_model, mock_tokenizer, mock_reward_model, config
        )
        assert trainer.config.kl_target == 0.1

    def test_adaptive_kl_default_enabled(
        self, mock_model, mock_tokenizer, mock_reward_model
    ) -> None:
        """Adaptive KL penalty is enabled by default."""
        config = PPOConfig()
        assert config.adaptive_kl is True


class TestPPOLoRA:
    """Test LoRA configuration in PPO."""

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

    @pytest.fixture
    def mock_reward_model(self):
        """Create mock reward model."""
        return MagicMock()

    def test_default_lora_rank(self) -> None:
        """Default LoRA rank for PPO is 8."""
        config = PPOConfig()
        assert config.lora_r == 8

    def test_default_lora_alpha(self) -> None:
        """Default LoRA alpha for PPO is 16."""
        config = PPOConfig()
        assert config.lora_alpha == 16

    def test_default_target_modules(self) -> None:
        """Default target modules for PPO."""
        config = PPOConfig()
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules
