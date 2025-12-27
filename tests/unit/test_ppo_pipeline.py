"""
Tests for PPOPipeline - SFT + Reward Model + PPO Training Orchestration.

Tests cover:
- Pipeline configuration
- Three-phase training: SFT → Reward Model → PPO
- Topic/Unit progress tracking
- Checkpoint save/resume
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import shutil
import json

import torch

from src.training.ppo_pipeline import (
    TopicStatus,
    PipelineConfig,
    TopicProgress,
    ChapterProgress,
    UnitProgress,
    PipelineResult,
    PPOPipeline,
)


class TestTopicStatus:
    """Test TopicStatus enum."""

    def test_pending_status_exists(self) -> None:
        """TopicStatus has PENDING value."""
        assert TopicStatus.PENDING is not None

    def test_sft_complete_status_exists(self) -> None:
        """TopicStatus has SFT_COMPLETE value."""
        assert TopicStatus.SFT_COMPLETE is not None

    def test_reward_training_status_exists(self) -> None:
        """TopicStatus has REWARD_TRAINING value."""
        assert TopicStatus.REWARD_TRAINING is not None

    def test_ppo_training_status_exists(self) -> None:
        """TopicStatus has PPO_TRAINING value."""
        assert TopicStatus.PPO_TRAINING is not None

    def test_passed_status_exists(self) -> None:
        """TopicStatus has PASSED value."""
        assert TopicStatus.PASSED is not None

    def test_failed_status_exists(self) -> None:
        """TopicStatus has FAILED value."""
        assert TopicStatus.FAILED is not None


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""

    def test_default_sft_epochs(self) -> None:
        """Default SFT epochs is 3."""
        config = PipelineConfig()
        assert config.sft_epochs_per_topic == 3

    def test_default_ppo_steps(self) -> None:
        """Default PPO steps is 100."""
        config = PipelineConfig()
        assert config.ppo_steps_per_topic == 100

    def test_default_reward_epochs(self) -> None:
        """Default reward model epochs is 1."""
        config = PipelineConfig()
        assert config.reward_model_epochs == 1

    def test_default_topic_pass_threshold(self) -> None:
        """Default topic pass threshold is 0.85."""
        config = PipelineConfig()
        assert config.topic_pass_threshold == 0.85

    def test_default_merge_after_unit(self) -> None:
        """Default merge_after_unit is True."""
        config = PipelineConfig()
        assert config.merge_after_unit is True

    def test_custom_config(self) -> None:
        """Custom config values are set correctly."""
        config = PipelineConfig(
            sft_epochs_per_topic=5,
            ppo_steps_per_topic=200,
            reward_model_epochs=2,
        )
        assert config.sft_epochs_per_topic == 5
        assert config.ppo_steps_per_topic == 200
        assert config.reward_model_epochs == 2


class TestTopicProgress:
    """Test TopicProgress dataclass."""

    def test_stores_topic_id(self) -> None:
        """Progress stores topic ID."""
        progress = TopicProgress(
            topic_id="unit-01/chapter-01/topic-01",
            status=TopicStatus.PENDING,
        )
        assert progress.topic_id == "unit-01/chapter-01/topic-01"

    def test_stores_status(self) -> None:
        """Progress stores status."""
        progress = TopicProgress(
            topic_id="unit-01/chapter-01/topic-01",
            status=TopicStatus.SFT_COMPLETE,
        )
        assert progress.status == TopicStatus.SFT_COMPLETE

    def test_stores_reward_score(self) -> None:
        """Progress stores reward score."""
        progress = TopicProgress(
            topic_id="test",
            status=TopicStatus.PPO_TRAINING,
            reward_score=0.78,
        )
        assert progress.reward_score == 0.78


class TestChapterProgress:
    """Test ChapterProgress dataclass."""

    def test_stores_chapter_id(self) -> None:
        """ChapterProgress stores chapter ID."""
        progress = ChapterProgress(
            chapter_id="unit-01/chapter-01",
            topics=[],
        )
        assert progress.chapter_id == "unit-01/chapter-01"

    def test_passed_topics_count(self) -> None:
        """ChapterProgress counts passed topics."""
        topics = [
            TopicProgress("t1", TopicStatus.PASSED),
            TopicProgress("t2", TopicStatus.PASSED),
            TopicProgress("t3", TopicStatus.FAILED),
        ]
        progress = ChapterProgress(
            chapter_id="chapter-01",
            topics=topics,
        )
        assert progress.passed_count == 2


class TestUnitProgress:
    """Test UnitProgress dataclass."""

    def test_stores_unit_id(self) -> None:
        """UnitProgress stores unit ID."""
        progress = UnitProgress(
            unit_id="unit-01",
            chapters=[],
        )
        assert progress.unit_id == "unit-01"

    def test_merged_status(self) -> None:
        """UnitProgress tracks merge status."""
        progress = UnitProgress(
            unit_id="unit-01",
            chapters=[],
            merged=True,
        )
        assert progress.merged is True


class TestPipelineResult:
    """Test PipelineResult dataclass."""

    def test_stores_success_status(self) -> None:
        """Result stores success status."""
        result = PipelineResult(
            success=True,
            total_topics=52,
            passed_topics=50,
            failed_topics=["t1", "t2"],
            total_training_time_hours=48.5,
        )
        assert result.success is True

    def test_stores_topic_counts(self) -> None:
        """Result stores topic counts."""
        result = PipelineResult(
            success=True,
            total_topics=52,
            passed_topics=50,
            failed_topics=["t1", "t2"],
            total_training_time_hours=48.5,
        )
        assert result.total_topics == 52
        assert result.passed_topics == 50


class TestPPOPipeline:
    """Test PPOPipeline class."""

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
        model.config.hidden_size = 768
        model.device = torch.device("cpu")
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "</s>"
        return tokenizer

    def test_initializes_with_config(self, mock_model, mock_tokenizer) -> None:
        """Pipeline initializes with configuration."""
        config = PipelineConfig()
        pipeline = PPOPipeline(mock_model, mock_tokenizer, config)
        assert pipeline.config == config

    def test_has_train_curriculum_method(self, mock_model, mock_tokenizer) -> None:
        """Pipeline has train_curriculum method."""
        pipeline = PPOPipeline(mock_model, mock_tokenizer, PipelineConfig())
        assert hasattr(pipeline, "train_curriculum")
        assert callable(pipeline.train_curriculum)

    def test_has_train_unit_method(self, mock_model, mock_tokenizer) -> None:
        """Pipeline has train_unit method."""
        pipeline = PPOPipeline(mock_model, mock_tokenizer, PipelineConfig())
        assert hasattr(pipeline, "train_unit")
        assert callable(pipeline.train_unit)

    def test_has_train_topic_method(self, mock_model, mock_tokenizer) -> None:
        """Pipeline has train_topic method."""
        pipeline = PPOPipeline(mock_model, mock_tokenizer, PipelineConfig())
        assert hasattr(pipeline, "train_topic")
        assert callable(pipeline.train_topic)

    def test_has_save_checkpoint_method(self, mock_model, mock_tokenizer) -> None:
        """Pipeline has save_checkpoint method."""
        pipeline = PPOPipeline(mock_model, mock_tokenizer, PipelineConfig())
        assert hasattr(pipeline, "save_checkpoint")
        assert callable(pipeline.save_checkpoint)

    def test_has_resume_from_checkpoint_method(
        self, mock_model, mock_tokenizer
    ) -> None:
        """Pipeline has resume_from_checkpoint method."""
        pipeline = PPOPipeline(mock_model, mock_tokenizer, PipelineConfig())
        assert hasattr(pipeline, "resume_from_checkpoint")
        assert callable(pipeline.resume_from_checkpoint)


class TestThreePhaseTraining:
    """Test the three-phase training: SFT → Reward → PPO."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = MagicMock()
        model.config = MagicMock()
        model.config.hidden_size = 768
        model.device = torch.device("cpu")
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        return tokenizer

    @pytest.fixture
    def sample_topic_data(self):
        """Sample topic training data."""
        return {
            "topic_id": "unit-01/chapter-01/topic-01",
            "sft_data": [
                {"question": "Q1?", "answer": "A1"},
                {"question": "Q2?", "answer": "A2"},
            ],
            "preference_pairs": [
                {"prompt": "Q1?", "chosen": "Good", "rejected": "Bad"},
            ],
            "prompts": ["Q1?", "Q2?"],
        }

    @patch("src.training.ppo_pipeline.SFTTrainer")
    @patch("src.training.ppo_pipeline.RewardModelTrainer")
    @patch("src.training.ppo_pipeline.PPOTrainer")
    def test_train_topic_runs_sft_first(
        self,
        mock_ppo_cls: MagicMock,
        mock_reward_cls: MagicMock,
        mock_sft_cls: MagicMock,
        mock_model,
        mock_tokenizer,
        sample_topic_data,
    ) -> None:
        """train_topic runs SFT training first."""
        # Setup mock SFT
        mock_sft = MagicMock()
        mock_sft.train_on_topic.return_value = MagicMock(success=True)
        mock_sft_cls.return_value = mock_sft

        # Setup mock Reward
        mock_reward = MagicMock()
        mock_reward.train.return_value = MagicMock(success=True)
        mock_reward.get_reward_model.return_value = MagicMock()
        mock_reward_cls.return_value = mock_reward

        # Setup mock PPO
        mock_ppo = MagicMock()
        mock_ppo.train_on_topic.return_value = MagicMock(
            success=True, final_reward_mean=0.85
        )
        mock_ppo_cls.return_value = mock_ppo

        config = PipelineConfig()
        pipeline = PPOPipeline(mock_model, mock_tokenizer, config)
        pipeline.train_topic(sample_topic_data)

        # Verify SFT was called first
        mock_sft.train_on_topic.assert_called()

    @patch("src.training.ppo_pipeline.SFTTrainer")
    @patch("src.training.ppo_pipeline.RewardModelTrainer")
    @patch("src.training.ppo_pipeline.PPOTrainer")
    def test_train_topic_trains_reward_model(
        self,
        mock_ppo_cls: MagicMock,
        mock_reward_cls: MagicMock,
        mock_sft_cls: MagicMock,
        mock_model,
        mock_tokenizer,
        sample_topic_data,
    ) -> None:
        """train_topic trains reward model after SFT."""
        # Setup mocks
        mock_sft = MagicMock()
        mock_sft.train_on_topic.return_value = MagicMock(success=True)
        mock_sft_cls.return_value = mock_sft

        mock_reward = MagicMock()
        mock_reward.train.return_value = MagicMock(success=True)
        mock_reward.get_reward_model.return_value = MagicMock()
        mock_reward_cls.return_value = mock_reward

        mock_ppo = MagicMock()
        mock_ppo.train_on_topic.return_value = MagicMock(
            success=True, final_reward_mean=0.85
        )
        mock_ppo_cls.return_value = mock_ppo

        config = PipelineConfig()
        pipeline = PPOPipeline(mock_model, mock_tokenizer, config)
        pipeline.train_topic(sample_topic_data)

        # Verify reward model was trained
        mock_reward.train.assert_called()

    @patch("src.training.ppo_pipeline.SFTTrainer")
    @patch("src.training.ppo_pipeline.RewardModelTrainer")
    @patch("src.training.ppo_pipeline.PPOTrainer")
    def test_train_topic_runs_ppo(
        self,
        mock_ppo_cls: MagicMock,
        mock_reward_cls: MagicMock,
        mock_sft_cls: MagicMock,
        mock_model,
        mock_tokenizer,
        sample_topic_data,
    ) -> None:
        """train_topic runs PPO after reward model."""
        # Setup mocks
        mock_sft = MagicMock()
        mock_sft.train_on_topic.return_value = MagicMock(success=True)
        mock_sft_cls.return_value = mock_sft

        mock_reward = MagicMock()
        mock_reward.train.return_value = MagicMock(success=True)
        mock_reward.get_reward_model.return_value = MagicMock()
        mock_reward_cls.return_value = mock_reward

        mock_ppo = MagicMock()
        mock_ppo.train_on_topic.return_value = MagicMock(
            success=True, final_reward_mean=0.85
        )
        mock_ppo_cls.return_value = mock_ppo

        config = PipelineConfig()
        pipeline = PPOPipeline(mock_model, mock_tokenizer, config)
        pipeline.train_topic(sample_topic_data)

        # Verify PPO was called
        mock_ppo.train_on_topic.assert_called()


class TestCheckpointSaveResume:
    """Test checkpoint saving and resuming."""

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
        model.config.hidden_size = 768
        model.device = torch.device("cpu")
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        return tokenizer

    def test_save_checkpoint_creates_file(
        self, mock_model, mock_tokenizer, temp_dir
    ) -> None:
        """save_checkpoint creates checkpoint file."""
        config = PipelineConfig(output_dir=temp_dir)
        pipeline = PPOPipeline(mock_model, mock_tokenizer, config)

        progress = {
            "current_unit": "unit-01",
            "completed_topics": ["t1", "t2"],
        }

        checkpoint_path = pipeline.save_checkpoint(progress)
        assert checkpoint_path.exists()

    def test_resume_loads_checkpoint(
        self, mock_model, mock_tokenizer, temp_dir
    ) -> None:
        """resume_from_checkpoint loads saved state."""
        config = PipelineConfig(output_dir=temp_dir)
        pipeline = PPOPipeline(mock_model, mock_tokenizer, config)

        progress = {
            "current_unit": "unit-03",
            "completed_topics": ["t1", "t2", "t3", "t4"],
        }
        checkpoint_path = pipeline.save_checkpoint(progress)

        pipeline2 = PPOPipeline(mock_model, mock_tokenizer, config)
        loaded_progress = pipeline2.resume_from_checkpoint(checkpoint_path)

        assert loaded_progress["current_unit"] == "unit-03"


class TestDataLoading:
    """Test data loading from prepared files."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory with sample data."""
        temp = tempfile.mkdtemp()
        data_dir = Path(temp) / "data"
        data_dir.mkdir()

        # Create sample SFT data
        sft_data = [
            {"topic_id": "t1", "question": "Q1", "answer": "A1"},
        ]
        with open(data_dir / "sft_data.jsonl", "w") as f:
            for item in sft_data:
                f.write(json.dumps(item) + "\n")

        # Create sample preference data
        pref_data = [
            {"topic_id": "t1", "prompt": "Q1", "chosen": "Good", "rejected": "Bad"},
        ]
        with open(data_dir / "preference_data.jsonl", "w") as f:
            for item in pref_data:
                f.write(json.dumps(item) + "\n")

        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = MagicMock()
        model.config = MagicMock()
        model.config.hidden_size = 768
        model.device = torch.device("cpu")
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        return tokenizer

    def test_load_sft_data(self, mock_model, mock_tokenizer, temp_dir) -> None:
        """Pipeline loads SFT data from file."""
        config = PipelineConfig(data_dir=str(Path(temp_dir) / "data"))
        pipeline = PPOPipeline(mock_model, mock_tokenizer, config)

        sft_data = pipeline._load_sft_data()
        assert len(sft_data) == 1

    def test_load_preference_data(
        self, mock_model, mock_tokenizer, temp_dir
    ) -> None:
        """Pipeline loads preference data from file."""
        config = PipelineConfig(data_dir=str(Path(temp_dir) / "data"))
        pipeline = PPOPipeline(mock_model, mock_tokenizer, config)

        pref_data = pipeline._load_preference_data()
        assert len(pref_data) == 1
