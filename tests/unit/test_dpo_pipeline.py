"""
Tests for DPOPipeline - SFT + DPO Training Orchestration.

Tests cover:
- Pipeline configuration
- Topic/Unit/Chapter progress tracking
- SFT → Eval → DPO decision logic
- Checkpoint save/resume
- Unit merge triggers
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path
import tempfile
import shutil
import json
from enum import Enum
from typing import Dict, List

import torch

from src.training.dpo_pipeline import (
    TopicStatus,
    PipelineConfig,
    TopicProgress,
    ChapterProgress,
    UnitProgress,
    PipelineResult,
    DPOPipeline,
)


class TestTopicStatus:
    """Test TopicStatus enum."""

    def test_pending_status_exists(self) -> None:
        """TopicStatus has PENDING value."""
        assert TopicStatus.PENDING is not None

    def test_sft_complete_status_exists(self) -> None:
        """TopicStatus has SFT_COMPLETE value."""
        assert TopicStatus.SFT_COMPLETE is not None

    def test_eval_passed_status_exists(self) -> None:
        """TopicStatus has EVAL_PASSED value."""
        assert TopicStatus.EVAL_PASSED is not None

    def test_dpo_needed_status_exists(self) -> None:
        """TopicStatus has DPO_NEEDED value."""
        assert TopicStatus.DPO_NEEDED is not None

    def test_dpo_complete_status_exists(self) -> None:
        """TopicStatus has DPO_COMPLETE value."""
        assert TopicStatus.DPO_COMPLETE is not None

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

    def test_default_dpo_steps(self) -> None:
        """Default DPO steps is 100."""
        config = PipelineConfig()
        assert config.dpo_steps_per_topic == 100

    def test_default_dpo_trigger_threshold(self) -> None:
        """Default DPO trigger threshold is 0.75."""
        config = PipelineConfig()
        assert config.dpo_trigger_threshold == 0.75

    def test_default_accuracy_threshold(self) -> None:
        """Default accuracy threshold is 0.70."""
        config = PipelineConfig()
        assert config.accuracy_threshold == 0.70

    def test_default_topic_pass_threshold(self) -> None:
        """Default topic pass threshold is 0.90."""
        config = PipelineConfig()
        assert config.topic_pass_threshold == 0.90

    def test_default_merge_after_unit(self) -> None:
        """Default merge_after_unit is True."""
        config = PipelineConfig()
        assert config.merge_after_unit is True

    def test_default_max_retries(self) -> None:
        """Default max retries is 2."""
        config = PipelineConfig()
        assert config.max_retries_per_topic == 2

    def test_custom_config(self) -> None:
        """Custom config values are set correctly."""
        config = PipelineConfig(
            sft_epochs_per_topic=5,
            dpo_steps_per_topic=200,
            dpo_trigger_threshold=0.80,
            max_retries_per_topic=3,
        )
        assert config.sft_epochs_per_topic == 5
        assert config.dpo_steps_per_topic == 200
        assert config.dpo_trigger_threshold == 0.80
        assert config.max_retries_per_topic == 3


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

    def test_stores_accuracy_score(self) -> None:
        """Progress stores accuracy score."""
        progress = TopicProgress(
            topic_id="test",
            status=TopicStatus.EVAL_PASSED,
            accuracy_score=0.85,
        )
        assert progress.accuracy_score == 0.85

    def test_stores_voice_score(self) -> None:
        """Progress stores voice score."""
        progress = TopicProgress(
            topic_id="test",
            status=TopicStatus.EVAL_PASSED,
            voice_score=0.72,
        )
        assert progress.voice_score == 0.72

    def test_stores_retry_count(self) -> None:
        """Progress stores retry count."""
        progress = TopicProgress(
            topic_id="test",
            status=TopicStatus.DPO_NEEDED,
            retry_count=1,
        )
        assert progress.retry_count == 1


class TestChapterProgress:
    """Test ChapterProgress dataclass."""

    def test_stores_chapter_id(self) -> None:
        """ChapterProgress stores chapter ID."""
        progress = ChapterProgress(
            chapter_id="unit-01/chapter-01",
            topics=[],
        )
        assert progress.chapter_id == "unit-01/chapter-01"

    def test_stores_topic_list(self) -> None:
        """ChapterProgress stores list of topics."""
        topics = [
            TopicProgress("t1", TopicStatus.PASSED),
            TopicProgress("t2", TopicStatus.PENDING),
        ]
        progress = ChapterProgress(
            chapter_id="chapter-01",
            topics=topics,
        )
        assert len(progress.topics) == 2

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

    def test_total_topics_count(self) -> None:
        """ChapterProgress counts total topics."""
        topics = [
            TopicProgress("t1", TopicStatus.PASSED),
            TopicProgress("t2", TopicStatus.FAILED),
        ]
        progress = ChapterProgress(
            chapter_id="chapter-01",
            topics=topics,
        )
        assert progress.total_count == 2


class TestUnitProgress:
    """Test UnitProgress dataclass."""

    def test_stores_unit_id(self) -> None:
        """UnitProgress stores unit ID."""
        progress = UnitProgress(
            unit_id="unit-01",
            chapters=[],
        )
        assert progress.unit_id == "unit-01"

    def test_stores_chapter_list(self) -> None:
        """UnitProgress stores list of chapters."""
        chapters = [
            ChapterProgress("c1", []),
            ChapterProgress("c2", []),
        ]
        progress = UnitProgress(
            unit_id="unit-01",
            chapters=chapters,
        )
        assert len(progress.chapters) == 2

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
            total_training_time_hours=40.5,
        )
        assert result.success is True

    def test_stores_topic_counts(self) -> None:
        """Result stores topic counts."""
        result = PipelineResult(
            success=True,
            total_topics=52,
            passed_topics=50,
            failed_topics=["t1", "t2"],
            total_training_time_hours=40.5,
        )
        assert result.total_topics == 52
        assert result.passed_topics == 50

    def test_stores_failed_topics_list(self) -> None:
        """Result stores list of failed topic IDs."""
        result = PipelineResult(
            success=True,
            total_topics=52,
            passed_topics=50,
            failed_topics=["unit-01/ch-01/t-01", "unit-02/ch-01/t-03"],
            total_training_time_hours=40.5,
        )
        assert len(result.failed_topics) == 2

    def test_stores_training_time(self) -> None:
        """Result stores training time in hours."""
        result = PipelineResult(
            success=True,
            total_topics=10,
            passed_topics=10,
            failed_topics=[],
            total_training_time_hours=5.5,
        )
        assert result.total_training_time_hours == 5.5


class TestDPOPipeline:
    """Test DPOPipeline class."""

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
    def sample_sft_data(self):
        """Sample SFT training data."""
        return [
            {
                "topic_id": "unit-01/chapter-01/topic-01",
                "question": "What drives the 4-year cycle?",
                "answer": "Market psychology, not the halving.",
            },
            {
                "topic_id": "unit-01/chapter-01/topic-01",
                "question": "What is accumulation?",
                "answer": "When smart money builds positions quietly.",
            },
        ]

    @pytest.fixture
    def sample_preference_data(self):
        """Sample preference pairs."""
        return [
            {
                "topic_id": "unit-01/chapter-01/topic-01",
                "prompt": "What drives the cycle?",
                "chosen": "Psychology drives cycles, with halving as a catalyst.",
                "rejected": "The halving causes the 4-year cycle.",
            },
        ]

    def test_initializes_with_config(self, mock_model, mock_tokenizer) -> None:
        """Pipeline initializes with configuration."""
        config = PipelineConfig()
        pipeline = DPOPipeline(mock_model, mock_tokenizer, config)
        assert pipeline.config == config

    def test_has_train_curriculum_method(self, mock_model, mock_tokenizer) -> None:
        """Pipeline has train_curriculum method."""
        pipeline = DPOPipeline(mock_model, mock_tokenizer, PipelineConfig())
        assert hasattr(pipeline, "train_curriculum")
        assert callable(pipeline.train_curriculum)

    def test_has_train_unit_method(self, mock_model, mock_tokenizer) -> None:
        """Pipeline has train_unit method."""
        pipeline = DPOPipeline(mock_model, mock_tokenizer, PipelineConfig())
        assert hasattr(pipeline, "train_unit")
        assert callable(pipeline.train_unit)

    def test_has_train_topic_method(self, mock_model, mock_tokenizer) -> None:
        """Pipeline has train_topic method."""
        pipeline = DPOPipeline(mock_model, mock_tokenizer, PipelineConfig())
        assert hasattr(pipeline, "train_topic")
        assert callable(pipeline.train_topic)

    def test_has_save_checkpoint_method(self, mock_model, mock_tokenizer) -> None:
        """Pipeline has save_checkpoint method."""
        pipeline = DPOPipeline(mock_model, mock_tokenizer, PipelineConfig())
        assert hasattr(pipeline, "save_checkpoint")
        assert callable(pipeline.save_checkpoint)

    def test_has_resume_from_checkpoint_method(
        self, mock_model, mock_tokenizer
    ) -> None:
        """Pipeline has resume_from_checkpoint method."""
        pipeline = DPOPipeline(mock_model, mock_tokenizer, PipelineConfig())
        assert hasattr(pipeline, "resume_from_checkpoint")
        assert callable(pipeline.resume_from_checkpoint)


class TestDPOTriggerLogic:
    """Test DPO trigger decision logic."""

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

    def test_should_run_dpo_when_accuracy_high_voice_low(
        self, mock_model, mock_tokenizer
    ) -> None:
        """DPO runs when accuracy >= 0.70 and voice < 0.75."""
        config = PipelineConfig(
            accuracy_threshold=0.70,
            dpo_trigger_threshold=0.75,
        )
        pipeline = DPOPipeline(mock_model, mock_tokenizer, config)

        # accuracy = 0.85, voice = 0.60 → should trigger DPO
        eval_result = {"accuracy": 0.85, "voice_score": 0.60}
        assert pipeline._should_run_dpo(eval_result) is True

    def test_should_not_run_dpo_when_voice_high(
        self, mock_model, mock_tokenizer
    ) -> None:
        """DPO not needed when voice >= 0.75."""
        config = PipelineConfig(
            accuracy_threshold=0.70,
            dpo_trigger_threshold=0.75,
        )
        pipeline = DPOPipeline(mock_model, mock_tokenizer, config)

        # accuracy = 0.90, voice = 0.80 → no DPO needed
        eval_result = {"accuracy": 0.90, "voice_score": 0.80}
        assert pipeline._should_run_dpo(eval_result) is False

    def test_should_not_run_dpo_when_accuracy_low(
        self, mock_model, mock_tokenizer
    ) -> None:
        """DPO not run when accuracy < 0.70 (needs more SFT first)."""
        config = PipelineConfig(
            accuracy_threshold=0.70,
            dpo_trigger_threshold=0.75,
        )
        pipeline = DPOPipeline(mock_model, mock_tokenizer, config)

        # accuracy = 0.50, voice = 0.40 → needs more SFT, not DPO
        eval_result = {"accuracy": 0.50, "voice_score": 0.40}
        assert pipeline._should_run_dpo(eval_result) is False

    def test_dpo_threshold_is_configurable(
        self, mock_model, mock_tokenizer
    ) -> None:
        """DPO trigger threshold can be customized."""
        config = PipelineConfig(
            accuracy_threshold=0.70,
            dpo_trigger_threshold=0.80,  # Higher threshold
        )
        pipeline = DPOPipeline(mock_model, mock_tokenizer, config)

        # voice = 0.75 is now below threshold (0.80), should trigger
        eval_result = {"accuracy": 0.85, "voice_score": 0.75}
        assert pipeline._should_run_dpo(eval_result) is True


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
        pipeline = DPOPipeline(mock_model, mock_tokenizer, config)

        # Create some progress
        progress = {
            "current_unit": "unit-01",
            "completed_topics": ["t1", "t2"],
        }

        checkpoint_path = pipeline.save_checkpoint(progress)
        assert checkpoint_path.exists()

    def test_checkpoint_contains_progress(
        self, mock_model, mock_tokenizer, temp_dir
    ) -> None:
        """Checkpoint file contains progress data."""
        config = PipelineConfig(output_dir=temp_dir)
        pipeline = DPOPipeline(mock_model, mock_tokenizer, config)

        progress = {
            "current_unit": "unit-02",
            "completed_topics": ["t1", "t2", "t3"],
        }

        checkpoint_path = pipeline.save_checkpoint(progress)

        # Load and verify
        with open(checkpoint_path) as f:
            loaded = json.load(f)
        assert loaded["current_unit"] == "unit-02"
        assert len(loaded["completed_topics"]) == 3

    def test_resume_loads_checkpoint(
        self, mock_model, mock_tokenizer, temp_dir
    ) -> None:
        """resume_from_checkpoint loads saved state."""
        config = PipelineConfig(output_dir=temp_dir)
        pipeline = DPOPipeline(mock_model, mock_tokenizer, config)

        # Save checkpoint
        progress = {
            "current_unit": "unit-03",
            "completed_topics": ["t1", "t2", "t3", "t4"],
        }
        checkpoint_path = pipeline.save_checkpoint(progress)

        # Create new pipeline and resume
        pipeline2 = DPOPipeline(mock_model, mock_tokenizer, config)
        loaded_progress = pipeline2.resume_from_checkpoint(checkpoint_path)

        assert loaded_progress["current_unit"] == "unit-03"
        assert len(loaded_progress["completed_topics"]) == 4


class TestUnitMerge:
    """Test unit merge after completion."""

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

    def test_has_merge_unit_adapters_method(
        self, mock_model, mock_tokenizer
    ) -> None:
        """Pipeline has _merge_unit_adapters method."""
        pipeline = DPOPipeline(mock_model, mock_tokenizer, PipelineConfig())
        assert hasattr(pipeline, "_merge_unit_adapters")
        assert callable(pipeline._merge_unit_adapters)

    def test_merge_enabled_by_default(self, mock_model, mock_tokenizer) -> None:
        """Unit merge is enabled by default."""
        config = PipelineConfig()
        assert config.merge_after_unit is True

    def test_merge_can_be_disabled(self, mock_model, mock_tokenizer) -> None:
        """Unit merge can be disabled."""
        config = PipelineConfig(merge_after_unit=False)
        pipeline = DPOPipeline(mock_model, mock_tokenizer, config)
        assert pipeline.config.merge_after_unit is False


class TestTopicTrainingFlow:
    """Test the complete topic training flow."""

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
        }

    @patch("src.training.dpo_pipeline.SFTTrainer")
    @patch("src.training.dpo_pipeline.Rank1DPOTrainer")
    def test_train_topic_runs_sft_first(
        self,
        mock_dpo_cls: MagicMock,
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

        # Setup mock DPO
        mock_dpo = MagicMock()
        mock_dpo_cls.return_value = mock_dpo

        config = PipelineConfig()
        pipeline = DPOPipeline(mock_model, mock_tokenizer, config)

        # Mock evaluation to avoid DPO
        with patch.object(
            pipeline, "_evaluate_topic", return_value={"accuracy": 0.95, "voice_score": 0.85}
        ):
            pipeline.train_topic(sample_topic_data)

        # Verify SFT was called
        mock_sft.train_on_topic.assert_called()

    @patch("src.training.dpo_pipeline.SFTTrainer")
    @patch("src.training.dpo_pipeline.Rank1DPOTrainer")
    def test_train_topic_runs_dpo_when_needed(
        self,
        mock_dpo_cls: MagicMock,
        mock_sft_cls: MagicMock,
        mock_model,
        mock_tokenizer,
        sample_topic_data,
    ) -> None:
        """train_topic runs DPO when voice is low."""
        # Setup mock SFT
        mock_sft = MagicMock()
        mock_sft.train_on_topic.return_value = MagicMock(success=True)
        mock_sft_cls.return_value = mock_sft

        # Setup mock DPO
        mock_dpo = MagicMock()
        mock_dpo.train_on_topic.return_value = MagicMock(success=True)
        mock_dpo_cls.return_value = mock_dpo

        config = PipelineConfig()
        pipeline = DPOPipeline(mock_model, mock_tokenizer, config)

        # Mock evaluation to trigger DPO (high accuracy, low voice)
        with patch.object(
            pipeline, "_evaluate_topic", return_value={"accuracy": 0.85, "voice_score": 0.60}
        ):
            pipeline.train_topic(sample_topic_data)

        # Verify DPO was called
        mock_dpo.train_on_topic.assert_called()

    @patch("src.training.dpo_pipeline.SFTTrainer")
    @patch("src.training.dpo_pipeline.Rank1DPOTrainer")
    def test_train_topic_returns_topic_progress(
        self,
        mock_dpo_cls: MagicMock,
        mock_sft_cls: MagicMock,
        mock_model,
        mock_tokenizer,
        sample_topic_data,
    ) -> None:
        """train_topic returns TopicProgress."""
        # Setup mocks
        mock_sft = MagicMock()
        mock_sft.train_on_topic.return_value = MagicMock(success=True)
        mock_sft_cls.return_value = mock_sft

        mock_dpo = MagicMock()
        mock_dpo_cls.return_value = mock_dpo

        config = PipelineConfig()
        pipeline = DPOPipeline(mock_model, mock_tokenizer, config)

        with patch.object(
            pipeline, "_evaluate_topic", return_value={"accuracy": 0.95, "voice_score": 0.85}
        ):
            result = pipeline.train_topic(sample_topic_data)

        assert isinstance(result, TopicProgress)


class TestRetryLogic:
    """Test retry logic for failed topics."""

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

    def test_max_retries_is_configurable(self, mock_model, mock_tokenizer) -> None:
        """Max retries can be configured."""
        config = PipelineConfig(max_retries_per_topic=5)
        pipeline = DPOPipeline(mock_model, mock_tokenizer, config)
        assert pipeline.config.max_retries_per_topic == 5

    def test_topic_marked_failed_after_max_retries(
        self, mock_model, mock_tokenizer
    ) -> None:
        """Topic is marked FAILED after max retries exceeded."""
        config = PipelineConfig(max_retries_per_topic=2)
        pipeline = DPOPipeline(mock_model, mock_tokenizer, config)

        # Topic that keeps failing
        progress = TopicProgress(
            topic_id="test-topic",
            status=TopicStatus.DPO_NEEDED,
            retry_count=2,  # At max retries
        )

        # After another failure, should be marked FAILED
        assert pipeline._should_mark_failed(progress) is True


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
            {"topic_id": "t2", "question": "Q2", "answer": "A2"},
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
        pipeline = DPOPipeline(mock_model, mock_tokenizer, config)

        sft_data = pipeline._load_sft_data()
        assert len(sft_data) == 2
        assert sft_data[0]["question"] == "Q1"

    def test_load_preference_data(
        self, mock_model, mock_tokenizer, temp_dir
    ) -> None:
        """Pipeline loads preference data from file."""
        config = PipelineConfig(data_dir=str(Path(temp_dir) / "data"))
        pipeline = DPOPipeline(mock_model, mock_tokenizer, config)

        pref_data = pipeline._load_preference_data()
        assert len(pref_data) == 1
        assert pref_data[0]["chosen"] == "Good"

    def test_groups_data_by_topic(
        self, mock_model, mock_tokenizer, temp_dir
    ) -> None:
        """Pipeline groups data by topic ID."""
        config = PipelineConfig(data_dir=str(Path(temp_dir) / "data"))
        pipeline = DPOPipeline(mock_model, mock_tokenizer, config)

        grouped = pipeline._group_data_by_topic()
        assert "t1" in grouped
        assert "t2" in grouped
        assert "sft_data" in grouped["t1"]
        assert "preference_pairs" in grouped["t1"]


class TestEvaluationIntegration:
    """Test integration with evaluation pipeline."""

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

    def test_has_evaluate_topic_method(self, mock_model, mock_tokenizer) -> None:
        """Pipeline has _evaluate_topic method."""
        pipeline = DPOPipeline(mock_model, mock_tokenizer, PipelineConfig())
        assert hasattr(pipeline, "_evaluate_topic")
        assert callable(pipeline._evaluate_topic)

    @patch("src.training.dpo_pipeline.QuizEvaluator")
    def test_evaluate_topic_returns_scores(
        self, mock_evaluator_cls: MagicMock, mock_model, mock_tokenizer
    ) -> None:
        """_evaluate_topic returns accuracy and voice scores."""
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = {
            "accuracy": 0.85,
            "voice_score": 0.72,
            "passed": True,
        }
        mock_evaluator_cls.return_value = mock_evaluator

        pipeline = DPOPipeline(mock_model, mock_tokenizer, PipelineConfig())

        topic_data = {
            "topic_id": "t1",
            "questions": [{"question": "Q?", "reference_answer": "A"}],
        }
        result = pipeline._evaluate_topic(topic_data)

        assert "accuracy" in result
        assert "voice_score" in result
