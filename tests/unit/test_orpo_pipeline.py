"""
Tests for ORPOPipeline - ORPO Training Orchestration.

Tests cover:
- Pipeline configuration
- Topic/Unit/Chapter progress tracking
- ORPO training flow
- Checkpoint save/resume
- Data grouping and evaluation
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import shutil
import json

torch = pytest.importorskip("torch")

from src.training.orpo_pipeline import (
    TopicStatus,
    PipelineConfig,
    TopicProgress,
    ChapterProgress,
    UnitProgress,
    PipelineResult,
    ORPOPipeline,
)


class TestTopicStatus:
    """Test TopicStatus enum."""

    def test_pending_status_exists(self) -> None:
        assert TopicStatus.PENDING is not None

    def test_orpo_complete_status_exists(self) -> None:
        assert TopicStatus.ORPO_COMPLETE is not None

    def test_eval_passed_status_exists(self) -> None:
        assert TopicStatus.EVAL_PASSED is not None

    def test_passed_status_exists(self) -> None:
        assert TopicStatus.PASSED is not None

    def test_failed_status_exists(self) -> None:
        assert TopicStatus.FAILED is not None


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""

    def test_default_orpo_steps(self) -> None:
        config = PipelineConfig()
        assert config.orpo_steps_per_topic == 100

    def test_default_orpo_learning_rate(self) -> None:
        config = PipelineConfig()
        assert config.orpo_learning_rate == 3e-4

    def test_default_orpo_lambda(self) -> None:
        config = PipelineConfig()
        assert config.orpo_lambda == 0.1

    def test_default_accuracy_threshold(self) -> None:
        config = PipelineConfig()
        assert config.accuracy_threshold == 0.70

    def test_default_topic_pass_threshold(self) -> None:
        config = PipelineConfig()
        assert config.topic_pass_threshold == 0.90

    def test_default_merge_after_unit(self) -> None:
        config = PipelineConfig()
        assert config.merge_after_unit is True

    def test_default_max_retries(self) -> None:
        config = PipelineConfig()
        assert config.max_retries_per_topic == 2

    def test_custom_config(self) -> None:
        config = PipelineConfig(
            orpo_steps_per_topic=200,
            orpo_learning_rate=1e-4,
            orpo_lambda=0.2,
            max_retries_per_topic=3,
        )
        assert config.orpo_steps_per_topic == 200
        assert config.orpo_learning_rate == 1e-4
        assert config.orpo_lambda == 0.2
        assert config.max_retries_per_topic == 3


class TestTopicProgress:
    """Test TopicProgress dataclass."""

    def test_stores_topic_id(self) -> None:
        progress = TopicProgress(
            topic_id="unit-01/chapter-01/topic-01",
            status=TopicStatus.PENDING,
        )
        assert progress.topic_id == "unit-01/chapter-01/topic-01"

    def test_stores_status(self) -> None:
        progress = TopicProgress(
            topic_id="test",
            status=TopicStatus.ORPO_COMPLETE,
        )
        assert progress.status == TopicStatus.ORPO_COMPLETE

    def test_stores_accuracy_score(self) -> None:
        progress = TopicProgress(
            topic_id="test",
            status=TopicStatus.EVAL_PASSED,
            accuracy_score=0.85,
        )
        assert progress.accuracy_score == 0.85

    def test_stores_voice_score(self) -> None:
        progress = TopicProgress(
            topic_id="test",
            status=TopicStatus.EVAL_PASSED,
            voice_score=0.72,
        )
        assert progress.voice_score == 0.72

    def test_stores_retry_count(self) -> None:
        progress = TopicProgress(
            topic_id="test",
            status=TopicStatus.FAILED,
            retry_count=1,
        )
        assert progress.retry_count == 1


class TestChapterProgress:
    """Test ChapterProgress dataclass."""

    def test_stores_chapter_id(self) -> None:
        progress = ChapterProgress(chapter_id="unit-01/chapter-01", topics=[])
        assert progress.chapter_id == "unit-01/chapter-01"

    def test_passed_topics_count(self) -> None:
        topics = [
            TopicProgress("t1", TopicStatus.PASSED),
            TopicProgress("t2", TopicStatus.PASSED),
            TopicProgress("t3", TopicStatus.FAILED),
        ]
        progress = ChapterProgress(chapter_id="chapter-01", topics=topics)
        assert progress.passed_count == 2

    def test_total_topics_count(self) -> None:
        topics = [
            TopicProgress("t1", TopicStatus.PASSED),
            TopicProgress("t2", TopicStatus.FAILED),
        ]
        progress = ChapterProgress(chapter_id="chapter-01", topics=topics)
        assert progress.total_count == 2


class TestUnitProgress:
    """Test UnitProgress dataclass."""

    def test_stores_unit_id(self) -> None:
        progress = UnitProgress(unit_id="unit-01", chapters=[])
        assert progress.unit_id == "unit-01"

    def test_merged_status(self) -> None:
        progress = UnitProgress(unit_id="unit-01", chapters=[], merged=True)
        assert progress.merged is True


class TestPipelineResult:
    """Test PipelineResult dataclass."""

    def test_stores_success_status(self) -> None:
        result = PipelineResult(
            success=True, total_topics=52, passed_topics=50,
            failed_topics=["t1", "t2"], total_training_time_hours=40.5,
        )
        assert result.success is True

    def test_stores_topic_counts(self) -> None:
        result = PipelineResult(
            success=True, total_topics=52, passed_topics=50,
            failed_topics=["t1", "t2"], total_training_time_hours=40.5,
        )
        assert result.total_topics == 52
        assert result.passed_topics == 50
        assert len(result.failed_topics) == 2


class TestORPOPipeline:
    """Test ORPOPipeline class."""

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.config = MagicMock()
        model.device = torch.device("cpu")
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "</s>"
        return tokenizer

    def test_initializes_with_config(self, mock_model, mock_tokenizer) -> None:
        config = PipelineConfig()
        pipeline = ORPOPipeline(mock_model, mock_tokenizer, config)
        assert pipeline.config == config

    def test_has_train_curriculum_method(self, mock_model, mock_tokenizer) -> None:
        pipeline = ORPOPipeline(mock_model, mock_tokenizer, PipelineConfig())
        assert callable(pipeline.train_curriculum)

    def test_has_train_topic_method(self, mock_model, mock_tokenizer) -> None:
        pipeline = ORPOPipeline(mock_model, mock_tokenizer, PipelineConfig())
        assert callable(pipeline.train_topic)

    def test_has_save_checkpoint_method(self, mock_model, mock_tokenizer) -> None:
        pipeline = ORPOPipeline(mock_model, mock_tokenizer, PipelineConfig())
        assert callable(pipeline.save_checkpoint)


class TestTopicIdExtraction:
    """Test _extract_topic_id logic."""

    def test_extracts_from_source_field(self) -> None:
        item = {"source": "unit-01/chapter-01/topic-01", "prompt": "test"}
        assert ORPOPipeline._extract_topic_id(item) == "unit-01/chapter-01/topic-01"

    def test_extracts_from_topic_id_field(self) -> None:
        item = {"topic_id": "topic-42", "prompt": "test"}
        assert ORPOPipeline._extract_topic_id(item) == "topic-42"

    def test_extracts_date_from_prompt(self) -> None:
        item = {"prompt": "What did Bob discuss on 2019-02-19?"}
        assert ORPOPipeline._extract_topic_id(item) == "episode-2019-02-19"

    def test_falls_back_to_general(self) -> None:
        item = {"prompt": "What is accumulation?"}
        assert ORPOPipeline._extract_topic_id(item) == "general"

    def test_source_takes_priority_over_date(self) -> None:
        item = {"source": "my-topic", "prompt": "What about 2019-02-19?"}
        assert ORPOPipeline._extract_topic_id(item) == "my-topic"


class TestCheckpointSaveResume:
    """Test checkpoint saving and resuming."""

    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.device = torch.device("cpu")
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        return tokenizer

    def test_save_checkpoint_creates_file(self, mock_model, mock_tokenizer, temp_dir) -> None:
        config = PipelineConfig(output_dir=temp_dir)
        pipeline = ORPOPipeline(mock_model, mock_tokenizer, config)
        progress = {"current_unit": "unit-01", "completed_topics": ["t1", "t2"]}
        checkpoint_path = pipeline.save_checkpoint(progress)
        assert checkpoint_path.exists()

    def test_resume_loads_checkpoint(self, mock_model, mock_tokenizer, temp_dir) -> None:
        config = PipelineConfig(output_dir=temp_dir)
        pipeline = ORPOPipeline(mock_model, mock_tokenizer, config)
        progress = {"current_unit": "unit-03", "completed_topics": ["t1", "t2", "t3", "t4"]}
        checkpoint_path = pipeline.save_checkpoint(progress)

        pipeline2 = ORPOPipeline(mock_model, mock_tokenizer, config)
        loaded = pipeline2.resume_from_checkpoint(checkpoint_path)
        assert loaded["current_unit"] == "unit-03"
        assert len(loaded["completed_topics"]) == 4


class TestRetryLogic:
    """Test retry logic for failed topics."""

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.device = torch.device("cpu")
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        return tokenizer

    def test_max_retries_configurable(self, mock_model, mock_tokenizer) -> None:
        config = PipelineConfig(max_retries_per_topic=5)
        pipeline = ORPOPipeline(mock_model, mock_tokenizer, config)
        assert pipeline.config.max_retries_per_topic == 5

    def test_topic_marked_failed_after_max_retries(self, mock_model, mock_tokenizer) -> None:
        config = PipelineConfig(max_retries_per_topic=2)
        pipeline = ORPOPipeline(mock_model, mock_tokenizer, config)
        progress = TopicProgress(topic_id="test", status=TopicStatus.FAILED, retry_count=2)
        assert pipeline._should_mark_failed(progress) is True


class TestDataLoading:
    """Test data loading from prepared files."""

    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        data_dir = Path(temp) / "data"
        data_dir.mkdir()
        pref_data = [
            {"prompt": "What did Bob discuss on 2019-02-19?", "chosen": "Good", "rejected": "Bad"},
            {"prompt": "What did Bob discuss on 2019-03-04?", "chosen": "Great", "rejected": "Poor"},
        ]
        with open(data_dir / "preference_data.jsonl", "w") as f:
            for item in pref_data:
                f.write(json.dumps(item) + "\n")
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.device = torch.device("cpu")
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        return tokenizer

    def test_load_preference_data(self, mock_model, mock_tokenizer, temp_dir) -> None:
        config = PipelineConfig(data_dir=str(Path(temp_dir) / "data"))
        pipeline = ORPOPipeline(mock_model, mock_tokenizer, config)
        pref_data = pipeline._load_preference_data()
        assert len(pref_data) == 2
        assert pref_data[0]["chosen"] == "Good"

    def test_groups_data_by_topic(self, mock_model, mock_tokenizer, temp_dir) -> None:
        config = PipelineConfig(data_dir=str(Path(temp_dir) / "data"))
        pipeline = ORPOPipeline(mock_model, mock_tokenizer, config)
        grouped = pipeline._group_data_by_topic()
        assert "episode-2019-02-19" in grouped
        assert "episode-2019-03-04" in grouped
        assert len(grouped["episode-2019-02-19"]["preference_pairs"]) == 1


class TestEvaluationIntegration:
    """Test integration with evaluation pipeline."""

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.device = torch.device("cpu")
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        return tokenizer

    def test_has_evaluate_topic_method(self, mock_model, mock_tokenizer) -> None:
        pipeline = ORPOPipeline(mock_model, mock_tokenizer, PipelineConfig())
        assert callable(pipeline._evaluate_topic)

    @patch("src.training.orpo_pipeline.QuizEvaluator")
    def test_evaluate_topic_returns_scores(
        self, mock_evaluator_cls: MagicMock, mock_model, mock_tokenizer
    ) -> None:
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = {
            "accuracy": 0.85,
            "voice_score": 0.72,
            "passed": True,
        }
        mock_evaluator_cls.return_value = mock_evaluator

        pipeline = ORPOPipeline(mock_model, mock_tokenizer, PipelineConfig())
        topic_data = {
            "topic_id": "t1",
            "questions": [{"question": "Q?", "reference_answer": "A"}],
        }
        result = pipeline._evaluate_topic(topic_data)
        assert "accuracy" in result
        assert "voice_score" in result
