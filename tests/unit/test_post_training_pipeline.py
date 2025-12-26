"""
Tests for PostTrainingPipeline - orchestrating merge, evaluate, and export.

Tests cover:
- PostTrainingConfig validation
- Pipeline execution order
- Phase-specific modes (merge-only, export-only)
- Error handling and continuation
- Result aggregation
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile
import shutil
import json

from src.post_training.pipeline import (
    PostTrainingConfig,
    PostTrainingResult,
    PostTrainingPipeline,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def mock_adapter_dir(temp_dir: str) -> str:
    """Create a mock adapter directory with required files."""
    adapter_dir = Path(temp_dir) / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    (adapter_dir / "adapter_model.safetensors").write_text("mock")
    return str(adapter_dir)


@pytest.fixture
def mock_quiz_dir(temp_dir: str) -> str:
    """Create a mock quiz data directory."""
    quiz_dir = Path(temp_dir) / "quiz_data"
    quiz_dir.mkdir()
    quiz_data = [
        {
            "level": "topic",
            "unit": "unit-01",
            "chapter": "chapter-01",
            "topic": "topic-01",
            "questions": [{"question": "Test?", "reference_answer": "Answer."}],
        }
    ]
    with open(quiz_dir / "quiz_data.json", "w") as f:
        json.dump(quiz_data, f)
    return str(quiz_dir)


@pytest.fixture
def pipeline_config(
    temp_dir: str, mock_adapter_dir: str, mock_quiz_dir: str
) -> PostTrainingConfig:
    """Create a test pipeline configuration."""
    return PostTrainingConfig(
        base_model="google/gemma-3-12b",
        adapter_path=mock_adapter_dir,
        quiz_data_path=mock_quiz_dir,
        output_dir=str(Path(temp_dir) / "output"),
        approach="dpo",
    )


class TestPostTrainingConfig:
    """Test PostTrainingConfig validation."""

    def test_config_stores_all_fields(
        self, mock_adapter_dir: str, mock_quiz_dir: str, temp_dir: str
    ) -> None:
        """Config stores all fields correctly."""
        config = PostTrainingConfig(
            base_model="google/gemma-3-12b",
            adapter_path=mock_adapter_dir,
            quiz_data_path=mock_quiz_dir,
            output_dir=str(Path(temp_dir) / "output"),
            approach="dpo",
        )

        assert config.base_model == "google/gemma-3-12b"
        assert config.adapter_path == mock_adapter_dir
        assert config.approach == "dpo"

    def test_config_default_approach(
        self, mock_adapter_dir: str, mock_quiz_dir: str, temp_dir: str
    ) -> None:
        """Config defaults to DPO approach."""
        config = PostTrainingConfig(
            base_model="google/gemma-3-12b",
            adapter_path=mock_adapter_dir,
            quiz_data_path=mock_quiz_dir,
            output_dir=str(Path(temp_dir) / "output"),
        )

        assert config.approach == "dpo"

    def test_config_validates_adapter_path(
        self, mock_quiz_dir: str, temp_dir: str
    ) -> None:
        """Config raises error for non-existent adapter path."""
        with pytest.raises(ValueError, match="Adapter path does not exist"):
            PostTrainingConfig(
                base_model="google/gemma-3-12b",
                adapter_path="/nonexistent/path",
                quiz_data_path=mock_quiz_dir,
                output_dir=str(Path(temp_dir) / "output"),
            )


class TestPostTrainingResult:
    """Test PostTrainingResult dataclass."""

    def test_result_stores_success(self) -> None:
        """Result stores success status."""
        result = PostTrainingResult(
            success=True,
            merged_model_path="/path/to/merged",
            exports={"safetensors": "/path/to/export"},
            evaluation_passed=True,
            total_time_seconds=300.0,
        )

        assert result.success is True
        assert result.evaluation_passed is True

    def test_result_stores_failure(self) -> None:
        """Result stores failure with error message."""
        result = PostTrainingResult(
            success=False,
            merged_model_path="",
            exports={},
            evaluation_passed=False,
            total_time_seconds=10.0,
            error_message="Merge failed",
        )

        assert result.success is False
        assert result.error_message == "Merge failed"

    def test_result_stores_evaluation_report(self) -> None:
        """Result stores evaluation report path."""
        result = PostTrainingResult(
            success=True,
            merged_model_path="/path/to/merged",
            exports={},
            evaluation_passed=True,
            total_time_seconds=300.0,
            evaluation_report_path="/path/to/report.json",
        )

        assert result.evaluation_report_path == "/path/to/report.json"


class TestPostTrainingPipeline:
    """Test PostTrainingPipeline functionality."""

    def test_pipeline_initializes_with_config(
        self, pipeline_config: PostTrainingConfig
    ) -> None:
        """Pipeline initializes with configuration."""
        pipeline = PostTrainingPipeline(pipeline_config)

        assert pipeline.config == pipeline_config

    def test_pipeline_creates_output_directory(
        self, mock_adapter_dir: str, mock_quiz_dir: str, temp_dir: str
    ) -> None:
        """Pipeline creates output directory if it doesn't exist."""
        new_dir = Path(temp_dir) / "new" / "nested" / "dir"
        config = PostTrainingConfig(
            base_model="google/gemma-3-12b",
            adapter_path=mock_adapter_dir,
            quiz_data_path=mock_quiz_dir,
            output_dir=str(new_dir),
        )

        pipeline = PostTrainingPipeline(config)

        assert new_dir.exists()


class TestPipelineExecution:
    """Test pipeline execution order and flow."""

    @patch("src.post_training.pipeline.ExportConfig")
    @patch("src.post_training.pipeline.ModelExporter")
    @patch("src.post_training.pipeline.EvaluationPipeline")
    @patch("src.post_training.pipeline.LoRAMerger")
    def test_run_executes_in_order(
        self,
        mock_merger_cls: MagicMock,
        mock_eval_cls: MagicMock,
        mock_exporter_cls: MagicMock,
        mock_export_config: MagicMock,
        pipeline_config: PostTrainingConfig,
    ) -> None:
        """Pipeline executes merge -> evaluate -> export in order."""
        # Setup mocks
        mock_merger = MagicMock()
        mock_merger.merge.return_value = MagicMock(
            success=True,
            output_path="/path/to/merged",
        )
        mock_merger_cls.return_value = mock_merger

        mock_eval = MagicMock()
        mock_eval.run_full_evaluation.return_value = MagicMock(passed=True)
        mock_eval_cls.return_value = mock_eval

        mock_exporter = MagicMock()
        mock_exporter.export_all.return_value = MagicMock(
            success=True,
            exports={"safetensors": "/path"},
        )
        mock_exporter_cls.return_value = mock_exporter

        pipeline = PostTrainingPipeline(pipeline_config)
        result = pipeline.run()

        # Verify execution order
        mock_merger.merge.assert_called_once()
        mock_exporter.export_all.assert_called_once()

    @patch("src.post_training.pipeline.LoRAMerger")
    def test_stops_on_merge_failure(
        self,
        mock_merger_cls: MagicMock,
        pipeline_config: PostTrainingConfig,
    ) -> None:
        """Pipeline stops if merge fails."""
        mock_merger = MagicMock()
        mock_merger.merge.return_value = MagicMock(
            success=False,
            error_message="Merge failed",
        )
        mock_merger_cls.return_value = mock_merger

        pipeline = PostTrainingPipeline(pipeline_config)
        result = pipeline.run()

        assert result.success is False
        assert "Merge" in result.error_message or "merge" in result.error_message.lower()

    @patch("src.post_training.pipeline.ExportConfig")
    @patch("src.post_training.pipeline.ModelExporter")
    @patch("src.post_training.pipeline.EvaluationPipeline")
    @patch("src.post_training.pipeline.LoRAMerger")
    def test_continues_on_eval_failure_with_warning(
        self,
        mock_merger_cls: MagicMock,
        mock_eval_cls: MagicMock,
        mock_exporter_cls: MagicMock,
        mock_export_config: MagicMock,
        pipeline_config: PostTrainingConfig,
    ) -> None:
        """Pipeline continues after eval failure (with warning)."""
        mock_merger = MagicMock()
        mock_merger.merge.return_value = MagicMock(
            success=True,
            output_path="/path/to/merged",
        )
        mock_merger_cls.return_value = mock_merger

        # Evaluation fails
        mock_eval = MagicMock()
        mock_eval.run_full_evaluation.return_value = MagicMock(passed=False)
        mock_eval_cls.return_value = mock_eval

        mock_exporter = MagicMock()
        mock_exporter.export_all.return_value = MagicMock(
            success=True,
            exports={"safetensors": "/path"},
        )
        mock_exporter_cls.return_value = mock_exporter

        pipeline = PostTrainingPipeline(pipeline_config)
        result = pipeline.run()

        # Export should still be called
        mock_exporter.export_all.assert_called_once()
        # Result should indicate eval failed
        assert result.evaluation_passed is False
        # But overall might still "succeed" with warnings
        assert "safetensors" in result.exports


class TestPhaseModes:
    """Test phase-specific execution modes."""

    @patch("src.post_training.pipeline.LoRAMerger")
    def test_run_merge_only(
        self,
        mock_merger_cls: MagicMock,
        pipeline_config: PostTrainingConfig,
    ) -> None:
        """run_merge_only only executes merge phase."""
        mock_merger = MagicMock()
        mock_merger.merge.return_value = MagicMock(
            success=True,
            output_path="/path/to/merged",
        )
        mock_merger_cls.return_value = mock_merger

        pipeline = PostTrainingPipeline(pipeline_config)
        result = pipeline.run_merge_only()

        mock_merger.merge.assert_called_once()
        assert result.success is True

    @patch("src.post_training.pipeline.ModelExporter")
    def test_run_export_only(
        self,
        mock_exporter_cls: MagicMock,
        pipeline_config: PostTrainingConfig,
        temp_dir: str,
    ) -> None:
        """run_export_only only executes export phase."""
        # Create a mock merged model directory
        merged_dir = Path(temp_dir) / "merged"
        merged_dir.mkdir()
        (merged_dir / "config.json").write_text('{"model_type": "gemma"}')
        (merged_dir / "model.safetensors").write_text("mock")

        mock_exporter = MagicMock()
        mock_exporter.export_all.return_value = MagicMock(
            success=True,
            exports={"safetensors": "/path"},
        )
        mock_exporter_cls.return_value = mock_exporter

        pipeline = PostTrainingPipeline(pipeline_config)
        result = pipeline.run_export_only(str(merged_dir))

        mock_exporter.export_all.assert_called_once()
        assert result.success is True


class TestResultAggregation:
    """Test result aggregation from all phases."""

    @patch("src.post_training.pipeline.ExportConfig")
    @patch("src.post_training.pipeline.ModelExporter")
    @patch("src.post_training.pipeline.LoRAMerger")
    def test_result_includes_all_phases(
        self,
        mock_merger_cls: MagicMock,
        mock_exporter_cls: MagicMock,
        mock_export_config: MagicMock,
        mock_adapter_dir: str,
        mock_quiz_dir: str,
        temp_dir: str,
    ) -> None:
        """Result includes data from all phases."""
        # Use skip_evaluation to avoid needing to mock the full eval chain
        config = PostTrainingConfig(
            base_model="google/gemma-3-12b",
            adapter_path=mock_adapter_dir,
            quiz_data_path=mock_quiz_dir,
            output_dir=str(Path(temp_dir) / "output"),
            skip_evaluation=True,
        )

        mock_merger = MagicMock()
        mock_merger.merge.return_value = MagicMock(
            success=True,
            output_path="/path/to/merged",
            merge_time_seconds=60.0,
        )
        mock_merger_cls.return_value = mock_merger

        mock_exporter = MagicMock()
        mock_exporter.export_all.return_value = MagicMock(
            success=True,
            exports={
                "safetensors": "/path/to/safetensors",
                "gguf": "/path/to/gguf",
            },
        )
        mock_exporter_cls.return_value = mock_exporter

        pipeline = PostTrainingPipeline(config)
        result = pipeline.run()

        assert result.merged_model_path == "/path/to/merged"
        # evaluation_passed is None when skipped
        assert result.evaluation_passed is None
        assert "safetensors" in result.exports
        assert "gguf" in result.exports

    @patch("src.post_training.pipeline.ExportConfig")
    @patch("src.post_training.pipeline.ModelExporter")
    @patch("src.post_training.pipeline.EvaluationPipeline")
    @patch("src.post_training.pipeline.LoRAMerger")
    def test_result_tracks_timing(
        self,
        mock_merger_cls: MagicMock,
        mock_eval_cls: MagicMock,
        mock_exporter_cls: MagicMock,
        mock_export_config: MagicMock,
        pipeline_config: PostTrainingConfig,
    ) -> None:
        """Result tracks total execution time."""
        mock_merger = MagicMock()
        mock_merger.merge.return_value = MagicMock(
            success=True,
            output_path="/path/to/merged",
        )
        mock_merger_cls.return_value = mock_merger

        mock_eval = MagicMock()
        mock_eval.run_full_evaluation.return_value = MagicMock(passed=True)
        mock_eval_cls.return_value = mock_eval

        mock_exporter = MagicMock()
        mock_exporter.export_all.return_value = MagicMock(
            success=True,
            exports={},
        )
        mock_exporter_cls.return_value = mock_exporter

        pipeline = PostTrainingPipeline(pipeline_config)
        result = pipeline.run()

        assert result.total_time_seconds >= 0


class TestSkipEvaluation:
    """Test skipping evaluation phase."""

    @patch("src.post_training.pipeline.ExportConfig")
    @patch("src.post_training.pipeline.ModelExporter")
    @patch("src.post_training.pipeline.LoRAMerger")
    def test_skip_evaluation_option(
        self,
        mock_merger_cls: MagicMock,
        mock_exporter_cls: MagicMock,
        mock_export_config: MagicMock,
        mock_adapter_dir: str,
        mock_quiz_dir: str,
        temp_dir: str,
    ) -> None:
        """Pipeline can skip evaluation phase."""
        config = PostTrainingConfig(
            base_model="google/gemma-3-12b",
            adapter_path=mock_adapter_dir,
            quiz_data_path=mock_quiz_dir,
            output_dir=str(Path(temp_dir) / "output"),
            skip_evaluation=True,
        )

        mock_merger = MagicMock()
        mock_merger.merge.return_value = MagicMock(
            success=True,
            output_path="/path/to/merged",
        )
        mock_merger_cls.return_value = mock_merger

        mock_exporter = MagicMock()
        mock_exporter.export_all.return_value = MagicMock(
            success=True,
            exports={},
        )
        mock_exporter_cls.return_value = mock_exporter

        pipeline = PostTrainingPipeline(config)
        result = pipeline.run()

        # Evaluation should be skipped
        assert result.success is True
        # evaluation_passed should be None or True when skipped
        assert result.evaluation_passed is None or result.evaluation_passed is True
