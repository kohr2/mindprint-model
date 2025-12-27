"""
Tests for ReportGenerator - generating evaluation reports in multiple formats.

Tests cover:
- JSON report generation
- Markdown report generation
- Summary generation
- Comparison report generation
"""

import pytest
from pathlib import Path
import json
import tempfile
import shutil

from src.evaluation.reporting import ReportGenerator
from src.evaluation.pipeline import (
    EvaluationReport,
    LevelResult,
    EvalLevel,
)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_topic_result() -> LevelResult:
    """Create a sample topic result."""
    return LevelResult(
        level=EvalLevel.TOPIC,
        identifier="unit-01/chapter-01/topic-01",
        total_questions=10,
        passed_questions=9,
        accuracy=0.90,
        avg_semantic_score=0.85,
        avg_voice_score=0.80,
        critical_distinctions_passed=True,
        negative_patterns_found=[],
        question_results=[],
    )


@pytest.fixture
def sample_unit_result() -> LevelResult:
    """Create a sample unit result."""
    return LevelResult(
        level=EvalLevel.UNIT,
        identifier="unit-01",
        total_questions=50,
        passed_questions=45,
        accuracy=0.90,
        avg_semantic_score=0.85,
        avg_voice_score=0.80,
        critical_distinctions_passed=True,
        negative_patterns_found=[],
        question_results=[],
    )


@pytest.fixture
def sample_report(sample_topic_result, sample_unit_result) -> EvaluationReport:
    """Create a sample evaluation report."""
    return EvaluationReport(
        model_name="gemma-3-12b-bob-v1",
        approach="dpo",
        timestamp="2024-01-01T12:00:00",
        topic_results=[sample_topic_result],
        chapter_results=[],
        unit_results=[sample_unit_result],
        final_result=None,
        overall_accuracy=0.90,
        overall_voice_score=0.80,
        passed=True,
        recommendations=["Model passed all evaluation criteria."],
    )


class TestReportGeneratorInit:
    """Test ReportGenerator initialization."""

    def test_creates_output_directory(self) -> None:
        """Creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "new" / "nested" / "dir"

            generator = ReportGenerator(str(new_dir))

            assert new_dir.exists()

    def test_uses_existing_directory(self, temp_output_dir: str) -> None:
        """Uses existing directory without error."""
        generator = ReportGenerator(temp_output_dir)

        assert generator.output_dir == Path(temp_output_dir)


class TestJSONReportGeneration:
    """Test JSON report generation."""

    def test_generates_json_file(
        self, temp_output_dir: str, sample_report: EvaluationReport
    ) -> None:
        """Generates a JSON file."""
        generator = ReportGenerator(temp_output_dir)

        output_path = generator.generate_json(sample_report)

        assert output_path.exists()
        assert output_path.suffix == ".json"

    def test_json_contains_all_fields(
        self, temp_output_dir: str, sample_report: EvaluationReport
    ) -> None:
        """JSON contains all report fields."""
        generator = ReportGenerator(temp_output_dir)

        output_path = generator.generate_json(sample_report)

        with open(output_path) as f:
            data = json.load(f)

        assert data["model_name"] == "gemma-3-12b-bob-v1"
        assert data["approach"] == "dpo"
        assert data["overall_accuracy"] == 0.90
        assert data["passed"] is True

    def test_json_includes_topic_details(
        self, temp_output_dir: str, sample_report: EvaluationReport
    ) -> None:
        """JSON includes detailed topic results."""
        generator = ReportGenerator(temp_output_dir)

        output_path = generator.generate_json(sample_report)

        with open(output_path) as f:
            data = json.load(f)

        assert "topic_details" in data
        assert len(data["topic_details"]) == 1
        assert data["topic_details"][0]["identifier"] == "unit-01/chapter-01/topic-01"

    def test_json_filename_includes_timestamp(
        self, temp_output_dir: str, sample_report: EvaluationReport
    ) -> None:
        """JSON filename includes sanitized timestamp."""
        generator = ReportGenerator(temp_output_dir)

        output_path = generator.generate_json(sample_report)

        assert "2024-01-01" in output_path.name


class TestMarkdownReportGeneration:
    """Test Markdown report generation."""

    def test_generates_markdown_file(
        self, temp_output_dir: str, sample_report: EvaluationReport
    ) -> None:
        """Generates a Markdown file."""
        generator = ReportGenerator(temp_output_dir)

        output_path = generator.generate_markdown(sample_report)

        assert output_path.exists()
        assert output_path.suffix == ".md"

    def test_markdown_has_title(
        self, temp_output_dir: str, sample_report: EvaluationReport
    ) -> None:
        """Markdown has model name in title."""
        generator = ReportGenerator(temp_output_dir)

        output_path = generator.generate_markdown(sample_report)

        content = output_path.read_text()

        assert "# Evaluation Report: gemma-3-12b-bob-v1" in content

    def test_markdown_has_summary_table(
        self, temp_output_dir: str, sample_report: EvaluationReport
    ) -> None:
        """Markdown has summary metrics table."""
        generator = ReportGenerator(temp_output_dir)

        output_path = generator.generate_markdown(sample_report)

        content = output_path.read_text()

        assert "## Summary Metrics" in content
        assert "Overall Accuracy" in content
        assert "Voice Fidelity" in content

    def test_markdown_has_topic_results(
        self, temp_output_dir: str, sample_report: EvaluationReport
    ) -> None:
        """Markdown includes topic results table."""
        generator = ReportGenerator(temp_output_dir)

        output_path = generator.generate_markdown(sample_report)

        content = output_path.read_text()

        assert "## Topic Results" in content
        assert "unit-01/chapter-01/topic-01" in content

    def test_markdown_has_recommendations(
        self, temp_output_dir: str, sample_report: EvaluationReport
    ) -> None:
        """Markdown includes recommendations."""
        generator = ReportGenerator(temp_output_dir)

        output_path = generator.generate_markdown(sample_report)

        content = output_path.read_text()

        assert "## Recommendations" in content
        assert "passed all" in content.lower()

    def test_markdown_shows_pass_status(
        self, temp_output_dir: str, sample_report: EvaluationReport
    ) -> None:
        """Markdown shows PASSED status."""
        generator = ReportGenerator(temp_output_dir)

        output_path = generator.generate_markdown(sample_report)

        content = output_path.read_text()

        assert "PASSED" in content


class TestFailedReport:
    """Test report generation for failed evaluation."""

    def test_markdown_shows_fail_status(
        self, temp_output_dir: str, sample_topic_result: LevelResult
    ) -> None:
        """Markdown shows FAILED status."""
        failed_report = EvaluationReport(
            model_name="failed-model",
            approach="dpo",
            timestamp="2024-01-01T12:00:00",
            topic_results=[sample_topic_result],
            chapter_results=[],
            unit_results=[],
            final_result=None,
            overall_accuracy=0.60,
            overall_voice_score=0.50,
            passed=False,
            failed_topics=["unit-01/chapter-01/topic-01"],
            critical_violations=["halving confusion"],
        )

        generator = ReportGenerator(temp_output_dir)
        output_path = generator.generate_markdown(failed_report)

        content = output_path.read_text()

        assert "FAILED" in content

    def test_markdown_lists_failed_topics(
        self, temp_output_dir: str, sample_topic_result: LevelResult
    ) -> None:
        """Markdown lists failed topics."""
        failed_report = EvaluationReport(
            model_name="failed-model",
            approach="dpo",
            timestamp="2024-01-01T12:00:00",
            topic_results=[sample_topic_result],
            chapter_results=[],
            unit_results=[],
            final_result=None,
            overall_accuracy=0.60,
            overall_voice_score=0.50,
            passed=False,
            failed_topics=["unit-01/chapter-01/topic-01"],
        )

        generator = ReportGenerator(temp_output_dir)
        output_path = generator.generate_markdown(failed_report)

        content = output_path.read_text()

        assert "## Failed Topics" in content
        assert "unit-01/chapter-01/topic-01" in content

    def test_markdown_lists_critical_violations(
        self, temp_output_dir: str, sample_topic_result: LevelResult
    ) -> None:
        """Markdown lists critical violations."""
        failed_report = EvaluationReport(
            model_name="failed-model",
            approach="dpo",
            timestamp="2024-01-01T12:00:00",
            topic_results=[sample_topic_result],
            chapter_results=[],
            unit_results=[],
            final_result=None,
            overall_accuracy=0.60,
            overall_voice_score=0.50,
            passed=False,
            critical_violations=["halving causes cycle"],
        )

        generator = ReportGenerator(temp_output_dir)
        output_path = generator.generate_markdown(failed_report)

        content = output_path.read_text()

        assert "## Critical Violations" in content
        assert "halving causes cycle" in content


class TestSummaryGeneration:
    """Test summary generation."""

    def test_generates_summary_string(
        self, temp_output_dir: str, sample_report: EvaluationReport
    ) -> None:
        """Generates a summary string."""
        generator = ReportGenerator(temp_output_dir)

        summary = generator.generate_summary(sample_report)

        assert isinstance(summary, str)
        assert "PASS" in summary

    def test_summary_includes_model_name(
        self, temp_output_dir: str, sample_report: EvaluationReport
    ) -> None:
        """Summary includes model name."""
        generator = ReportGenerator(temp_output_dir)

        summary = generator.generate_summary(sample_report)

        assert "gemma-3-12b-bob-v1" in summary

    def test_summary_includes_metrics(
        self, temp_output_dir: str, sample_report: EvaluationReport
    ) -> None:
        """Summary includes key metrics."""
        generator = ReportGenerator(temp_output_dir)

        summary = generator.generate_summary(sample_report)

        assert "Acc=" in summary
        assert "Voice=" in summary


class TestComparisonReport:
    """Test comparison report generation."""

    def test_generates_comparison_file(
        self, temp_output_dir: str, sample_report: EvaluationReport
    ) -> None:
        """Generates a comparison file."""
        generator = ReportGenerator(temp_output_dir)

        output_path = generator.generate_comparison([sample_report])

        assert output_path.exists()
        assert output_path.suffix == ".md"

    def test_comparison_has_overview_table(
        self, temp_output_dir: str, sample_report: EvaluationReport
    ) -> None:
        """Comparison has overview table."""
        generator = ReportGenerator(temp_output_dir)

        output_path = generator.generate_comparison([sample_report])

        content = output_path.read_text()

        assert "## Overview" in content
        assert "| Model |" in content

    def test_comparison_multiple_models(
        self, temp_output_dir: str, sample_topic_result: LevelResult, sample_unit_result: LevelResult
    ) -> None:
        """Comparison works with multiple models."""
        report1 = EvaluationReport(
            model_name="model-v1",
            approach="dpo",
            timestamp="2024-01-01T12:00:00",
            topic_results=[sample_topic_result],
            chapter_results=[],
            unit_results=[sample_unit_result],
            final_result=None,
            overall_accuracy=0.85,
            overall_voice_score=0.75,
            passed=True,
        )

        report2 = EvaluationReport(
            model_name="model-v2",
            approach="dpo",
            timestamp="2024-01-01T13:00:00",
            topic_results=[sample_topic_result],
            chapter_results=[],
            unit_results=[sample_unit_result],
            final_result=None,
            overall_accuracy=0.90,
            overall_voice_score=0.80,
            passed=True,
        )

        generator = ReportGenerator(temp_output_dir)
        output_path = generator.generate_comparison([report1, report2])

        content = output_path.read_text()

        assert "model-v1" in content
        assert "model-v2" in content
        assert "Models Compared:** 2" in content

    def test_comparison_per_unit_section(
        self, temp_output_dir: str, sample_topic_result: LevelResult, sample_unit_result: LevelResult
    ) -> None:
        """Comparison has per-unit breakdown."""
        report = EvaluationReport(
            model_name="test-model",
            approach="dpo",
            timestamp="2024-01-01T12:00:00",
            topic_results=[sample_topic_result],
            chapter_results=[],
            unit_results=[sample_unit_result],
            final_result=None,
            overall_accuracy=0.90,
            overall_voice_score=0.80,
            passed=True,
        )

        generator = ReportGenerator(temp_output_dir)
        output_path = generator.generate_comparison([report])

        content = output_path.read_text()

        assert "## Per-Unit Comparison" in content
        assert "unit-01" in content


class TestGenerateAll:
    """Test generate_all convenience method."""

    def test_generates_all_formats(
        self, temp_output_dir: str, sample_report: EvaluationReport
    ) -> None:
        """generate_all creates JSON, Markdown, and summary."""
        generator = ReportGenerator(temp_output_dir)

        paths = generator.generate_all(sample_report)

        assert "json" in paths
        assert "markdown" in paths
        assert "summary" in paths

        assert paths["json"].exists()
        assert paths["markdown"].exists()
        assert isinstance(paths["summary"], str)
