"""
Tests for EvaluationPipeline - hierarchical evaluation pipeline for mindprint models.

Tests cover:
- Data class behaviors (QuestionResult, LevelResult, EvaluationReport)
- Quiz data loading
- Level evaluation
- Report building
- Recommendation generation
"""

import pytest
from pathlib import Path
import json
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.evaluation.pipeline import (
    EvaluationPipeline,
    EvalLevel,
    QuestionResult,
    LevelResult,
    EvaluationReport,
)


@pytest.fixture
def temp_quiz_dir():
    """Create a temporary directory with quiz data."""
    temp_dir = tempfile.mkdtemp()

    # Create quiz_data.json
    quiz_data = [
        {
            "level": "topic",
            "unit": "unit-01",
            "chapter": "chapter-01",
            "topic": "topic-01",
            "title": "The Three Types",
            "questions": [
                {
                    "question": "What is the difference between a trader and gambler?",
                    "reference_answer": "A trader follows rules, a gambler doesn't.",
                    "type": "open",
                },
            ],
        },
    ]
    with open(Path(temp_dir) / "quiz_data.json", "w") as f:
        json.dump(quiz_data, f)

    # Create chapter_tests.json
    chapter_data = [
        {
            "level": "chapter",
            "unit": "unit-01",
            "chapter": "chapter-01",
            "title": "Market Participants",
            "questions": [
                {
                    "question": "Explain market psychology.",
                    "reference_answer": "Market psychology drives cycles.",
                    "type": "open",
                },
            ],
        },
    ]
    with open(Path(temp_dir) / "chapter_tests.json", "w") as f:
        json.dump(chapter_data, f)

    # Create unit_exams.json
    unit_data = [
        {
            "level": "unit",
            "unit": "unit-01",
            "title": "Foundation Unit Exam",
            "questions": [
                {
                    "question": "Describe the 4-year cycle.",
                    "reference_answer": "The 4-year cycle is driven by psychology.",
                    "type": "open",
                },
            ],
        },
    ]
    with open(Path(temp_dir) / "unit_exams.json", "w") as f:
        json.dump(unit_data, f)

    # Create final_assessment.json
    final_data = {
        "level": "final",
        "title": "Final Assessment",
        "questions": [
            {
                "question": "Synthesize your understanding of cycles.",
                "reference_answer": "Cycles are psychological phenomena.",
                "type": "open",
            },
        ],
    }
    with open(Path(temp_dir) / "final_assessment.json", "w") as f:
        json.dump(final_data, f)

    yield temp_dir
    shutil.rmtree(temp_dir)


class TestEvalLevel:
    """Test EvalLevel enum."""

    def test_has_all_levels(self) -> None:
        """All expected levels exist."""
        assert EvalLevel.TOPIC.value == "topic"
        assert EvalLevel.CHAPTER.value == "chapter"
        assert EvalLevel.UNIT.value == "unit"
        assert EvalLevel.FINAL.value == "final"


class TestQuestionResult:
    """Test QuestionResult dataclass."""

    def test_stores_all_fields(self) -> None:
        """QuestionResult stores all fields correctly."""
        result = QuestionResult(
            question_id="unit-01/q1",
            question="Test question?",
            reference_answer="Reference answer.",
            generated_answer="Generated answer.",
            semantic_score=0.85,
            voice_score=0.75,
            passed=True,
            violations=["test violation"],
        )

        assert result.question_id == "unit-01/q1"
        assert result.semantic_score == 0.85
        assert result.passed is True
        assert "test violation" in result.violations


class TestLevelResult:
    """Test LevelResult dataclass."""

    def test_topic_passed_thresholds(self) -> None:
        """Topic passes with 90% accuracy and 0.75 voice."""
        result = LevelResult(
            level=EvalLevel.TOPIC,
            identifier="unit-01/chapter-01/topic-01",
            total_questions=10,
            passed_questions=9,
            accuracy=0.90,
            avg_semantic_score=0.85,
            avg_voice_score=0.76,
            critical_distinctions_passed=True,
            negative_patterns_found=[],
            question_results=[],
        )

        assert result.passed is True

    def test_topic_fails_below_accuracy(self) -> None:
        """Topic fails with < 90% accuracy."""
        result = LevelResult(
            level=EvalLevel.TOPIC,
            identifier="unit-01/chapter-01/topic-01",
            total_questions=10,
            passed_questions=8,
            accuracy=0.80,  # Below 90%
            avg_semantic_score=0.85,
            avg_voice_score=0.80,
            critical_distinctions_passed=True,
            negative_patterns_found=[],
            question_results=[],
        )

        assert result.passed is False

    def test_chapter_passed_thresholds(self) -> None:
        """Chapter passes with 85% accuracy and 0.75 voice."""
        result = LevelResult(
            level=EvalLevel.CHAPTER,
            identifier="unit-01/chapter-01",
            total_questions=20,
            passed_questions=17,
            accuracy=0.85,
            avg_semantic_score=0.80,
            avg_voice_score=0.75,
            critical_distinctions_passed=True,
            negative_patterns_found=[],
            question_results=[],
        )

        assert result.passed is True

    def test_unit_passed_thresholds(self) -> None:
        """Unit passes with 80% accuracy and 0.75 voice."""
        result = LevelResult(
            level=EvalLevel.UNIT,
            identifier="unit-01",
            total_questions=50,
            passed_questions=40,
            accuracy=0.80,
            avg_semantic_score=0.75,
            avg_voice_score=0.75,
            critical_distinctions_passed=True,
            negative_patterns_found=[],
            question_results=[],
        )

        assert result.passed is True

    def test_final_requires_higher_voice(self) -> None:
        """Final requires 0.80 voice (higher than other levels)."""
        result = LevelResult(
            level=EvalLevel.FINAL,
            identifier="final",
            total_questions=20,
            passed_questions=17,
            accuracy=0.85,
            avg_semantic_score=0.80,
            avg_voice_score=0.78,  # Below 0.80 required for final
            critical_distinctions_passed=True,
            negative_patterns_found=[],
            question_results=[],
        )

        assert result.passed is False

    def test_fails_with_negative_patterns(self) -> None:
        """Level fails if negative patterns found."""
        result = LevelResult(
            level=EvalLevel.TOPIC,
            identifier="unit-01/chapter-01/topic-01",
            total_questions=10,
            passed_questions=10,
            accuracy=1.0,
            avg_semantic_score=0.95,
            avg_voice_score=0.90,
            critical_distinctions_passed=True,
            negative_patterns_found=["halving causes"],  # Violation!
            question_results=[],
        )

        assert result.passed is False

    def test_to_dict(self) -> None:
        """to_dict includes all relevant fields."""
        result = LevelResult(
            level=EvalLevel.TOPIC,
            identifier="test",
            total_questions=10,
            passed_questions=9,
            accuracy=0.90,
            avg_semantic_score=0.85,
            avg_voice_score=0.80,
            critical_distinctions_passed=True,
            negative_patterns_found=[],
            question_results=[],
        )

        result_dict = result.to_dict()

        assert result_dict["level"] == "topic"
        assert result_dict["identifier"] == "test"
        assert "passed" in result_dict


class TestEvaluationReport:
    """Test EvaluationReport dataclass."""

    def test_to_dict_includes_summary(self) -> None:
        """to_dict includes summary statistics."""
        topic_result = LevelResult(
            level=EvalLevel.TOPIC,
            identifier="test",
            total_questions=10,
            passed_questions=9,
            accuracy=0.90,
            avg_semantic_score=0.85,
            avg_voice_score=0.80,
            critical_distinctions_passed=True,
            negative_patterns_found=[],
            question_results=[],
        )

        report = EvaluationReport(
            model_name="test-model",
            approach="dpo",
            timestamp="2024-01-01T00:00:00",
            topic_results=[topic_result],
            chapter_results=[],
            unit_results=[],
            final_result=None,
            overall_accuracy=0.90,
            overall_voice_score=0.80,
            passed=True,
        )

        report_dict = report.to_dict()

        assert "summary" in report_dict
        assert report_dict["summary"]["topics"]["count"] == 1
        assert report_dict["summary"]["topics"]["passed"] == 1


class TestQuizDataLoading:
    """Test quiz data loading."""

    def test_loads_all_quiz_files(self, temp_quiz_dir: str) -> None:
        """Loads all quiz data files correctly."""
        with patch("src.evaluation.pipeline.VoiceFidelityEvaluator"):
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()

            pipeline = EvaluationPipeline(
                model=mock_model,
                tokenizer=mock_tokenizer,
                quiz_data_path=temp_quiz_dir,
            )

        assert len(pipeline.quiz_data["topics"]) == 1
        assert len(pipeline.quiz_data["chapters"]) == 1
        assert len(pipeline.quiz_data["units"]) == 1
        assert pipeline.quiz_data["final"] is not None

    def test_handles_missing_files(self) -> None:
        """Handles missing quiz files gracefully."""
        with tempfile.TemporaryDirectory() as empty_dir:
            with patch("src.evaluation.pipeline.VoiceFidelityEvaluator"):
                mock_model = MagicMock()
                mock_tokenizer = MagicMock()

                pipeline = EvaluationPipeline(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    quiz_data_path=empty_dir,
                )

            assert pipeline.quiz_data["topics"] == []
            assert pipeline.quiz_data["final"] is None


class TestRecommendationGeneration:
    """Test recommendation generation."""

    def test_generates_critical_recommendation(self, temp_quiz_dir: str) -> None:
        """Generates CRITICAL recommendation for halving violations."""
        with patch("src.evaluation.pipeline.VoiceFidelityEvaluator"):
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()

            pipeline = EvaluationPipeline(
                model=mock_model,
                tokenizer=mock_tokenizer,
                quiz_data_path=temp_quiz_dir,
            )

            recommendations = pipeline._generate_recommendations(
                topic_results=[],
                chapter_results=[],
                unit_results=[],
                final_result=None,
                failed_topics=[],
                critical_violations=["unit-01/topic-01: halving causes"],
            )

            assert any("CRITICAL" in r for r in recommendations)
            assert any("halving" in r.lower() for r in recommendations)

    def test_generates_high_recommendation_for_many_failures(
        self, temp_quiz_dir: str
    ) -> None:
        """Generates HIGH recommendation when >30% topics fail."""
        with patch("src.evaluation.pipeline.VoiceFidelityEvaluator"):
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()

            pipeline = EvaluationPipeline(
                model=mock_model,
                tokenizer=mock_tokenizer,
                quiz_data_path=temp_quiz_dir,
            )

            # Create 10 topic results
            topic_results = [
                LevelResult(
                    level=EvalLevel.TOPIC,
                    identifier=f"unit-01/chapter-01/topic-{i:02d}",
                    total_questions=10,
                    passed_questions=9,
                    accuracy=0.90,
                    avg_semantic_score=0.85,
                    avg_voice_score=0.80,
                    critical_distinctions_passed=True,
                    negative_patterns_found=[],
                    question_results=[],
                )
                for i in range(10)
            ]

            # 4 failed topics (40%)
            failed_topics = [
                "unit-01/chapter-01/topic-01",
                "unit-01/chapter-01/topic-02",
                "unit-01/chapter-01/topic-03",
                "unit-01/chapter-01/topic-04",
            ]

            recommendations = pipeline._generate_recommendations(
                topic_results=topic_results,
                chapter_results=[],
                unit_results=[],
                final_result=None,
                failed_topics=failed_topics,
                critical_violations=[],
            )

            assert any("HIGH" in r for r in recommendations)

    def test_generates_pass_recommendation_when_clean(
        self, temp_quiz_dir: str
    ) -> None:
        """Generates pass recommendation when no issues."""
        with patch("src.evaluation.pipeline.VoiceFidelityEvaluator"):
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()

            pipeline = EvaluationPipeline(
                model=mock_model,
                tokenizer=mock_tokenizer,
                quiz_data_path=temp_quiz_dir,
            )

            recommendations = pipeline._generate_recommendations(
                topic_results=[],
                chapter_results=[],
                unit_results=[],
                final_result=None,
                failed_topics=[],
                critical_violations=[],
            )

            assert any("passed all" in r.lower() for r in recommendations)


class TestChapterAndUnitSynthesis:
    """Test chapter and unit test synthesis from topics."""

    def test_synthesizes_chapter_tests(self, temp_quiz_dir: str) -> None:
        """Synthesizes chapter tests from topic data when not provided."""
        # Create quiz data without chapter tests
        with tempfile.TemporaryDirectory() as temp_dir:
            quiz_data = [
                {
                    "unit": "unit-01",
                    "chapter": "chapter-01",
                    "topic": "topic-01",
                    "questions": [{"question": "Q1", "reference_answer": "A1"}],
                },
                {
                    "unit": "unit-01",
                    "chapter": "chapter-01",
                    "topic": "topic-02",
                    "questions": [{"question": "Q2", "reference_answer": "A2"}],
                },
            ]
            with open(Path(temp_dir) / "quiz_data.json", "w") as f:
                json.dump(quiz_data, f)

            with patch("src.evaluation.pipeline.VoiceFidelityEvaluator"):
                mock_model = MagicMock()
                mock_tokenizer = MagicMock()

                pipeline = EvaluationPipeline(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    quiz_data_path=temp_dir,
                )

                chapter_tests = pipeline._get_chapter_tests()

                # Should synthesize 1 chapter test from 2 topics
                assert len(chapter_tests) == 1
                assert len(chapter_tests[0]["questions"]) == 2

    def test_synthesizes_unit_exams(self, temp_quiz_dir: str) -> None:
        """Synthesizes unit exams from topic data when not provided."""
        with tempfile.TemporaryDirectory() as temp_dir:
            quiz_data = [
                {
                    "unit": "unit-01",
                    "chapter": "chapter-01",
                    "topic": "topic-01",
                    "questions": [{"question": "Q1", "reference_answer": "A1"}],
                },
                {
                    "unit": "unit-01",
                    "chapter": "chapter-02",
                    "topic": "topic-01",
                    "questions": [{"question": "Q2", "reference_answer": "A2"}],
                },
            ]
            with open(Path(temp_dir) / "quiz_data.json", "w") as f:
                json.dump(quiz_data, f)

            with patch("src.evaluation.pipeline.VoiceFidelityEvaluator"):
                mock_model = MagicMock()
                mock_tokenizer = MagicMock()

                pipeline = EvaluationPipeline(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    quiz_data_path=temp_dir,
                )

                unit_exams = pipeline._get_unit_exams()

                # Should synthesize 1 unit exam from 2 topics
                assert len(unit_exams) == 1
                assert len(unit_exams[0]["questions"]) == 2
