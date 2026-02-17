"""
Tests for DataPipeline - orchestrating the full data preparation flow.

Tests cover:
- Pipeline configuration
- SFT data creation
- Preference pair creation
- Output file saving
- Statistics tracking
"""

import pytest
from pathlib import Path
import json
import tempfile
import shutil
from unittest.mock import MagicMock

from src.data_prep.pipeline import DataPipeline, PipelineConfig, PipelineStats
from src.data_prep.textbook_parser import Question, TopicQuiz, ChapterTest, UnitExam


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def fixtures_path() -> Path:
    """Path to test fixtures."""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def sample_topic_quiz() -> TopicQuiz:
    """Sample TopicQuiz for testing."""
    return TopicQuiz(
        unit="unit-01",
        chapter="chapter-01",
        topic="topic-01",
        title="The Three Types",
        questions=[
            Question(
                question="What is the difference between a gambler and a trader?",
                reference_answer="Look, the key difference is discipline. A trader follows systematic rules, while a gambler makes emotional decisions based on hope.",
                question_type="open",
                key_concepts=["gambler_definition", "trader_definition"],
            ),
            Question(
                question="Why is position sizing important?",
                reference_answer="I've learned that position sizing is everything. It's not about being right, it's about surviving being wrong.",
                question_type="open",
                key_concepts=["position_sizing", "risk_management"],
            ),
        ],
    )


@pytest.fixture
def sample_chapter_test() -> ChapterTest:
    """Sample ChapterTest for testing."""
    return ChapterTest(
        unit="unit-01",
        chapter="chapter-01",
        title="Market Participants",
        questions=[
            Question(
                question="True or False: Technical analysis guarantees profits.",
                reference_answer="False. Technical analysis is a tool, not a guarantee.",
                question_type="true_false",
            ),
            Question(
                question="What makes someone an investor vs a trader?",
                reference_answer="Time horizon is key. An investor thinks in years, a trader in days or weeks.",
                question_type="open",
            ),
        ],
    )


@pytest.fixture
def sample_unit_exam() -> UnitExam:
    """Sample UnitExam for testing."""
    return UnitExam(
        unit="unit-01",
        title="Foundation Unit Exam",
        questions=[
            Question(
                question="Explain the 4-year cycle concept.",
                reference_answer="The 4-year cycle is about market psychology, not halvings.",
                question_type="open",
            ),
        ],
    )


class TestPipelineConfig:
    """Test pipeline configuration."""

    def test_default_values(self, temp_output_dir: str, fixtures_path: Path) -> None:
        """Config has sensible defaults."""
        config = PipelineConfig(
            textbook_path=str(fixtures_path),
            output_path=temp_output_dir,
        )

        assert config.target_questions_per_topic == 10
        assert config.augment_questions is True
        assert config.include_critical_distinctions is True

    def test_custom_values(self, temp_output_dir: str, fixtures_path: Path) -> None:
        """Config accepts custom values."""
        config = PipelineConfig(
            textbook_path=str(fixtures_path),
            output_path=temp_output_dir,
            target_questions_per_topic=5,
            augment_questions=False,
            include_critical_distinctions=False,
        )

        assert config.target_questions_per_topic == 5
        assert config.augment_questions is False
        assert config.include_critical_distinctions is False


class TestPipelineStats:
    """Test pipeline statistics."""

    def test_default_stats(self) -> None:
        """Stats initialize to zero."""
        stats = PipelineStats()

        assert stats.topics_processed == 0
        assert stats.chapters_processed == 0
        assert stats.sft_examples == 0
        assert stats.timestamp  # Has a timestamp

    def test_stats_tracking(self) -> None:
        """Stats can be updated."""
        stats = PipelineStats()
        stats.topics_processed = 10
        stats.sft_examples = 100

        assert stats.topics_processed == 10
        assert stats.sft_examples == 100


class TestSFTDataCreation:
    """Test SFT training data creation."""

    def test_creates_sft_from_topic_quiz(
        self, temp_output_dir: str, fixtures_path: Path, sample_topic_quiz: TopicQuiz
    ) -> None:
        """Creates SFT data from topic quizzes."""
        config = PipelineConfig(
            textbook_path=str(fixtures_path),
            output_path=temp_output_dir,
            augment_questions=False,
        )
        pipeline = DataPipeline(config)

        sft_data = pipeline._create_sft_data([sample_topic_quiz], [], [])

        assert len(sft_data) == 2
        assert sft_data[0]["instruction"] == sample_topic_quiz.questions[0].question
        assert sft_data[0]["output"] == sample_topic_quiz.questions[0].reference_answer
        assert sft_data[0]["input"] == ""
        assert "unit-01" in sft_data[0]["source"]

    def test_excludes_mc_and_tf_from_chapters(
        self, temp_output_dir: str, fixtures_path: Path, sample_chapter_test: ChapterTest
    ) -> None:
        """Excludes multiple choice and true/false from SFT."""
        config = PipelineConfig(
            textbook_path=str(fixtures_path),
            output_path=temp_output_dir,
            augment_questions=False,
        )
        pipeline = DataPipeline(config)

        sft_data = pipeline._create_sft_data([], [sample_chapter_test], [])

        # Should only include the open-ended question
        assert len(sft_data) == 1
        assert "investor vs a trader" in sft_data[0]["instruction"]


class TestPreferencePairCreation:
    """Test preference pair creation."""

    def test_creates_pairs_from_topic_quiz(
        self, temp_output_dir: str, fixtures_path: Path, sample_topic_quiz: TopicQuiz
    ) -> None:
        """Creates preference pairs from topic quizzes."""
        config = PipelineConfig(
            textbook_path=str(fixtures_path),
            output_path=temp_output_dir,
            augment_questions=False,
        )
        pipeline = DataPipeline(config)

        pairs = pipeline._create_preference_pairs([sample_topic_quiz], [], [])

        assert len(pairs) == 2
        assert pairs[0].prompt == sample_topic_quiz.questions[0].question
        assert pairs[0].chosen == sample_topic_quiz.questions[0].reference_answer
        assert pairs[0].rejected != pairs[0].chosen

    def test_pairs_only_from_open_ended(
        self, temp_output_dir: str, fixtures_path: Path, sample_chapter_test: ChapterTest
    ) -> None:
        """Creates pairs only from open-ended questions."""
        config = PipelineConfig(
            textbook_path=str(fixtures_path),
            output_path=temp_output_dir,
            augment_questions=False,
        )
        pipeline = DataPipeline(config)

        pairs = pipeline._create_preference_pairs([], [sample_chapter_test], [])

        # Should only include the open-ended question
        assert len(pairs) == 1


class TestTranscriptHierarchySynthesis:
    """Test transcript quiz/chapter/unit synthesis helpers."""

    def test_creates_topic_quizzes_from_transcript_sources(
        self, temp_output_dir: str, fixtures_path: Path
    ) -> None:
        config = PipelineConfig(
            textbook_path=str(fixtures_path),
            output_path=temp_output_dir,
            augment_questions=False,
        )
        pipeline = DataPipeline(config)

        transcript_questions = [
            Question(
                question="Q1",
                reference_answer="A1",
                source="episode-2026-01-15",
            ),
            Question(
                question="Q2",
                reference_answer="A2",
                source="episode-2026-01-15",
            ),
            Question(
                question="Q3",
                reference_answer="A3",
                source="episode-2026-02-02",
            ),
        ]

        quizzes = pipeline._create_transcript_topic_quizzes(transcript_questions)

        assert len(quizzes) == 2
        assert quizzes[0].unit == "unit-2026"
        assert quizzes[0].chapter in {"chapter-01", "chapter-02"}
        assert all(q.topic.startswith("episode-2026-") for q in quizzes)

    def test_weekly_bucket_groups_transcripts_into_fewer_topics(
        self, temp_output_dir: str, fixtures_path: Path
    ) -> None:
        config = PipelineConfig(
            textbook_path=str(fixtures_path),
            output_path=temp_output_dir,
            augment_questions=False,
            transcript_topic_bucket_days=7,
        )
        pipeline = DataPipeline(config)

        transcript_questions = [
            Question(
                question="Q1",
                reference_answer="A1",
                source="episode-2026-01-12",  # ISO week 3
            ),
            Question(
                question="Q2",
                reference_answer="A2",
                source="episode-2026-01-15",  # ISO week 3
            ),
            Question(
                question="Q3",
                reference_answer="A3",
                source="episode-2026-01-22",  # ISO week 4
            ),
        ]

        quizzes = pipeline._create_transcript_topic_quizzes(transcript_questions)

        assert len(quizzes) == 2
        assert quizzes[0].topic.startswith("transcript-2026-w")
        assert quizzes[1].topic.startswith("transcript-2026-w")

    def test_bucket_source_weekly_and_biweekly(
        self, temp_output_dir: str, fixtures_path: Path
    ) -> None:
        weekly = DataPipeline(
            PipelineConfig(
                textbook_path=str(fixtures_path),
                output_path=temp_output_dir,
                transcript_topic_bucket_days=7,
                augment_questions=False,
            )
        )
        biweekly = DataPipeline(
            PipelineConfig(
                textbook_path=str(fixtures_path),
                output_path=temp_output_dir,
                transcript_topic_bucket_days=14,
                augment_questions=False,
            )
        )

        assert weekly._bucket_transcript_source("episode-2026-01-15").startswith(
            "transcript-2026-w"
        )
        assert biweekly._bucket_transcript_source("episode-2026-01-15").startswith(
            "transcript-2026-bw"
        )
        assert weekly._bucket_transcript_source("unit-01/chapter-01/topic-01") == (
            "unit-01/chapter-01/topic-01"
        )

    def test_synthesizes_chapter_and_unit_from_transcript_topics(
        self, temp_output_dir: str, fixtures_path: Path
    ) -> None:
        config = PipelineConfig(
            textbook_path=str(fixtures_path),
            output_path=temp_output_dir,
            augment_questions=False,
        )
        pipeline = DataPipeline(config)

        topic_quizzes = [
            TopicQuiz(
                unit="unit-2026",
                chapter="chapter-01",
                topic="episode-2026-01-15",
                title="Episode 2026-01-15",
                questions=[Question(question="Q1", reference_answer="A1", source="episode-2026-01-15")],
            ),
            TopicQuiz(
                unit="unit-2026",
                chapter="chapter-02",
                topic="episode-2026-02-02",
                title="Episode 2026-02-02",
                questions=[Question(question="Q2", reference_answer="A2", source="episode-2026-02-02")],
            ),
        ]

        chapter_tests, unit_exams = pipeline._synthesize_hierarchy_from_topics(topic_quizzes)

        assert len(chapter_tests) == 2
        assert len(unit_exams) == 1
        assert unit_exams[0].unit == "unit-2026"
        assert len(unit_exams[0].questions) == 2

    def test_filters_synthetic_transcript_questions_for_training(
        self, temp_output_dir: str, fixtures_path: Path
    ) -> None:
        config = PipelineConfig(
            textbook_path=str(fixtures_path),
            output_path=temp_output_dir,
            augment_questions=False,
            use_transcripts=True,
            transcript_dir=temp_output_dir,
            exclude_synthetic_transcript_questions=True,
        )
        pipeline = DataPipeline(config)

        real_q = Question(
            question="Real question",
            reference_answer="Real answer",
            source="episode-2026-01-12",
            key_concepts=["cycle"],
        )
        synthetic_q = Question(
            question="Synthetic question",
            reference_answer="Synthetic answer",
            source="episode-2026-01-12",
            key_concepts=[DataPipeline.SYNTHETIC_TRANSCRIPT_TAG],
        )

        filtered = pipeline._filter_training_transcript_questions([real_q, synthetic_q])
        assert len(filtered) == 1
        assert filtered[0].question == "Real question"

    def test_merges_weak_weekly_bucket_into_adjacent(
        self, temp_output_dir: str, fixtures_path: Path
    ) -> None:
        config = PipelineConfig(
            textbook_path=str(fixtures_path),
            output_path=temp_output_dir,
            augment_questions=False,
            transcript_topic_bucket_days=7,
            transcript_merge_weak_buckets=True,
            transcript_min_train_pairs_per_bucket=30,
            transcript_holdout_ratio_for_bucketing=0.2,
        )
        pipeline = DataPipeline(config)

        questions = [
            Question(
                question=f"W3-{i}",
                reference_answer="A",
                source="episode-2026-01-12",  # ISO week 03
            )
            for i in range(20)
        ] + [
            Question(
                question=f"W4-{i}",
                reference_answer="A",
                source="episode-2026-01-22",  # ISO week 04
            )
            for i in range(50)
        ]

        source_map = pipeline._build_transcript_source_map(questions)
        assert source_map["episode-2026-01-12"] == "transcript-2026-w04"
        assert source_map["episode-2026-01-22"] == "transcript-2026-w04"


class TestTranscriptAugmentation:
    """Test transcript LLM augmentation path."""

    def test_augment_transcript_questions_increases_count(
        self, temp_output_dir: str, fixtures_path: Path
    ) -> None:
        config = PipelineConfig(
            textbook_path=str(fixtures_path),
            output_path=temp_output_dir,
            augment_questions=False,
            target_questions_per_episode=5,
        )
        pipeline = DataPipeline(config)

        with tempfile.TemporaryDirectory() as transcript_dir:
            raw = Path(transcript_dir) / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            (raw / "2026-01-01T10-00-00Z.txt").write_text(
                "Bob talks about cycle structure and risk.", encoding="utf-8"
            )

            pipeline.transcript_processor = MagicMock()
            pipeline.transcript_processor.transcripts_dir = Path(transcript_dir)
            pipeline.transcript_processor.parse_transcript.return_value = (
                "Bob talks about cycle structure and risk."
            )
            pipeline.transcript_processor.load_episode_summary.return_value = None

            pipeline.transcript_question_gen = MagicMock()
            pipeline.transcript_question_gen.generate_for_episode.return_value = [
                Question(question=f"Generated Q{i}", reference_answer=f"A{i}", source="episode-2026-01-01")
                for i in range(5)
            ]

            base = [
                Question(question="Base Q1", reference_answer="A1", source="episode-2026-01-01"),
                Question(question="Base Q2", reference_answer="A2", source="episode-2026-01-01"),
                Question(question="Base Q3", reference_answer="A3", source="episode-2026-01-01"),
            ]

            augmented = pipeline._augment_transcript_questions(base)
            assert len(augmented) == 5

    def test_augment_transcript_questions_fallback_on_error(
        self, temp_output_dir: str, fixtures_path: Path
    ) -> None:
        config = PipelineConfig(
            textbook_path=str(fixtures_path),
            output_path=temp_output_dir,
            augment_questions=False,
            target_questions_per_episode=5,
        )
        pipeline = DataPipeline(config)

        with tempfile.TemporaryDirectory() as transcript_dir:
            raw = Path(transcript_dir) / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            (raw / "2026-01-01T10-00-00Z.txt").write_text(
                "Bob talks about cycle structure and risk.", encoding="utf-8"
            )

            pipeline.transcript_processor = MagicMock()
            pipeline.transcript_processor.transcripts_dir = Path(transcript_dir)
            pipeline.transcript_processor.parse_transcript.return_value = (
                "Bob talks about cycle structure and risk."
            )
            pipeline.transcript_processor.load_episode_summary.return_value = None

            pipeline.transcript_question_gen = MagicMock()
            pipeline.transcript_question_gen.generate_for_episode.side_effect = RuntimeError(
                "llm failed"
            )

            base = [
                Question(question="Base Q1", reference_answer="A1", source="episode-2026-01-01"),
                Question(question="Base Q2", reference_answer="A2", source="episode-2026-01-01"),
                Question(question="Base Q3", reference_answer="A3", source="episode-2026-01-01"),
            ]

            augmented = pipeline._augment_transcript_questions(base)
            assert len(augmented) == 5


class TestOutputSaving:
    """Test output file saving."""

    def test_saves_jsonl_file(
        self, temp_output_dir: str, fixtures_path: Path
    ) -> None:
        """Saves data as JSONL."""
        config = PipelineConfig(
            textbook_path=str(fixtures_path),
            output_path=temp_output_dir,
            augment_questions=False,
        )
        pipeline = DataPipeline(config)

        test_data = [{"key": "value1"}, {"key": "value2"}]
        pipeline._save_jsonl(test_data, "test.jsonl")

        output_file = Path(temp_output_dir) / "test.jsonl"
        assert output_file.exists()

        with open(output_file) as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert json.loads(lines[0])["key"] == "value1"

    def test_saves_json_file(
        self, temp_output_dir: str, fixtures_path: Path
    ) -> None:
        """Saves data as JSON."""
        config = PipelineConfig(
            textbook_path=str(fixtures_path),
            output_path=temp_output_dir,
            augment_questions=False,
        )
        pipeline = DataPipeline(config)

        test_data = {"items": [1, 2, 3]}
        pipeline._save_json(test_data, "test.json")

        output_file = Path(temp_output_dir) / "test.json"
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)
            assert data["items"] == [1, 2, 3]


class TestFinalAssessment:
    """Test final assessment creation."""

    def test_creates_final_assessment(
        self, temp_output_dir: str, fixtures_path: Path, sample_unit_exam: UnitExam
    ) -> None:
        """Creates final assessment from unit exams."""
        config = PipelineConfig(
            textbook_path=str(fixtures_path),
            output_path=temp_output_dir,
            augment_questions=False,
        )
        pipeline = DataPipeline(config)

        final = pipeline._create_final_assessment([sample_unit_exam])

        assert final["level"] == "final"
        assert final["title"] == "Final Assessment"
        assert len(final["questions"]) >= 1
        assert "source_unit" in final["questions"][0]


class TestCriticalDistinctionsIntegration:
    """Test integration with critical distinctions."""

    def test_includes_critical_distinctions(
        self, temp_output_dir: str, fixtures_path: Path, sample_topic_quiz: TopicQuiz
    ) -> None:
        """Includes critical distinctions when enabled."""
        config = PipelineConfig(
            textbook_path=str(fixtures_path),
            output_path=temp_output_dir,
            augment_questions=False,
            include_critical_distinctions=True,
        )
        pipeline = DataPipeline(config)

        pairs = pipeline._create_preference_pairs([sample_topic_quiz], [], [])
        initial_count = len(pairs)

        # Add critical distinctions
        critical_pairs = pipeline.critical_distinctions.get_all_pairs()
        pairs.extend(critical_pairs)

        assert len(pairs) > initial_count
        assert len(pairs) == initial_count + 10  # 10 critical pairs

    def test_excludes_critical_distinctions_when_disabled(
        self, temp_output_dir: str, fixtures_path: Path
    ) -> None:
        """Excludes critical distinctions when disabled."""
        config = PipelineConfig(
            textbook_path=str(fixtures_path),
            output_path=temp_output_dir,
            augment_questions=False,
            include_critical_distinctions=False,
        )
        pipeline = DataPipeline(config)

        # Config should disable critical distinctions
        assert config.include_critical_distinctions is False


class TestStatsToDict:
    """Test stats serialization."""

    def test_converts_stats_to_dict(
        self, temp_output_dir: str, fixtures_path: Path
    ) -> None:
        """Converts stats to dictionary."""
        config = PipelineConfig(
            textbook_path=str(fixtures_path),
            output_path=temp_output_dir,
            augment_questions=False,
        )
        pipeline = DataPipeline(config)
        pipeline.stats.topics_processed = 5
        pipeline.stats.sft_examples = 50

        stats_dict = pipeline._stats_to_dict()

        assert stats_dict["topics_processed"] == 5
        assert stats_dict["sft_examples"] == 50
        assert "timestamp" in stats_dict


class TestPipelineInitialization:
    """Test pipeline initialization."""

    def test_creates_output_directory(
        self, fixtures_path: Path
    ) -> None:
        """Creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_output = Path(temp_dir) / "new" / "nested" / "dir"

            config = PipelineConfig(
                textbook_path=str(fixtures_path),
                output_path=str(new_output),
                augment_questions=False,
            )
            DataPipeline(config)

            assert new_output.exists()

    def test_initializes_without_question_generator(
        self, temp_output_dir: str, fixtures_path: Path
    ) -> None:
        """Initializes without question generator when augmentation disabled."""
        config = PipelineConfig(
            textbook_path=str(fixtures_path),
            output_path=temp_output_dir,
            augment_questions=False,
        )
        pipeline = DataPipeline(config)

        assert pipeline.question_gen is None

    def test_initializes_with_question_generator(
        self, temp_output_dir: str, fixtures_path: Path
    ) -> None:
        """Initializes with question generator when augmentation enabled."""
        config = PipelineConfig(
            textbook_path=str(fixtures_path),
            output_path=temp_output_dir,
            augment_questions=True,
        )
        pipeline = DataPipeline(config)

        assert pipeline.question_gen is not None
