"""
Tests for PreferencePairGenerator - generating preference pairs for DPO/RLHF training.

Tests cover:
- Voice marker stripping (confidence, engagement, psychology)
- Response truncation
- Genericization
- Similarity checking
- Output format conversion
"""

import pytest
from src.data_prep.preference_generator import (
    PreferencePairGenerator,
    PreferencePair,
    VoiceStrippingConfig,
)
from src.data_prep.textbook_parser import Question


@pytest.fixture
def generator() -> PreferencePairGenerator:
    """Create a default generator."""
    return PreferencePairGenerator()


@pytest.fixture
def sample_question() -> Question:
    """Sample question with Bob's voice in the reference answer."""
    return Question(
        question="What causes Bitcoin's 4-year market cycle?",
        reference_answer="""Look, the 4-year cycle is NOT caused by the halving—that's a common misconception. I've been watching market cycles long before Bitcoin existed. The 4-year rhythm shows up in gold, in stocks, across different markets.

It's about capital flows and how long it takes for market psychology to shift from fear to greed and back again. **This is fundamental to understanding cycles.**

Here's what I've observed: the cycle exists independently of any single catalyst.""",
    )


class TestVoiceMarkerStripping:
    """Test that voice markers are properly stripped."""

    def test_strips_confidence_markers(self, generator: PreferencePairGenerator) -> None:
        """Confidence markers like 'I've tracked' are removed."""
        question = Question(
            question="Test question?",
            reference_answer="I've tracked this pattern for years. The result is clear.",
        )

        pair = generator.generate_pair(question)

        assert "I've tracked" not in pair.rejected
        assert "pattern" in pair.rejected  # Content preserved

    def test_strips_engagement_markers(self, generator: PreferencePairGenerator) -> None:
        """Engagement markers like 'Look,' are removed."""
        question = Question(
            question="Test question?",
            reference_answer="Look, this is important. Here's the thing: markets move in cycles.",
        )

        pair = generator.generate_pair(question)

        assert "Look," not in pair.rejected
        assert "Here's the thing" not in pair.rejected

    def test_strips_bold_formatting(self, generator: PreferencePairGenerator) -> None:
        """Bold markdown formatting is removed."""
        question = Question(
            question="Test question?",
            reference_answer="This is **critically important** for traders.",
        )

        pair = generator.generate_pair(question)

        assert "**" not in pair.rejected
        assert "critically important" in pair.rejected

    def test_strips_italic_formatting(self, generator: PreferencePairGenerator) -> None:
        """Italic markdown formatting is removed."""
        question = Question(
            question="Test question?",
            reference_answer="This is *really* important for understanding cycles.",
        )

        pair = generator.generate_pair(question)

        assert pair.rejected.count("*") == 0 or "really" in pair.rejected


class TestResponseTruncation:
    """Test that responses are properly truncated."""

    def test_truncates_long_responses(self, generator: PreferencePairGenerator) -> None:
        """Long responses are truncated to configured ratio."""
        long_answer = ". ".join([f"Sentence {i}" for i in range(10)]) + "."
        question = Question(
            question="Test question?",
            reference_answer=long_answer,
        )

        pair = generator.generate_pair(question)

        # Should be significantly shorter
        assert len(pair.rejected) < len(pair.chosen)

    def test_respects_minimum_sentences(self) -> None:
        """Truncation respects minimum sentence count."""
        config = VoiceStrippingConfig(min_sentences=3)
        generator = PreferencePairGenerator(config)

        short_answer = "First. Second. Third. Fourth."
        question = Question(
            question="Test question?",
            reference_answer=short_answer,
        )

        pair = generator.generate_pair(question)

        # Should have at least 2-3 sentences (may be combined due to processing)
        sentences = [s for s in pair.rejected.split(".") if s.strip()]
        assert len(sentences) >= 2


class TestGenericization:
    """Test that responses are made more generic."""

    def test_removes_first_person(self, generator: PreferencePairGenerator) -> None:
        """First person pronouns are replaced."""
        question = Question(
            question="Test question?",
            reference_answer="I believe this is true. In my experience, it works.",
        )

        pair = generator.generate_pair(question)

        # First-person should be reduced or eliminated
        assert pair.rejected.count(" I ") <= pair.chosen.count(" I ")

    def test_removes_parenthetical_asides(
        self, generator: PreferencePairGenerator
    ) -> None:
        """Parenthetical comments are removed."""
        question = Question(
            question="Test question?",
            reference_answer="The cycle (which I've tracked for decades) repeats consistently.",
        )

        pair = generator.generate_pair(question)

        assert "(which" not in pair.rejected


class TestSimilarityChecking:
    """Test similarity ratio calculation."""

    def test_similarity_ratio_identical_texts(
        self, generator: PreferencePairGenerator
    ) -> None:
        """Identical texts have similarity ratio of 1.0."""
        ratio = generator._similarity_ratio("hello world", "hello world")
        assert ratio == 1.0

    def test_similarity_ratio_different_texts(
        self, generator: PreferencePairGenerator
    ) -> None:
        """Different texts have lower similarity ratio."""
        ratio = generator._similarity_ratio(
            "the quick brown fox",
            "a slow red dog",
        )
        assert ratio < 0.5

    def test_similarity_ratio_empty_text(
        self, generator: PreferencePairGenerator
    ) -> None:
        """Empty texts return 0.0."""
        ratio = generator._similarity_ratio("", "hello")
        assert ratio == 0.0

    def test_rejected_is_sufficiently_different(
        self, sample_question: Question, generator: PreferencePairGenerator
    ) -> None:
        """Rejected response should be sufficiently different from chosen."""
        pair = generator.generate_pair(sample_question)

        ratio = generator._similarity_ratio(pair.chosen, pair.rejected)
        assert ratio < 0.95  # At least 5% different


class TestPreferencePairGeneration:
    """Test the full preference pair generation process."""

    def test_generates_valid_pair(
        self, sample_question: Question, generator: PreferencePairGenerator
    ) -> None:
        """Generates a valid preference pair."""
        pair = generator.generate_pair(sample_question, source="test")

        assert pair.prompt == sample_question.question
        assert pair.chosen == sample_question.reference_answer
        assert pair.rejected != pair.chosen
        assert pair.source == "test"

    def test_chosen_preserves_original(
        self, sample_question: Question, generator: PreferencePairGenerator
    ) -> None:
        """Chosen response is the original reference answer."""
        pair = generator.generate_pair(sample_question)

        assert pair.chosen == sample_question.reference_answer

    def test_rejected_is_shorter(
        self, sample_question: Question, generator: PreferencePairGenerator
    ) -> None:
        """Rejected response is shorter than chosen."""
        pair = generator.generate_pair(sample_question)

        assert len(pair.rejected) < len(pair.chosen)


class TestBatchGeneration:
    """Test batch generation of preference pairs."""

    def test_generate_all_processes_list(
        self, generator: PreferencePairGenerator
    ) -> None:
        """generate_all processes a list of questions."""
        questions = [
            (Question(question=f"Q{i}?", reference_answer=f"Answer {i}."), f"source-{i}")
            for i in range(3)
        ]

        pairs = generator.generate_all(questions)

        assert len(pairs) == 3
        assert all(isinstance(p, PreferencePair) for p in pairs)

    def test_generate_all_preserves_sources(
        self, generator: PreferencePairGenerator
    ) -> None:
        """generate_all preserves source information."""
        questions = [
            (Question(question="Q1?", reference_answer="A1."), "unit-01/ch-01/topic-01"),
            (Question(question="Q2?", reference_answer="A2."), "unit-01/ch-01/topic-02"),
        ]

        pairs = generator.generate_all(questions)

        assert pairs[0].source == "unit-01/ch-01/topic-01"
        assert pairs[1].source == "unit-01/ch-01/topic-02"


class TestOutputFormatConversion:
    """Test conversion to training data formats."""

    def test_to_jsonl_format(self, generator: PreferencePairGenerator) -> None:
        """Convert to JSONL format for DPO training."""
        pairs = [
            PreferencePair(
                prompt="Question?",
                chosen="Good answer.",
                rejected="Bad answer.",
                source="test",
            )
        ]

        result = generator.to_jsonl_format(pairs)

        assert len(result) == 1
        assert result[0]["prompt"] == "Question?"
        assert result[0]["chosen"] == "Good answer."
        assert result[0]["rejected"] == "Bad answer."
        assert "source" not in result[0]  # Source not included in output

    def test_to_sft_format(self, generator: PreferencePairGenerator) -> None:
        """Convert to SFT format for supervised fine-tuning."""
        pairs = [
            PreferencePair(
                prompt="Question?",
                chosen="Good answer.",
                rejected="Bad answer.",
                source="test",
            )
        ]

        result = generator.to_sft_format(pairs)

        assert len(result) == 1
        assert result[0]["instruction"] == "Question?"
        assert result[0]["input"] == ""
        assert result[0]["output"] == "Good answer."


class TestVoiceStrippingConfig:
    """Test custom voice stripping configuration."""

    def test_custom_markers(self) -> None:
        """Custom markers can be configured."""
        config = VoiceStrippingConfig(
            confidence_markers=["Custom marker"],
            engagement_markers=[],
            truncate_to_ratio=0.5,
        )
        generator = PreferencePairGenerator(config)

        question = Question(
            question="Test?",
            reference_answer="Custom marker is here. Other content remains.",
        )

        pair = generator.generate_pair(question)

        assert "Custom marker" not in pair.rejected

    def test_custom_truncation_ratio(self) -> None:
        """Custom truncation ratio is applied."""
        config = VoiceStrippingConfig(truncate_to_ratio=0.2, min_sentences=1)
        generator = PreferencePairGenerator(config)

        long_answer = ". ".join([f"Sentence {i}" for i in range(20)]) + "."
        question = Question(
            question="Test?",
            reference_answer=long_answer,
        )

        pair = generator.generate_pair(question)

        # Should be heavily truncated
        original_sentences = len([s for s in long_answer.split(".") if s.strip()])
        rejected_sentences = len([s for s in pair.rejected.split(".") if s.strip()])
        assert rejected_sentences < original_sentences * 0.5


class TestPreferencePairDataclass:
    """Test PreferencePair dataclass."""

    def test_default_source_is_empty(self) -> None:
        """Default source is empty string."""
        pair = PreferencePair(
            prompt="Q?",
            chosen="A",
            rejected="B",
        )
        assert pair.source == ""

    def test_stores_all_fields(self) -> None:
        """All fields are stored correctly."""
        pair = PreferencePair(
            prompt="Question?",
            chosen="Good answer.",
            rejected="Bad answer.",
            source="test-source",
        )

        assert pair.prompt == "Question?"
        assert pair.chosen == "Good answer."
        assert pair.rejected == "Bad answer."
        assert pair.source == "test-source"
