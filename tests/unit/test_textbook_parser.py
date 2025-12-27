"""
Tests for TextbookParser - parsing Bob Loukas textbook markdown into structured data.

Tests cover:
- Topic test parsing (### Question N with **Reference Answer**: blocks)
- Chapter test parsing (multiple choice, T/F with <details> blocks)
- Curriculum loading
- Statistics generation
"""

import pytest
from pathlib import Path
from src.data_prep.textbook_parser import (
    TextbookParser,
    Question,
    TopicQuiz,
    ChapterTest,
)


@pytest.fixture
def fixtures_path() -> Path:
    """Path to test fixtures."""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def sample_topic_test_content() -> str:
    """Sample topic test markdown content."""
    return """# Unit 01, Chapter 01, Topic 01 Test

**Source:** Sample Topic

---

## Quiz: Sample Topic Quiz

### Question 1
What is the fundamental difference between a gambler and a trader?

**Reference Answer**:
Look, the difference isn't about how much analysis you do. The difference is **systematic discipline versus hope-driven action**.

A gambler is outcome-focused. A trader has written rules.

**Evaluation Focus**:
- [ ] Factual accuracy
- [ ] Voice fidelity

**Key Concepts Tested**: `gambler_definition`, `trader_definition`

---

### Question 2
Why is discipline more important than analysis?

**Reference Answer**:
Here's what I've observed over 25 years: brilliant analysis with poor execution loses money. Average analysis with disciplined execution makes money.

The market doesn't reward intelligence. It rewards patience and consistency.

**Evaluation Focus**:
- [ ] Understanding of discipline
- [ ] Practical wisdom

**Key Concepts Tested**: `discipline`, `execution`

---
"""


@pytest.fixture
def sample_chapter_test_content() -> str:
    """Sample chapter test markdown content with multiple choice and T/F."""
    return """# Chapter 01 Test

**Topics Covered:** Topic 01, Topic 02

---

### Question 1
**True or False:** A trader who uses technical analysis is always more successful than a gambler.

<details>
<summary>Answer</summary>

**False.** Technical analysis doesn't make you a trader - systematic execution does. A gambler with charts is still a gambler.

</details>

---

### Question 2
Which of the following best describes an investor's approach to a 30% drawdown?

A) Panic and sell to preserve capital
B) Double down because "it has to recover"
C) Accept it as normal volatility within their thesis
D) Hedge immediately with options

<details>
<summary>Answer</summary>

**C)** An investor expects volatility and has already accounted for it in their position sizing. They don't react emotionally to drawdowns within their expected range.

</details>

---
"""


class TestTopicQuestionParsing:
    """Test parsing of topic test questions."""

    def test_parse_topic_question_extracts_question_text(
        self, fixtures_path: Path
    ) -> None:
        """Parse correctly extracts the question text."""
        test_file = fixtures_path / "sample_topic_test.md"
        parser = TextbookParser(str(fixtures_path))

        questions = parser._parse_topic_test(test_file)

        assert len(questions) == 2
        assert "fundamental difference" in questions[0].question
        assert "discipline more important" in questions[1].question

    def test_parse_topic_question_extracts_reference_answer(
        self, fixtures_path: Path
    ) -> None:
        """Parse correctly extracts the reference answer."""
        test_file = fixtures_path / "sample_topic_test.md"
        parser = TextbookParser(str(fixtures_path))

        questions = parser._parse_topic_test(test_file)

        assert "Look, the difference isn't" in questions[0].reference_answer
        assert "systematic discipline" in questions[0].reference_answer
        assert "Here's what I've observed" in questions[1].reference_answer

    def test_parse_topic_question_extracts_key_concepts(
        self, fixtures_path: Path
    ) -> None:
        """Parse correctly extracts key concepts from backticks."""
        test_file = fixtures_path / "sample_topic_test.md"
        parser = TextbookParser(str(fixtures_path))

        questions = parser._parse_topic_test(test_file)

        assert "gambler_definition" in questions[0].key_concepts
        assert "trader_definition" in questions[0].key_concepts
        assert "discipline" in questions[1].key_concepts

    def test_parse_topic_question_extracts_evaluation_focus(
        self, fixtures_path: Path
    ) -> None:
        """Parse correctly extracts evaluation focus items."""
        test_file = fixtures_path / "sample_topic_test.md"
        parser = TextbookParser(str(fixtures_path))

        questions = parser._parse_topic_test(test_file)

        assert "Factual accuracy" in questions[0].evaluation_focus
        assert "Voice fidelity" in questions[0].evaluation_focus

    def test_parse_topic_question_sets_type_to_open(
        self, fixtures_path: Path
    ) -> None:
        """Topic questions should have type 'open'."""
        test_file = fixtures_path / "sample_topic_test.md"
        parser = TextbookParser(str(fixtures_path))

        questions = parser._parse_topic_test(test_file)

        for q in questions:
            assert q.question_type == "open"


class TestChapterQuestionParsing:
    """Test parsing of chapter test questions (multiple choice, T/F)."""

    def test_parse_chapter_question_true_false(
        self, fixtures_path: Path
    ) -> None:
        """Parse correctly identifies true/false questions."""
        test_file = fixtures_path / "sample_chapter_test.md"
        parser = TextbookParser(str(fixtures_path))

        questions = parser._parse_chapter_test(test_file)

        # First question is T/F
        assert questions[0].question_type == "true_false"
        assert "technical analysis" in questions[0].question.lower()

    def test_parse_chapter_question_multiple_choice(
        self, fixtures_path: Path
    ) -> None:
        """Parse correctly identifies multiple choice questions."""
        test_file = fixtures_path / "sample_chapter_test.md"
        parser = TextbookParser(str(fixtures_path))

        questions = parser._parse_chapter_test(test_file)

        # Second question is multiple choice
        assert questions[1].question_type == "multiple_choice"
        assert questions[1].options is not None
        assert len(questions[1].options) == 4

    def test_parse_chapter_question_extracts_correct_option(
        self, fixtures_path: Path
    ) -> None:
        """Parse correctly extracts the correct option letter."""
        test_file = fixtures_path / "sample_chapter_test.md"
        parser = TextbookParser(str(fixtures_path))

        questions = parser._parse_chapter_test(test_file)

        # Second question should have C as correct
        assert questions[1].correct_option == "C"

    def test_parse_chapter_question_extracts_answer_from_details(
        self, fixtures_path: Path
    ) -> None:
        """Parse correctly extracts answer from <details> block."""
        test_file = fixtures_path / "sample_chapter_test.md"
        parser = TextbookParser(str(fixtures_path))

        questions = parser._parse_chapter_test(test_file)

        assert "False" in questions[0].reference_answer
        assert "systematic execution" in questions[0].reference_answer


class TestCurriculumLoading:
    """Test curriculum.yaml loading."""

    def test_load_curriculum_parses_yaml(self, fixtures_path: Path) -> None:
        """Curriculum is loaded and parsed correctly."""
        parser = TextbookParser(str(fixtures_path))

        assert "units" in parser.curriculum
        assert len(parser.curriculum["units"]) == 1

    def test_get_topic_title_from_curriculum(self, fixtures_path: Path) -> None:
        """Get topic title correctly from curriculum."""
        parser = TextbookParser(str(fixtures_path))

        title = parser._get_topic_title("unit-01", "chapter-01", "topic-01")

        assert title == "The Three Types"

    def test_get_chapter_title_from_curriculum(self, fixtures_path: Path) -> None:
        """Get chapter title correctly from curriculum."""
        parser = TextbookParser(str(fixtures_path))

        title = parser._get_chapter_title("unit-01", "chapter-01")

        assert title == "Market Participants"


class TestQuestionDataclass:
    """Test Question dataclass behavior."""

    def test_question_default_type_is_open(self) -> None:
        """Default question type is 'open'."""
        q = Question(
            question="Test question?",
            reference_answer="Test answer.",
        )

        assert q.question_type == "open"

    def test_question_empty_lists_by_default(self) -> None:
        """Evaluation focus and key concepts default to empty lists."""
        q = Question(
            question="Test question?",
            reference_answer="Test answer.",
        )

        assert q.evaluation_focus == []
        assert q.key_concepts == []


class TestTopicQuizDataclass:
    """Test TopicQuiz dataclass behavior."""

    def test_topic_quiz_identifier(self) -> None:
        """Identifier property returns correct format."""
        quiz = TopicQuiz(
            unit="unit-01",
            chapter="chapter-01",
            topic="topic-01",
            title="The Three Types",
            questions=[],
        )

        assert quiz.identifier == "unit-01/chapter-01/topic-01"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_question_block_returns_none(self, fixtures_path: Path) -> None:
        """Empty question block is skipped."""
        parser = TextbookParser(str(fixtures_path))

        result = parser._parse_topic_question_block("")

        assert result is None

    def test_question_without_answer_returns_none(
        self, fixtures_path: Path
    ) -> None:
        """Question without reference answer is skipped."""
        parser = TextbookParser(str(fixtures_path))
        block = "What is the question?\n\n**Evaluation Focus**:\n- [ ] Test"

        result = parser._parse_topic_question_block(block)

        assert result is None

    def test_missing_curriculum_raises_error(self, tmp_path: Path) -> None:
        """Missing curriculum.yaml raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            TextbookParser(str(tmp_path))
