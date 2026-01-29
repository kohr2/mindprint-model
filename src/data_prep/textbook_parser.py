"""
TextbookParser - Parse Bob Loukas textbook markdown into structured data.

Handles two test formats:
1. Topic tests: ### Question N with **Reference Answer**: blocks
2. Chapter tests: Multiple choice, True/False, Short Answer with <details> blocks
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class Question:
    """A single question with its reference answer."""

    question: str
    reference_answer: str
    question_type: str = "open"  # "open", "multiple_choice", "true_false"
    evaluation_focus: List[str] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    options: Optional[List[str]] = None  # For multiple choice
    correct_option: Optional[str] = None  # For multiple choice
    source: str = ""  # Source identifier (e.g., "episode-2026-01-24" or "unit-01/chapter-01/topic-01")


@dataclass
class TopicQuiz:
    """Quiz data for a single topic."""

    unit: str
    chapter: str
    topic: str
    title: str
    questions: List[Question]

    @property
    def identifier(self) -> str:
        return f"{self.unit}/{self.chapter}/{self.topic}"


@dataclass
class ChapterTest:
    """Test data for a chapter (aggregates topics)."""

    unit: str
    chapter: str
    title: str
    questions: List[Question]

    @property
    def identifier(self) -> str:
        return f"{self.unit}/{self.chapter}"


@dataclass
class UnitExam:
    """Exam data for a unit."""

    unit: str
    title: str
    questions: List[Question]

    @property
    def identifier(self) -> str:
        return self.unit


class TextbookParser:
    """Parse Bob Loukas textbook markdown into structured data."""

    def __init__(self, textbook_path: str):
        """
        Initialize the parser.

        Args:
            textbook_path: Path to the textbook directory containing curriculum.yaml
        """
        self.root = Path(textbook_path)
        self.curriculum = self._load_curriculum()

    def _load_curriculum(self) -> Dict:
        """Load and parse curriculum.yaml."""
        curriculum_path = self.root / "curriculum.yaml"
        if not curriculum_path.exists():
            raise FileNotFoundError(f"curriculum.yaml not found at {curriculum_path}")

        with open(curriculum_path, "r") as f:
            return yaml.safe_load(f)

    def parse_all_topics(self) -> List[TopicQuiz]:
        """Parse all topic test files."""
        topic_quizzes = []
        tests_dir = self.root / "tests"

        if not tests_dir.exists():
            logger.warning(f"Tests directory not found: {tests_dir}")
            return topic_quizzes

        # Walk through all unit directories
        for unit_dir in sorted(tests_dir.iterdir()):
            if not unit_dir.is_dir() or not unit_dir.name.startswith("unit-"):
                continue

            unit_name = unit_dir.name

            # Walk through chapter directories within unit
            for chapter_dir in sorted(unit_dir.iterdir()):
                if not chapter_dir.is_dir() or not chapter_dir.name.startswith("chapter-"):
                    continue

                chapter_name = chapter_dir.name

                # Find topic test files
                for test_file in sorted(chapter_dir.glob("topic-*-test.md")):
                    topic_name = test_file.stem.replace("-test", "")

                    try:
                        questions = self._parse_topic_test(test_file)
                        if questions:
                            topic_quiz = TopicQuiz(
                                unit=unit_name,
                                chapter=chapter_name,
                                topic=topic_name,
                                title=self._get_topic_title(unit_name, chapter_name, topic_name),
                                questions=questions,
                            )
                            topic_quizzes.append(topic_quiz)
                            logger.info(
                                f"Parsed {len(questions)} questions from {topic_quiz.identifier}"
                            )
                    except Exception as e:
                        logger.error(f"Error parsing {test_file}: {e}")

        return topic_quizzes

    def parse_all_chapters(self) -> List[ChapterTest]:
        """Parse all chapter test files."""
        chapter_tests = []
        tests_dir = self.root / "tests"

        if not tests_dir.exists():
            return chapter_tests

        for unit_dir in sorted(tests_dir.iterdir()):
            if not unit_dir.is_dir() or not unit_dir.name.startswith("unit-"):
                continue

            unit_name = unit_dir.name

            # Find chapter test files (directly in unit directory)
            for test_file in sorted(unit_dir.glob("chapter-*-test.md")):
                chapter_name = test_file.stem.replace("-test", "")

                try:
                    questions = self._parse_chapter_test(test_file)
                    if questions:
                        chapter_test = ChapterTest(
                            unit=unit_name,
                            chapter=chapter_name,
                            title=self._get_chapter_title(unit_name, chapter_name),
                            questions=questions,
                        )
                        chapter_tests.append(chapter_test)
                        logger.info(
                            f"Parsed {len(questions)} questions from {chapter_test.identifier}"
                        )
                except Exception as e:
                    logger.error(f"Error parsing {test_file}: {e}")

        return chapter_tests

    def parse_all_units(self) -> List[UnitExam]:
        """Parse all unit exam files."""
        unit_exams = []
        tests_dir = self.root / "tests"

        if not tests_dir.exists():
            return unit_exams

        for unit_dir in sorted(tests_dir.iterdir()):
            if not unit_dir.is_dir() or not unit_dir.name.startswith("unit-"):
                continue

            unit_name = unit_dir.name
            test_file = unit_dir / f"{unit_name}-test.md"

            if test_file.exists():
                try:
                    questions = self._parse_chapter_test(test_file)  # Same format as chapter
                    if questions:
                        unit_exam = UnitExam(
                            unit=unit_name,
                            title=self._get_unit_title(unit_name),
                            questions=questions,
                        )
                        unit_exams.append(unit_exam)
                        logger.info(
                            f"Parsed {len(questions)} questions from {unit_exam.identifier}"
                        )
                except Exception as e:
                    logger.error(f"Error parsing {test_file}: {e}")

        return unit_exams

    def _parse_topic_test(self, test_path: Path) -> List[Question]:
        """
        Parse a topic test file.

        Format:
        ### Question N
        <question text>

        **Reference Answer**:
        <answer text>

        **Evaluation Focus**:
        - [ ] item1
        - [ ] item2

        **Key Concepts Tested**: `concept1`, `concept2`
        """
        content = test_path.read_text(encoding="utf-8")
        questions = []

        # Split by question markers
        question_blocks = re.split(r"### Question \d+", content)[1:]

        for block in question_blocks:
            question = self._parse_topic_question_block(block)
            if question:
                questions.append(question)

        return questions

    def _parse_topic_question_block(self, block: str) -> Optional[Question]:
        """Parse a single question block from a topic test."""
        lines = block.strip().split("\n")

        question_text = ""
        reference_answer = ""
        evaluation_focus = []
        key_concepts = []

        in_answer = False
        in_focus = False
        answer_lines = []

        for line in lines:
            # Check for section markers
            if line.startswith("**Reference Answer**"):
                in_answer = True
                in_focus = False
                # Get any text after the colon on the same line
                after_colon = line.split(":", 1)
                if len(after_colon) > 1 and after_colon[1].strip():
                    answer_lines.append(after_colon[1].strip())
                continue
            elif line.startswith("**Evaluation Focus**"):
                in_answer = False
                in_focus = True
                continue
            elif line.startswith("**Key Concepts Tested**"):
                in_answer = False
                in_focus = False
                # Extract concepts from backticks
                concepts = re.findall(r"`([^`]+)`", line)
                key_concepts.extend(concepts)
                continue
            elif line.startswith("---"):
                # End of question block
                break

            # Collect content based on current section
            if in_answer:
                answer_lines.append(line)
            elif in_focus:
                # Parse checklist items
                match = re.match(r"- \[[ x]\] (.+)", line)
                if match:
                    evaluation_focus.append(match.group(1))
            elif not in_answer and not in_focus and not question_text:
                # First non-empty line is the question
                stripped = line.strip()
                if stripped and not stripped.startswith("**"):
                    question_text = stripped

        reference_answer = "\n".join(answer_lines).strip()

        if not question_text or not reference_answer:
            return None

        return Question(
            question=question_text,
            reference_answer=reference_answer,
            question_type="open",
            evaluation_focus=evaluation_focus,
            key_concepts=key_concepts,
        )

    def _parse_chapter_test(self, test_path: Path) -> List[Question]:
        """
        Parse a chapter/unit test file.

        Format uses <details><summary>Answer</summary> blocks.
        Supports multiple choice, true/false, and short answer.
        """
        content = test_path.read_text(encoding="utf-8")
        questions = []

        # Split by question markers
        question_blocks = re.split(r"### Question \d+", content)[1:]

        for block in question_blocks:
            question = self._parse_chapter_question_block(block)
            if question:
                questions.append(question)

        return questions

    def _parse_chapter_question_block(self, block: str) -> Optional[Question]:
        """Parse a single question block from a chapter test."""
        # Determine question type
        is_tf = "**True or False:**" in block or "True or False:" in block
        has_options = bool(re.search(r"^[A-D]\)", block, re.MULTILINE))

        # Extract question text
        lines = block.strip().split("\n")
        question_lines = []
        options = []
        in_details = False
        answer_lines = []

        for i, line in enumerate(lines):
            # Check for details block (answer)
            if "<details>" in line:
                in_details = True
                continue
            if "</details>" in line:
                in_details = False
                continue
            if "<summary>" in line:
                continue

            if in_details:
                # Inside answer block
                answer_lines.append(line)
            else:
                # Outside answer block
                if line.startswith("---"):
                    continue

                # Check for multiple choice options
                option_match = re.match(r"^([A-D])\) (.+)", line)
                if option_match:
                    options.append(f"{option_match.group(1)}) {option_match.group(2)}")
                else:
                    # Part of question text
                    stripped = line.strip()
                    if stripped and not stripped.startswith("**Topics") and not stripped.startswith("**Instructions"):
                        question_lines.append(stripped)

        question_text = " ".join(question_lines).strip()
        answer_text = "\n".join(answer_lines).strip()

        # Clean up question text
        question_text = re.sub(r"\*\*True or False:\*\*\s*", "", question_text)
        question_text = re.sub(r"\*\*", "", question_text)
        question_text = question_text.strip()

        if not question_text or not answer_text:
            return None

        # Determine question type and extract correct option
        if is_tf:
            question_type = "true_false"
            correct_option = None
        elif options:
            question_type = "multiple_choice"
            # Extract correct option from answer
            correct_match = re.search(r"\*\*([A-D])\)", answer_text)
            correct_option = correct_match.group(1) if correct_match else None
        else:
            question_type = "open"
            correct_option = None

        return Question(
            question=question_text,
            reference_answer=answer_text,
            question_type=question_type,
            options=options if options else None,
            correct_option=correct_option,
        )

    def _get_topic_title(self, unit: str, chapter: str, topic: str) -> str:
        """Get topic title from curriculum."""
        for curriculum_unit in self.curriculum.get("units", []):
            if unit in curriculum_unit.get("id", ""):
                for curriculum_chapter in curriculum_unit.get("chapters", []):
                    if chapter in curriculum_chapter.get("id", ""):
                        for curriculum_topic in curriculum_chapter.get("topics", []):
                            if topic in curriculum_topic.get("id", ""):
                                return curriculum_topic.get("title", topic)
        return topic

    def _get_chapter_title(self, unit: str, chapter: str) -> str:
        """Get chapter title from curriculum."""
        for curriculum_unit in self.curriculum.get("units", []):
            if unit in curriculum_unit.get("id", ""):
                for curriculum_chapter in curriculum_unit.get("chapters", []):
                    if chapter in curriculum_chapter.get("id", ""):
                        return curriculum_chapter.get("title", chapter)
        return chapter

    def _get_unit_title(self, unit: str) -> str:
        """Get unit title from curriculum."""
        for curriculum_unit in self.curriculum.get("units", []):
            if unit in curriculum_unit.get("id", ""):
                return curriculum_unit.get("title", unit)
        return unit

    def get_topic_content(self, unit: str, chapter: str, topic: str) -> Optional[str]:
        """
        Get the content markdown for a topic.

        Used for question generation context.
        """
        units_dir = self.root / "units"

        # Find matching unit directory
        for unit_dir in units_dir.iterdir():
            if not unit_dir.is_dir():
                continue
            if unit not in unit_dir.name:
                continue

            # Find matching chapter directory
            for chapter_dir in unit_dir.iterdir():
                if not chapter_dir.is_dir():
                    continue
                if chapter not in chapter_dir.name:
                    continue

                # Find matching topic file
                for topic_file in chapter_dir.glob("*.md"):
                    if topic in topic_file.name and "test" not in topic_file.name:
                        return topic_file.read_text(encoding="utf-8")

        return None

    def get_style_guide(self) -> Optional[str]:
        """Get Bob's style guide content."""
        style_guide_path = self.root / "bob-style-guide.md"
        if style_guide_path.exists():
            return style_guide_path.read_text(encoding="utf-8")
        return None

    def get_statistics(self) -> Dict:
        """Get statistics about the parsed content."""
        topic_quizzes = self.parse_all_topics()
        chapter_tests = self.parse_all_chapters()
        unit_exams = self.parse_all_units()

        total_topic_questions = sum(len(q.questions) for q in topic_quizzes)
        total_chapter_questions = sum(len(t.questions) for t in chapter_tests)
        total_unit_questions = sum(len(e.questions) for e in unit_exams)

        return {
            "topics": len(topic_quizzes),
            "chapters": len(chapter_tests),
            "units": len(unit_exams),
            "topic_questions": total_topic_questions,
            "chapter_questions": total_chapter_questions,
            "unit_questions": total_unit_questions,
            "total_questions": total_topic_questions + total_chapter_questions + total_unit_questions,
            "avg_questions_per_topic": total_topic_questions / len(topic_quizzes) if topic_quizzes else 0,
        }
