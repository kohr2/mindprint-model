"""
DataPipeline - Orchestrates the full data preparation flow.

Coordinates parsing, question generation, preference pair creation,
and output generation for the Bob Loukas mindprint training data.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import logging
from datetime import datetime

from .textbook_parser import TextbookParser, Question, TopicQuiz, ChapterTest, UnitExam
from .question_generator import QuestionGenerator, GenerationConfig
from .preference_generator import PreferencePairGenerator, PreferencePair
from .critical_distinctions import CriticalDistinctions

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline."""

    textbook_path: str
    output_path: str
    target_questions_per_topic: int = 10
    augment_questions: bool = True
    include_critical_distinctions: bool = True
    api_key: Optional[str] = None  # For question generation


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""

    topics_processed: int = 0
    chapters_processed: int = 0
    units_processed: int = 0
    total_questions_before: int = 0
    total_questions_after: int = 0
    questions_generated: int = 0
    sft_examples: int = 0
    preference_pairs: int = 0
    critical_pairs: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DataPipeline:
    """Complete data preparation pipeline for mindprint training."""

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.output_path = Path(config.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.parser = TextbookParser(config.textbook_path)

        if config.augment_questions:
            gen_config = GenerationConfig(
                target_questions=config.target_questions_per_topic
            )
            self.question_gen = QuestionGenerator(
                parser=self.parser,
                config=gen_config,
                api_key=config.api_key,
            )
        else:
            self.question_gen = None

        self.preference_gen = PreferencePairGenerator()
        self.critical_distinctions = CriticalDistinctions()

        self.stats = PipelineStats()

    def run(self) -> PipelineStats:
        """
        Execute the full data preparation pipeline.

        Returns:
            Statistics from the pipeline run
        """
        logger.info("Starting data preparation pipeline")
        logger.info(f"Textbook path: {self.config.textbook_path}")
        logger.info(f"Output path: {self.output_path}")

        # Phase 1: Parse all test files
        logger.info("Phase 1: Parsing test files")
        topic_quizzes = self.parser.parse_all_topics()
        chapter_tests = self.parser.parse_all_chapters()
        unit_exams = self.parser.parse_all_units()

        self.stats.topics_processed = len(topic_quizzes)
        self.stats.chapters_processed = len(chapter_tests)
        self.stats.units_processed = len(unit_exams)
        self.stats.total_questions_before = sum(
            len(q.questions) for q in topic_quizzes
        )

        logger.info(f"Parsed {len(topic_quizzes)} topics, {len(chapter_tests)} chapters, {len(unit_exams)} units")
        logger.info(f"Total topic questions: {self.stats.total_questions_before}")

        # Phase 2: Augment questions (if enabled)
        if self.config.augment_questions and self.question_gen:
            logger.info("Phase 2: Augmenting questions to target count")
            topic_quizzes = self.question_gen.augment_all(topic_quizzes)

            self.stats.total_questions_after = sum(
                len(q.questions) for q in topic_quizzes
            )
            self.stats.questions_generated = (
                self.stats.total_questions_after - self.stats.total_questions_before
            )
            logger.info(f"Generated {self.stats.questions_generated} new questions")
        else:
            logger.info("Phase 2: Skipping question augmentation")
            self.stats.total_questions_after = self.stats.total_questions_before

        # Phase 3: Generate SFT data
        logger.info("Phase 3: Generating SFT training data")
        sft_data = self._create_sft_data(topic_quizzes, chapter_tests, unit_exams)
        self.stats.sft_examples = len(sft_data)
        logger.info(f"Generated {len(sft_data)} SFT examples")

        # Phase 4: Generate preference pairs
        logger.info("Phase 4: Generating preference pairs")
        preference_pairs = self._create_preference_pairs(
            topic_quizzes, chapter_tests, unit_exams
        )
        self.stats.preference_pairs = len(preference_pairs)
        logger.info(f"Generated {len(preference_pairs)} preference pairs")

        # Phase 5: Add critical distinctions
        if self.config.include_critical_distinctions:
            logger.info("Phase 5: Adding critical distinction pairs")
            critical_pairs = self.critical_distinctions.get_all_pairs()
            self.stats.critical_pairs = len(critical_pairs)
            preference_pairs.extend(critical_pairs)
            logger.info(f"Added {len(critical_pairs)} critical distinction pairs")
        else:
            logger.info("Phase 5: Skipping critical distinctions")

        # Phase 6: Save all outputs
        logger.info("Phase 6: Saving outputs")
        self._save_all(
            sft_data=sft_data,
            preference_pairs=preference_pairs,
            topic_quizzes=topic_quizzes,
            chapter_tests=chapter_tests,
            unit_exams=unit_exams,
        )

        logger.info("Pipeline complete!")
        self._print_summary()

        return self.stats

    def _create_sft_data(
        self,
        topic_quizzes: List[TopicQuiz],
        chapter_tests: List[ChapterTest],
        unit_exams: List[UnitExam],
    ) -> List[Dict]:
        """Create SFT training data from all questions."""
        sft_data = []

        # From topic quizzes
        for quiz in topic_quizzes:
            for question in quiz.questions:
                sft_data.append({
                    "instruction": question.question,
                    "input": "",
                    "output": question.reference_answer,
                    "source": quiz.identifier,
                })

        # From chapter tests (short answer only, MC/TF don't work well for SFT)
        for test in chapter_tests:
            for question in test.questions:
                if question.question_type == "open":
                    sft_data.append({
                        "instruction": question.question,
                        "input": "",
                        "output": question.reference_answer,
                        "source": test.identifier,
                    })

        # From unit exams
        for exam in unit_exams:
            for question in exam.questions:
                if question.question_type == "open":
                    sft_data.append({
                        "instruction": question.question,
                        "input": "",
                        "output": question.reference_answer,
                        "source": exam.identifier,
                    })

        return sft_data

    def _create_preference_pairs(
        self,
        topic_quizzes: List[TopicQuiz],
        chapter_tests: List[ChapterTest],
        unit_exams: List[UnitExam],
    ) -> List[PreferencePair]:
        """Create preference pairs from all questions."""
        all_questions: List[Tuple[Question, str]] = []

        # From topic quizzes
        for quiz in topic_quizzes:
            for question in quiz.questions:
                all_questions.append((question, quiz.identifier))

        # From chapter tests (open-ended only)
        for test in chapter_tests:
            for question in test.questions:
                if question.question_type == "open":
                    all_questions.append((question, test.identifier))

        # From unit exams
        for exam in unit_exams:
            for question in exam.questions:
                if question.question_type == "open":
                    all_questions.append((question, exam.identifier))

        return self.preference_gen.generate_all(all_questions)

    def _save_all(
        self,
        sft_data: List[Dict],
        preference_pairs: List[PreferencePair],
        topic_quizzes: List[TopicQuiz],
        chapter_tests: List[ChapterTest],
        unit_exams: List[UnitExam],
    ):
        """Save all output files."""
        # SFT data
        self._save_jsonl(sft_data, "sft_data.jsonl")

        # Preference pairs
        preference_data = [
            {
                "prompt": p.prompt,
                "chosen": p.chosen,
                "rejected": p.rejected,
            }
            for p in preference_pairs
        ]
        self._save_jsonl(preference_data, "preference_data.jsonl")

        # Quiz data (topic-level)
        quiz_data = [
            {
                "level": "topic",
                "unit": q.unit,
                "chapter": q.chapter,
                "topic": q.topic,
                "title": q.title,
                "questions": [
                    {
                        "question": question.question,
                        "reference_answer": question.reference_answer,
                        "type": question.question_type,
                        "key_concepts": question.key_concepts,
                    }
                    for question in q.questions
                ],
            }
            for q in topic_quizzes
        ]
        self._save_json(quiz_data, "quiz_data.json")

        # Chapter tests
        chapter_data = [
            {
                "level": "chapter",
                "unit": t.unit,
                "chapter": t.chapter,
                "title": t.title,
                "questions": [
                    {
                        "question": question.question,
                        "reference_answer": question.reference_answer,
                        "type": question.question_type,
                        "options": question.options,
                        "correct_option": question.correct_option,
                    }
                    for question in t.questions
                ],
            }
            for t in chapter_tests
        ]
        self._save_json(chapter_data, "chapter_tests.json")

        # Unit exams
        unit_data = [
            {
                "level": "unit",
                "unit": e.unit,
                "title": e.title,
                "questions": [
                    {
                        "question": question.question,
                        "reference_answer": question.reference_answer,
                        "type": question.question_type,
                        "options": question.options,
                        "correct_option": question.correct_option,
                    }
                    for question in e.questions
                ],
            }
            for e in unit_exams
        ]
        self._save_json(unit_data, "unit_exams.json")

        # Final assessment (synthesized from unit exams)
        final_assessment = self._create_final_assessment(unit_exams)
        self._save_json(final_assessment, "final_assessment.json")

        # Critical distinctions (separate file for reference)
        critical_data = self.critical_distinctions.to_jsonl_format()
        self._save_jsonl(critical_data, "critical_distinctions.jsonl")

        # Save pipeline stats
        self._save_json(self._stats_to_dict(), "pipeline_stats.json")

    def _create_final_assessment(self, unit_exams: List[UnitExam]) -> Dict:
        """Create a final assessment from unit exam questions."""
        # Sample questions from each unit for the final assessment
        questions_per_unit = 5
        final_questions = []

        for exam in unit_exams:
            # Take up to N open-ended questions from each unit
            open_questions = [
                q for q in exam.questions if q.question_type == "open"
            ]
            selected = open_questions[:questions_per_unit]

            for question in selected:
                final_questions.append({
                    "question": question.question,
                    "reference_answer": question.reference_answer,
                    "type": question.question_type,
                    "source_unit": exam.unit,
                })

        return {
            "level": "final",
            "title": "Final Assessment",
            "description": "Cross-unit assessment testing comprehensive understanding",
            "questions": final_questions,
        }

    def _save_jsonl(self, data: List[Dict], filename: str):
        """Save data as JSONL."""
        filepath = self.output_path / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(data)} records to {filepath}")

    def _save_json(self, data, filename: str):
        """Save data as JSON."""
        filepath = self.output_path / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved to {filepath}")

    def _stats_to_dict(self) -> Dict:
        """Convert stats to dictionary."""
        return {
            "topics_processed": self.stats.topics_processed,
            "chapters_processed": self.stats.chapters_processed,
            "units_processed": self.stats.units_processed,
            "total_questions_before": self.stats.total_questions_before,
            "total_questions_after": self.stats.total_questions_after,
            "questions_generated": self.stats.questions_generated,
            "sft_examples": self.stats.sft_examples,
            "preference_pairs": self.stats.preference_pairs,
            "critical_pairs": self.stats.critical_pairs,
            "timestamp": self.stats.timestamp,
        }

    def _print_summary(self):
        """Print a summary of the pipeline run."""
        print("\n" + "=" * 60)
        print("DATA PREPARATION PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Topics processed:      {self.stats.topics_processed}")
        print(f"Chapters processed:    {self.stats.chapters_processed}")
        print(f"Units processed:       {self.stats.units_processed}")
        print("-" * 60)
        print(f"Questions before:      {self.stats.total_questions_before}")
        print(f"Questions generated:   {self.stats.questions_generated}")
        print(f"Questions after:       {self.stats.total_questions_after}")
        print("-" * 60)
        print(f"SFT examples:          {self.stats.sft_examples}")
        print(f"Preference pairs:      {self.stats.preference_pairs}")
        print(f"Critical pairs:        {self.stats.critical_pairs}")
        print(f"Total preference:      {self.stats.preference_pairs + self.stats.critical_pairs}")
        print("=" * 60)
        print(f"Output directory: {self.output_path}")
        print("=" * 60 + "\n")
