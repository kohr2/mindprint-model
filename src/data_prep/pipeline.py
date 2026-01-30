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
import random
from datetime import datetime

from .textbook_parser import TextbookParser, Question, TopicQuiz, ChapterTest, UnitExam
from .question_generator import QuestionGenerator, GenerationConfig
from .preference_generator import PreferencePairGenerator, PreferencePair
from .critical_distinctions import CriticalDistinctions
from .transcript_processor import TranscriptProcessor
from .transcript_question_generator import TranscriptQuestionGenerator

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline."""

    textbook_path: Optional[str] = None
    output_path: str = ""
    transcript_dir: Optional[str] = None
    summaries_dir: Optional[str] = None  # Episode summaries from mindprint-agent
    use_transcripts: bool = False
    combine_with_textbook: bool = False
    target_questions_per_topic: int = 15  # Increased from 10
    target_questions_per_episode: int = 15
    augment_questions: bool = True
    include_critical_distinctions: bool = True
    api_key: Optional[str] = None  # For question generation
    textbook_ratio: float = 0.6  # Ratio of textbook data when combining (0.6 = 60% textbook, 40% transcripts)
    # New quality control parameters
    enhance_voice: bool = True
    normalize_lengths: bool = True
    min_answer_length: int = 600
    max_answer_length: int = 1200
    min_voice_marker_density: float = 20.0  # Percentage


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

        # Initialize textbook components (if using textbook)
        self.parser = None
        self.question_gen = None
        if config.textbook_path:
            self.parser = TextbookParser(config.textbook_path)
            if config.augment_questions:
                gen_config = GenerationConfig(
                    target_questions=config.target_questions_per_topic,
                    min_answer_length=config.min_answer_length,
                    max_answer_length=config.max_answer_length,
                    min_voice_marker_density=config.min_voice_marker_density,
                    enhance_voice=config.enhance_voice,
                    normalize_lengths=config.normalize_lengths,
                )
                self.question_gen = QuestionGenerator(
                    parser=self.parser,
                    config=gen_config,
                    api_key=config.api_key,
                )

        # Initialize transcript components (if using transcripts)
        self.transcript_processor = None
        self.transcript_question_gen = None
        if config.use_transcripts and config.transcript_dir:
            self.transcript_processor = TranscriptProcessor(
                transcripts_dir=config.transcript_dir,
                summaries_dir=config.summaries_dir,
            )
            if config.augment_questions:
                gen_config = GenerationConfig(
                    target_questions=config.target_questions_per_episode
                )
                self.transcript_question_gen = TranscriptQuestionGenerator(
                    processor=self.transcript_processor,
                    config=gen_config,
                    api_key=config.api_key,
                )

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
        logger.info(f"Output path: {self.output_path}")

        # Determine data sources
        use_textbook = self.config.textbook_path is not None
        use_transcripts = self.config.use_transcripts and self.config.transcript_dir is not None
        combine = self.config.combine_with_textbook and use_textbook and use_transcripts

        if combine:
            logger.info("Mode: Combined (textbook + transcripts)")
        elif use_transcripts and not use_textbook:
            logger.info("Mode: Transcripts only")
        elif use_textbook and not use_transcripts:
            logger.info("Mode: Textbook only")
        else:
            raise ValueError("Must specify either textbook_path or use_transcripts=True")

        # Phase 1: Process textbook data (if applicable)
        topic_quizzes = []
        chapter_tests = []
        unit_exams = []
        textbook_sft_data = []
        textbook_preference_pairs = []

        if use_textbook:
            logger.info("Phase 1: Processing textbook data")
            logger.info(f"Textbook path: {self.config.textbook_path}")
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

            # Augment textbook questions
            if self.config.augment_questions and self.question_gen:
                logger.info("Augmenting textbook questions")
                topic_quizzes = self.question_gen.augment_all(topic_quizzes)
                
                # Ensure questions have proper source identifiers
                for quiz in topic_quizzes:
                    for question in quiz.questions:
                        if not question.source:
                            question.source = quiz.identifier
                
                self.stats.total_questions_after = sum(
                    len(q.questions) for q in topic_quizzes
                )
                self.stats.questions_generated = (
                    self.stats.total_questions_after - self.stats.total_questions_before
                )
            else:
                self.stats.total_questions_after = self.stats.total_questions_before
                
                # Ensure questions have proper source identifiers even without augmentation
                for quiz in topic_quizzes:
                    for question in quiz.questions:
                        if not question.source:
                            question.source = quiz.identifier

            # Generate textbook SFT and preference data
            textbook_sft_data = self._create_sft_data(topic_quizzes, chapter_tests, unit_exams)
            textbook_preference_pairs = self._create_preference_pairs(
                topic_quizzes, chapter_tests, unit_exams
            )

        # Phase 2: Process transcript data (if applicable)
        transcript_questions = []
        transcript_sft_data = []
        transcript_preference_pairs = []

        if use_transcripts:
            logger.info("Phase 2: Processing transcript data")
            logger.info(f"Transcript dir: {self.config.transcript_dir}")

            # Process all transcripts
            transcript_questions = self.transcript_processor.process_all_transcripts()

            # Generate questions if augmentation enabled
            if self.config.augment_questions and self.transcript_question_gen:
                logger.info("Generating questions from transcripts")
                # This would require refactoring to process episode by episode
                # For now, use the basic questions from processor
                pass

            # Create SFT data from transcript questions
            for question in transcript_questions:
                # Use question.source if set, otherwise default to 'transcript'
                source = question.source if question.source else 'transcript'
                transcript_sft_data.append({
                    "instruction": question.question,
                    "input": "",
                    "output": question.reference_answer,
                    "source": source,
                })

            # Create preference pairs
            transcript_preference_pairs = self.preference_gen.generate_all(
                [(q, q.source or 'transcript') for q in transcript_questions]
            )

            logger.info(f"Generated {len(transcript_sft_data)} transcript SFT examples")
            logger.info(f"Generated {len(transcript_preference_pairs)} transcript preference pairs")

        # Phase 3: Combine datasets (if requested)
        if combine:
            logger.info("Phase 3: Combining datasets")
            sft_data = self._combine_sft_data(
                textbook_sft_data, transcript_sft_data, self.config.textbook_ratio
            )
            preference_pairs = textbook_preference_pairs + transcript_preference_pairs
        elif use_transcripts:
            sft_data = transcript_sft_data
            preference_pairs = transcript_preference_pairs
        else:
            sft_data = textbook_sft_data
            preference_pairs = textbook_preference_pairs

        self.stats.sft_examples = len(sft_data)
        self.stats.preference_pairs = len(preference_pairs)

        # Phase 4: Add critical distinctions
        if self.config.include_critical_distinctions:
            logger.info("Phase 4: Adding critical distinction pairs")
            critical_pairs = self.critical_distinctions.get_all_pairs()
            self.stats.critical_pairs = len(critical_pairs)
            preference_pairs.extend(critical_pairs)
            logger.info(f"Added {len(critical_pairs)} critical distinction pairs")

        # Phase 5: Save all outputs
        logger.info("Phase 5: Saving outputs")
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

    def _combine_sft_data(
        self,
        textbook_data: List[Dict],
        transcript_data: List[Dict],
        textbook_ratio: float,
    ) -> List[Dict]:
        """
        Combine textbook and transcript SFT data with specified ratio.

        Args:
            textbook_data: Textbook SFT examples
            transcript_data: Transcript SFT examples
            textbook_ratio: Ratio of textbook data (0.6 = 60% textbook, 40% transcripts)

        Returns:
            Combined SFT data
        """
        total = len(textbook_data) + len(transcript_data)
        if total == 0:
            return []

        # Calculate target counts
        target_textbook = int(total * textbook_ratio)
        target_transcript = total - target_textbook

        # Sample to achieve ratio
        import random
        combined = []

        # Add textbook data (up to target)
        textbook_sample = textbook_data[:target_textbook] if len(textbook_data) >= target_textbook else textbook_data
        combined.extend(textbook_sample)

        # Add transcript data (up to target)
        transcript_sample = transcript_data[:target_transcript] if len(transcript_data) >= target_transcript else transcript_data
        combined.extend(transcript_sample)

        # Shuffle to mix sources
        random.shuffle(combined)

        logger.info(f"Combined {len(textbook_sample)} textbook + {len(transcript_sample)} transcript = {len(combined)} total examples")
        return combined

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
                # Use question.source if set, otherwise use quiz identifier
                source = question.source if question.source else quiz.identifier
                sft_data.append({
                    "instruction": question.question,
                    "input": "",
                    "output": question.reference_answer,
                    "source": source,  # Ensure proper topic mapping
                })

        # From chapter tests (short answer only, MC/TF don't work well for SFT)
        for test in chapter_tests:
            for question in test.questions:
                if question.question_type == "open":
                    # Use question.source if set, otherwise use test identifier
                    source = question.source if question.source else test.identifier
                    sft_data.append({
                        "instruction": question.question,
                        "input": "",
                        "output": question.reference_answer,
                        "source": source,  # Ensure proper topic mapping
                    })

        # From unit exams
        for exam in unit_exams:
            for question in exam.questions:
                if question.question_type == "open":
                    # Use question.source if set, otherwise use exam identifier
                    source = question.source if question.source else exam.identifier
                    sft_data.append({
                        "instruction": question.question,
                        "input": "",
                        "output": question.reference_answer,
                        "source": source,  # Ensure proper topic mapping
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
                # Use question.source if set, otherwise use quiz identifier
                source = question.source if question.source else quiz.identifier
                all_questions.append((question, source))

        # From chapter tests (open-ended only)
        for test in chapter_tests:
            for question in test.questions:
                if question.question_type == "open":
                    # Use question.source if set, otherwise use test identifier
                    source = question.source if question.source else test.identifier
                    all_questions.append((question, source))

        # From unit exams
        for exam in unit_exams:
            for question in exam.questions:
                if question.question_type == "open":
                    # Use question.source if set, otherwise use exam identifier
                    source = question.source if question.source else exam.identifier
                    all_questions.append((question, source))

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
                "source": p.source,  # Include source for proper topic mapping
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
