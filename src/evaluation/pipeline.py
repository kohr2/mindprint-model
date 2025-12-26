"""
EvaluationPipeline - Hierarchical evaluation pipeline for mindprint models.

Runs progressive evaluation: Topic → Chapter → Unit → Final
with early termination on critical failures.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import json
import logging

from .voice_evaluator import VoiceFidelityEvaluator, VoiceEvaluationResult

logger = logging.getLogger(__name__)


class EvalLevel(Enum):
    """Evaluation level in the hierarchy."""

    TOPIC = "topic"
    CHAPTER = "chapter"
    UNIT = "unit"
    FINAL = "final"


@dataclass
class QuestionResult:
    """Result for a single question."""

    question_id: str
    question: str
    reference_answer: str
    generated_answer: str
    semantic_score: float
    voice_score: float
    passed: bool
    violations: List[str] = field(default_factory=list)


@dataclass
class LevelResult:
    """Aggregated result for an evaluation level."""

    level: EvalLevel
    identifier: str  # e.g., "unit-01/chapter-01/topic-01"
    total_questions: int
    passed_questions: int
    accuracy: float
    avg_semantic_score: float
    avg_voice_score: float
    critical_distinctions_passed: bool
    negative_patterns_found: List[str]
    question_results: List[QuestionResult]

    @property
    def passed(self) -> bool:
        """Check if this level passed all criteria."""
        thresholds = {
            EvalLevel.TOPIC: (0.90, 0.75),
            EvalLevel.CHAPTER: (0.85, 0.75),
            EvalLevel.UNIT: (0.80, 0.75),
            EvalLevel.FINAL: (0.85, 0.80),
        }
        acc_thresh, voice_thresh = thresholds[self.level]

        return (
            self.accuracy >= acc_thresh
            and self.avg_voice_score >= voice_thresh
            and self.critical_distinctions_passed
            and len(self.negative_patterns_found) == 0
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "level": self.level.value,
            "identifier": self.identifier,
            "total_questions": self.total_questions,
            "passed_questions": self.passed_questions,
            "accuracy": self.accuracy,
            "avg_semantic_score": self.avg_semantic_score,
            "avg_voice_score": self.avg_voice_score,
            "critical_distinctions_passed": self.critical_distinctions_passed,
            "negative_patterns_found": self.negative_patterns_found,
            "passed": self.passed,
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report."""

    model_name: str
    approach: str  # "dpo" or "ppo"
    timestamp: str

    topic_results: List[LevelResult]
    chapter_results: List[LevelResult]
    unit_results: List[LevelResult]
    final_result: Optional[LevelResult]

    overall_accuracy: float
    overall_voice_score: float
    passed: bool

    failed_topics: List[str] = field(default_factory=list)
    critical_violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "approach": self.approach,
            "timestamp": self.timestamp,
            "overall_accuracy": self.overall_accuracy,
            "overall_voice_score": self.overall_voice_score,
            "passed": self.passed,
            "failed_topics": self.failed_topics,
            "critical_violations": self.critical_violations,
            "recommendations": self.recommendations,
            "summary": {
                "topics": {
                    "count": len(self.topic_results),
                    "passed": sum(1 for r in self.topic_results if r.passed),
                },
                "chapters": {
                    "count": len(self.chapter_results),
                    "passed": sum(1 for r in self.chapter_results if r.passed),
                },
                "units": {
                    "count": len(self.unit_results),
                    "passed": sum(1 for r in self.unit_results if r.passed),
                },
                "final": self.final_result.to_dict() if self.final_result else None,
            },
        }


class EvaluationPipeline:
    """
    Hierarchical evaluation pipeline for mindprint models.

    Runs progressive evaluation: Topic → Chapter → Unit → Final
    Early termination on critical failures.
    """

    def __init__(
        self,
        model,
        tokenizer,
        quiz_data_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        early_termination: bool = True,
    ):
        """
        Initialize the pipeline.

        Args:
            model: The language model to evaluate
            tokenizer: Tokenizer for the model
            quiz_data_path: Path to directory containing quiz data files
            embedding_model: Model for semantic similarity
            early_termination: Whether to stop early on critical failures
        """
        self.model = model
        self.tokenizer = tokenizer
        self.quiz_data = self._load_quiz_data(quiz_data_path)
        self.voice_evaluator = VoiceFidelityEvaluator(embedding_model)
        self.early_termination = early_termination

    def _load_quiz_data(self, path: str) -> Dict:
        """Load hierarchical quiz data from files."""
        quiz_path = Path(path)

        data = {
            "topics": [],
            "chapters": [],
            "units": [],
            "final": None,
        }

        # Load topic quizzes
        topic_file = quiz_path / "quiz_data.json"
        if topic_file.exists():
            with open(topic_file) as f:
                data["topics"] = json.load(f)

        # Load chapter tests
        chapter_file = quiz_path / "chapter_tests.json"
        if chapter_file.exists():
            with open(chapter_file) as f:
                data["chapters"] = json.load(f)

        # Load unit exams
        unit_file = quiz_path / "unit_exams.json"
        if unit_file.exists():
            with open(unit_file) as f:
                data["units"] = json.load(f)

        # Load final assessment
        final_file = quiz_path / "final_assessment.json"
        if final_file.exists():
            with open(final_file) as f:
                data["final"] = json.load(f)

        return data

    def run_full_evaluation(
        self,
        model_name: str,
        approach: str,
    ) -> EvaluationReport:
        """
        Run complete hierarchical evaluation.

        Progression: Topic → Chapter → Unit → Final
        Early termination on critical failures if enabled.

        Args:
            model_name: Name of the model being evaluated
            approach: Training approach ("dpo" or "ppo")

        Returns:
            Complete EvaluationReport
        """
        topic_results = []
        chapter_results = []
        unit_results = []
        final_result = None

        failed_topics = []
        critical_violations = []

        # Phase 1: Topic-level evaluation
        print("=" * 60)
        print("PHASE 1: TOPIC QUIZZES")
        print("=" * 60)

        for topic_quiz in self.quiz_data["topics"]:
            result = self._evaluate_level(
                level=EvalLevel.TOPIC,
                identifier=f"{topic_quiz['unit']}/{topic_quiz['chapter']}/{topic_quiz['topic']}",
                questions=topic_quiz["questions"],
            )
            topic_results.append(result)

            if not result.passed:
                failed_topics.append(result.identifier)
                status = "FAIL"
            else:
                status = "PASS"

            print(
                f"  [{status}] {result.identifier}: "
                f"{result.accuracy:.1%} accuracy, {result.avg_voice_score:.2f} voice"
            )

            if not result.critical_distinctions_passed:
                critical_violations.extend(
                    [f"{result.identifier}: {v}" for v in result.negative_patterns_found]
                )

        # Early termination check
        if self.quiz_data["topics"]:
            topic_pass_rate = sum(1 for r in topic_results if r.passed) / len(
                topic_results
            )
            if self.early_termination and topic_pass_rate < 0.50:
                print(f"\nEarly termination: Only {topic_pass_rate:.1%} topics passed")
                return self._build_report(
                    model_name,
                    approach,
                    datetime.now().isoformat(),
                    topic_results,
                    [],
                    [],
                    None,
                    failed_topics,
                    critical_violations,
                    passed=False,
                )

        # Phase 2: Chapter-level evaluation
        print("\n" + "=" * 60)
        print("PHASE 2: CHAPTER TESTS")
        print("=" * 60)

        for chapter_test in self._get_chapter_tests():
            result = self._evaluate_level(
                level=EvalLevel.CHAPTER,
                identifier=f"{chapter_test['unit']}/{chapter_test['chapter']}",
                questions=chapter_test["questions"],
            )
            chapter_results.append(result)

            status = "PASS" if result.passed else "FAIL"
            print(
                f"  [{status}] {result.identifier}: "
                f"{result.accuracy:.1%} accuracy, {result.avg_voice_score:.2f} voice"
            )

        # Phase 3: Unit-level evaluation
        print("\n" + "=" * 60)
        print("PHASE 3: UNIT EXAMS")
        print("=" * 60)

        for unit_exam in self._get_unit_exams():
            result = self._evaluate_level(
                level=EvalLevel.UNIT,
                identifier=unit_exam["unit"],
                questions=unit_exam["questions"],
            )
            unit_results.append(result)

            status = "PASS" if result.passed else "FAIL"
            print(
                f"  [{status}] {result.identifier}: "
                f"{result.accuracy:.1%} accuracy, {result.avg_voice_score:.2f} voice"
            )

        # Phase 4: Final assessment
        print("\n" + "=" * 60)
        print("PHASE 4: FINAL ASSESSMENT")
        print("=" * 60)

        if self.quiz_data["final"]:
            final_result = self._evaluate_level(
                level=EvalLevel.FINAL,
                identifier="final",
                questions=self.quiz_data["final"]["questions"],
            )
            status = "PASS" if final_result.passed else "FAIL"
            print(
                f"  [{status}] Final: "
                f"{final_result.accuracy:.1%} accuracy, {final_result.avg_voice_score:.2f} voice"
            )

        # Build final report
        return self._build_report(
            model_name,
            approach,
            datetime.now().isoformat(),
            topic_results,
            chapter_results,
            unit_results,
            final_result,
            failed_topics,
            critical_violations,
        )

    def _evaluate_level(
        self,
        level: EvalLevel,
        identifier: str,
        questions: List[Dict],
    ) -> LevelResult:
        """Evaluate a single level (topic/chapter/unit/final)."""
        question_results = []
        all_generated = []
        all_references = []

        for i, q in enumerate(questions):
            # Only evaluate open-ended questions
            if q.get("type") not in [None, "open"]:
                continue

            # Generate answer
            generated = self._generate_answer(q["question"])
            reference = q["reference_answer"]

            all_generated.append(generated)
            all_references.append(reference)

            # Individual voice evaluation
            voice_result = self.voice_evaluator.evaluate([generated], [reference])

            question_results.append(
                QuestionResult(
                    question_id=f"{identifier}/q{i + 1}",
                    question=q["question"],
                    reference_answer=reference,
                    generated_answer=generated,
                    semantic_score=voice_result.semantic_similarity,
                    voice_score=voice_result.overall_score,
                    passed=voice_result.passed,
                    violations=voice_result.violations,
                )
            )

        # Handle case with no questions
        if not question_results:
            return LevelResult(
                level=level,
                identifier=identifier,
                total_questions=0,
                passed_questions=0,
                accuracy=0.0,
                avg_semantic_score=0.0,
                avg_voice_score=0.0,
                critical_distinctions_passed=True,
                negative_patterns_found=[],
                question_results=[],
            )

        # Aggregate voice evaluation
        overall_voice = self.voice_evaluator.evaluate(all_generated, all_references)

        passed_count = sum(1 for qr in question_results if qr.passed)
        accuracy = passed_count / len(question_results)

        return LevelResult(
            level=level,
            identifier=identifier,
            total_questions=len(question_results),
            passed_questions=passed_count,
            accuracy=accuracy,
            avg_semantic_score=overall_voice.semantic_similarity,
            avg_voice_score=overall_voice.overall_score,
            critical_distinctions_passed=overall_voice.critical_distinctions_passed,
            negative_patterns_found=overall_voice.violations,
            question_results=question_results,
        )

    def _generate_answer(self, question: str) -> str:
        """Generate model answer for a question."""
        prompt = self._format_prompt(question)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        return answer.strip()

    def _format_prompt(self, question: str) -> str:
        """Format question as model prompt (Gemma-3 chat format)."""
        return f"""<start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
"""

    def _get_chapter_tests(self) -> List[Dict]:
        """Get or synthesize chapter tests."""
        if self.quiz_data["chapters"]:
            return self.quiz_data["chapters"]

        # Synthesize from topics
        chapters = defaultdict(list)
        for topic in self.quiz_data["topics"]:
            key = (topic["unit"], topic["chapter"])
            chapters[key].extend(topic["questions"])

        return [
            {"unit": unit, "chapter": chapter, "questions": questions}
            for (unit, chapter), questions in chapters.items()
        ]

    def _get_unit_exams(self) -> List[Dict]:
        """Get or synthesize unit exams."""
        if self.quiz_data["units"]:
            return self.quiz_data["units"]

        # Synthesize from topics
        units = defaultdict(list)
        for topic in self.quiz_data["topics"]:
            units[topic["unit"]].extend(topic["questions"])

        return [
            {"unit": unit, "questions": questions} for unit, questions in units.items()
        ]

    def _build_report(
        self,
        model_name: str,
        approach: str,
        timestamp: str,
        topic_results: List[LevelResult],
        chapter_results: List[LevelResult],
        unit_results: List[LevelResult],
        final_result: Optional[LevelResult],
        failed_topics: List[str],
        critical_violations: List[str],
        passed: Optional[bool] = None,
    ) -> EvaluationReport:
        """Build comprehensive evaluation report."""
        # Calculate overall metrics
        all_results = topic_results + chapter_results + unit_results
        if final_result:
            all_results.append(final_result)

        overall_accuracy = (
            sum(r.accuracy for r in all_results) / len(all_results)
            if all_results
            else 0.0
        )
        overall_voice = (
            sum(r.avg_voice_score for r in all_results) / len(all_results)
            if all_results
            else 0.0
        )

        # Determine pass/fail
        if passed is None:
            passed = (
                all(r.passed for r in unit_results)
                and (final_result is None or final_result.passed)
                and len(critical_violations) == 0
            )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            topic_results,
            chapter_results,
            unit_results,
            final_result,
            failed_topics,
            critical_violations,
        )

        return EvaluationReport(
            model_name=model_name,
            approach=approach,
            timestamp=timestamp,
            topic_results=topic_results,
            chapter_results=chapter_results,
            unit_results=unit_results,
            final_result=final_result,
            overall_accuracy=overall_accuracy,
            overall_voice_score=overall_voice,
            passed=passed,
            failed_topics=failed_topics,
            critical_violations=critical_violations,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        topic_results: List[LevelResult],
        chapter_results: List[LevelResult],
        unit_results: List[LevelResult],
        final_result: Optional[LevelResult],
        failed_topics: List[str],
        critical_violations: List[str],
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Critical violations are highest priority
        if critical_violations:
            recommendations.append(
                "CRITICAL: Model has halving/cycle confusion. "
                "Add more preference pairs emphasizing this distinction."
            )

        # Failed topics analysis
        if topic_results and len(failed_topics) > len(topic_results) * 0.3:
            recommendations.append(
                f"HIGH: {len(failed_topics)} topics failed. "
                "Consider additional SFT epochs or data augmentation."
            )
        elif failed_topics:
            # Identify patterns in failures
            unit_failures = defaultdict(int)
            for topic in failed_topics:
                unit = topic.split("/")[0]
                unit_failures[unit] += 1

            if unit_failures:
                worst_unit = max(unit_failures.items(), key=lambda x: x[1])
                recommendations.append(
                    f"MEDIUM: Unit '{worst_unit[0]}' has {worst_unit[1]} topic failures. "
                    "Focus additional training on this unit's content."
                )

        # Voice score analysis
        low_voice_topics = [r.identifier for r in topic_results if r.avg_voice_score < 0.70]
        if low_voice_topics:
            recommendations.append(
                f"MEDIUM: {len(low_voice_topics)} topics have low voice fidelity. "
                "Increase weight on voice-related preference pairs."
            )

        # Semantic similarity analysis
        low_semantic = [r.identifier for r in topic_results if r.avg_semantic_score < 0.65]
        if low_semantic:
            recommendations.append(
                f"LOW: {len(low_semantic)} topics have low semantic similarity. "
                "Model may be generating off-topic content."
            )

        if not recommendations:
            recommendations.append("Model passed all evaluation criteria.")

        return recommendations
