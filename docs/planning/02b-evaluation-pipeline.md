# Phase 2b: Shared Evaluation Pipeline

## Objective

Implement a unified evaluation pipeline that measures trained model quality across **factual accuracy** (quiz performance) and **voice fidelity** (Bob's communication style). This pipeline is used by both DPO and PPO approaches as the final quality gate.

## Architecture Position

```
┌────────────────────────────────────────────────────────────────────┐
│                        EVALUATION PIPELINE                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐                                                  │
│   │   TRAINED    │  (From DPO or PPO branch)                       │
│   │    MODEL     │                                                  │
│   └──────┬───────┘                                                  │
│          │                                                          │
│          ▼                                                          │
│   ┌──────────────────────────────────────────────────────────────┐ │
│   │                    HIERARCHICAL QUIZ                          │ │
│   │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │ │
│   │  │ TOPIC   │───▶│ CHAPTER │───▶│  UNIT   │───▶│  FINAL  │   │ │
│   │  │ (10 Q)  │    │ (30 Q)  │    │ (50 Q)  │    │ (20 Q)  │   │ │
│   │  │  ≥90%   │    │  ≥85%   │    │  ≥80%   │    │  ≥85%   │   │ │
│   │  └─────────┘    └─────────┘    └─────────┘    └─────────┘   │ │
│   └──────────────────────────────────────────────────────────────┘ │
│          │                                                          │
│          ▼                                                          │
│   ┌──────────────────────────────────────────────────────────────┐ │
│   │                    VOICE FIDELITY                             │ │
│   │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │ │
│   │  │SEMANTIC │    │ MARKER  │    │CRITICAL │    │NEGATIVE │   │ │
│   │  │SIMILARITY│   │ SCORE   │    │DISTINCT.│    │PATTERNS │   │ │
│   │  │  ≥0.70  │    │  ≥0.60  │    │  100%   │    │   0     │   │ │
│   │  └─────────┘    └─────────┘    └─────────┘    └─────────┘   │ │
│   └──────────────────────────────────────────────────────────────┘ │
│          │                                                          │
│          ▼                                                          │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│   │   AGGREGATE  │───▶│   REPORT     │───▶│   GATE       │        │
│   │   (Per Unit) │    │  (Dashboard) │    │  (Pass/Fail) │        │
│   └──────────────┘    └──────────────┘    └──────────────┘        │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

## Evaluation Hierarchy

### Level 1: Topic Quizzes (Granular)

Each topic has 10 questions testing specific concepts.

| Attribute | Value |
|-----------|-------|
| Questions per topic | 10 |
| Pass threshold | ≥90% (9/10) |
| Voice threshold | ≥0.75 |
| Failure action | Flag for targeted retraining |

### Level 2: Chapter Tests (Integrated)

Each chapter aggregates 3-4 topics with cross-concept questions.

| Attribute | Value |
|-----------|-------|
| Questions per chapter | ~30 (10 per topic + 5 synthesis) |
| Pass threshold | ≥85% |
| Voice threshold | ≥0.75 |
| Failure action | Review topic-level failures |

### Level 3: Unit Exams (Comprehensive)

Each unit tests deep understanding across chapters.

| Attribute | Value |
|-----------|-------|
| Questions per unit | ~50 |
| Pass threshold | ≥80% |
| Voice threshold | ≥0.75 |
| Failure action | Block progression to next unit |

### Level 4: Final Assessment (Holistic)

Cross-unit questions testing full mindprint integration.

| Attribute | Value |
|-----------|-------|
| Total questions | 20 (5 per unit) |
| Pass threshold | ≥85% |
| Voice threshold | ≥0.80 |
| Critical distinctions | 100% (halving vs cycle) |

## Implementation

### Core Evaluation Engine

```python
# src/evaluation/pipeline.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json
from pathlib import Path
from collections import defaultdict

from .voice_evaluator import VoiceFidelityEvaluator, VoiceEvaluationResult


class EvalLevel(Enum):
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
            self.accuracy >= acc_thresh and
            self.avg_voice_score >= voice_thresh and
            self.critical_distinctions_passed and
            len(self.negative_patterns_found) == 0
        )


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
    
    # Detailed breakdowns
    failed_topics: List[str] = field(default_factory=list)
    critical_violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


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
        self.model = model
        self.tokenizer = tokenizer
        self.quiz_data = self._load_quiz_data(quiz_data_path)
        self.voice_evaluator = VoiceFidelityEvaluator(embedding_model)
        self.early_termination = early_termination
    
    def _load_quiz_data(self, path: str) -> Dict:
        """Load hierarchical quiz data."""
        quiz_path = Path(path)
        
        data = {
            "topics": [],
            "chapters": [],
            "units": [],
            "final": None,
        }
        
        # Load topic quizzes
        if (quiz_path / "quiz_data.json").exists():
            with open(quiz_path / "quiz_data.json") as f:
                data["topics"] = json.load(f)
        
        # Load chapter tests (if separate)
        if (quiz_path / "chapter_tests.json").exists():
            with open(quiz_path / "chapter_tests.json") as f:
                data["chapters"] = json.load(f)
        
        # Load unit exams
        if (quiz_path / "unit_exams.json").exists():
            with open(quiz_path / "unit_exams.json") as f:
                data["units"] = json.load(f)
        
        # Load final assessment
        if (quiz_path / "final_assessment.json").exists():
            with open(quiz_path / "final_assessment.json") as f:
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
        """
        from datetime import datetime
        
        topic_results = []
        chapter_results = []
        unit_results = []
        final_result = None
        
        failed_topics = []
        critical_violations = []
        
        # Phase 1: Topic-level evaluation
        print("═" * 60)
        print("PHASE 1: TOPIC QUIZZES")
        print("═" * 60)
        
        for topic_quiz in self.quiz_data["topics"]:
            result = self._evaluate_level(
                level=EvalLevel.TOPIC,
                identifier=f"{topic_quiz['unit']}/{topic_quiz['chapter']}/{topic_quiz['topic']}",
                questions=topic_quiz["questions"],
            )
            topic_results.append(result)
            
            if not result.passed:
                failed_topics.append(result.identifier)
                print(f"  ✗ {result.identifier}: {result.accuracy:.1%} accuracy, {result.avg_voice_score:.2f} voice")
            else:
                print(f"  ✓ {result.identifier}: {result.accuracy:.1%} accuracy, {result.avg_voice_score:.2f} voice")
            
            if not result.critical_distinctions_passed:
                critical_violations.extend([
                    f"{result.identifier}: {v}" for v in result.negative_patterns_found
                ])
        
        # Early termination check
        topic_pass_rate = sum(1 for r in topic_results if r.passed) / len(topic_results)
        if self.early_termination and topic_pass_rate < 0.50:
            print(f"\n⚠️  Early termination: Only {topic_pass_rate:.1%} topics passed")
            return self._build_report(
                model_name, approach, datetime.now().isoformat(),
                topic_results, [], [], None,
                failed_topics, critical_violations,
                passed=False,
            )
        
        # Phase 2: Chapter-level evaluation
        print("\n" + "═" * 60)
        print("PHASE 2: CHAPTER TESTS")
        print("═" * 60)
        
        for chapter_test in self._get_chapter_tests():
            result = self._evaluate_level(
                level=EvalLevel.CHAPTER,
                identifier=f"{chapter_test['unit']}/{chapter_test['chapter']}",
                questions=chapter_test["questions"],
            )
            chapter_results.append(result)
            
            status = "✓" if result.passed else "✗"
            print(f"  {status} {result.identifier}: {result.accuracy:.1%} accuracy, {result.avg_voice_score:.2f} voice")
        
        # Phase 3: Unit-level evaluation
        print("\n" + "═" * 60)
        print("PHASE 3: UNIT EXAMS")
        print("═" * 60)
        
        for unit_exam in self._get_unit_exams():
            result = self._evaluate_level(
                level=EvalLevel.UNIT,
                identifier=unit_exam["unit"],
                questions=unit_exam["questions"],
            )
            unit_results.append(result)
            
            status = "✓" if result.passed else "✗"
            print(f"  {status} {result.identifier}: {result.accuracy:.1%} accuracy, {result.avg_voice_score:.2f} voice")
        
        # Phase 4: Final assessment
        print("\n" + "═" * 60)
        print("PHASE 4: FINAL ASSESSMENT")
        print("═" * 60)
        
        if self.quiz_data["final"]:
            final_result = self._evaluate_level(
                level=EvalLevel.FINAL,
                identifier="final",
                questions=self.quiz_data["final"]["questions"],
            )
            status = "✓" if final_result.passed else "✗"
            print(f"  {status} Final: {final_result.accuracy:.1%} accuracy, {final_result.avg_voice_score:.2f} voice")
        
        # Build final report
        return self._build_report(
            model_name, approach, datetime.now().isoformat(),
            topic_results, chapter_results, unit_results, final_result,
            failed_topics, critical_violations,
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
            # Generate answer
            generated = self._generate_answer(q["question"])
            reference = q["reference_answer"]
            
            all_generated.append(generated)
            all_references.append(reference)
            
            # Individual voice evaluation
            voice_result = self.voice_evaluator.evaluate([generated], [reference])
            
            question_results.append(QuestionResult(
                question_id=f"{identifier}/q{i+1}",
                question=q["question"],
                reference_answer=reference,
                generated_answer=generated,
                semantic_score=voice_result.semantic_similarity,
                voice_score=voice_result.overall_score,
                passed=voice_result.passed,
                violations=voice_result.violations,
            ))
        
        # Aggregate voice evaluation
        overall_voice = self.voice_evaluator.evaluate(all_generated, all_references)
        
        passed_count = sum(1 for qr in question_results if qr.passed)
        accuracy = passed_count / len(questions) if questions else 0.0
        
        return LevelResult(
            level=level,
            identifier=identifier,
            total_questions=len(questions),
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
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return answer.strip()
    
    def _format_prompt(self, question: str) -> str:
        """Format question as model prompt."""
        # Gemma-3 chat format
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
            {"unit": unit, "questions": questions}
            for unit, questions in units.items()
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
        
        overall_accuracy = sum(r.accuracy for r in all_results) / len(all_results) if all_results else 0.0
        overall_voice = sum(r.avg_voice_score for r in all_results) / len(all_results) if all_results else 0.0
        
        # Determine pass/fail
        if passed is None:
            passed = (
                all(r.passed for r in unit_results) and
                (final_result is None or final_result.passed) and
                len(critical_violations) == 0
            )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            topic_results, chapter_results, unit_results, final_result,
            failed_topics, critical_violations,
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
        if len(failed_topics) > len(topic_results) * 0.3:
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
            
            worst_unit = max(unit_failures.items(), key=lambda x: x[1])
            recommendations.append(
                f"MEDIUM: Unit '{worst_unit[0]}' has {worst_unit[1]} topic failures. "
                "Focus additional training on this unit's content."
            )
        
        # Voice score analysis
        low_voice_topics = [
            r.identifier for r in topic_results
            if r.avg_voice_score < 0.70
        ]
        if low_voice_topics:
            recommendations.append(
                f"MEDIUM: {len(low_voice_topics)} topics have low voice fidelity. "
                "Increase weight on voice-related preference pairs."
            )
        
        # Semantic similarity analysis
        low_semantic = [
            r.identifier for r in topic_results
            if r.avg_semantic_score < 0.65
        ]
        if low_semantic:
            recommendations.append(
                f"LOW: {len(low_semantic)} topics have low semantic similarity. "
                "Model may be generating off-topic content."
            )
        
        if not recommendations:
            recommendations.append("Model passed all evaluation criteria.")
        
        return recommendations
```

### Report Generator

```python
# src/evaluation/reporting.py

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import json
from datetime import datetime

from .pipeline import EvaluationReport, LevelResult


class ReportGenerator:
    """Generate evaluation reports in multiple formats."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all(self, report: EvaluationReport):
        """Generate all report formats."""
        self.generate_json(report)
        self.generate_markdown(report)
        self.generate_summary(report)
    
    def generate_json(self, report: EvaluationReport) -> Path:
        """Export complete report as JSON."""
        output_path = self.output_dir / f"eval_{report.timestamp.replace(':', '-')}.json"
        
        # Convert to serializable dict
        report_dict = {
            "model_name": report.model_name,
            "approach": report.approach,
            "timestamp": report.timestamp,
            "overall_accuracy": report.overall_accuracy,
            "overall_voice_score": report.overall_voice_score,
            "passed": report.passed,
            "failed_topics": report.failed_topics,
            "critical_violations": report.critical_violations,
            "recommendations": report.recommendations,
            "summary": {
                "topics": self._summarize_level(report.topic_results),
                "chapters": self._summarize_level(report.chapter_results),
                "units": self._summarize_level(report.unit_results),
                "final": self._summarize_single(report.final_result) if report.final_result else None,
            },
        }
        
        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        
        return output_path
    
    def generate_markdown(self, report: EvaluationReport) -> Path:
        """Generate detailed markdown report."""
        output_path = self.output_dir / f"eval_{report.timestamp.replace(':', '-')}.md"
        
        lines = [
            f"# Evaluation Report: {report.model_name}",
            f"",
            f"**Approach:** {report.approach.upper()}",
            f"**Timestamp:** {report.timestamp}",
            f"**Overall Result:** {'✅ PASSED' if report.passed else '❌ FAILED'}",
            f"",
            f"## Summary Metrics",
            f"",
            f"| Metric | Value | Threshold |",
            f"|--------|-------|-----------|",
            f"| Overall Accuracy | {report.overall_accuracy:.1%} | ≥80% |",
            f"| Voice Fidelity | {report.overall_voice_score:.2f} | ≥0.75 |",
            f"| Critical Violations | {len(report.critical_violations)} | 0 |",
            f"",
        ]
        
        # Topic results
        lines.extend([
            "## Topic Results",
            "",
            "| Topic | Accuracy | Voice | Status |",
            "|-------|----------|-------|--------|",
        ])
        for r in report.topic_results:
            status = "✅" if r.passed else "❌"
            lines.append(f"| {r.identifier} | {r.accuracy:.1%} | {r.avg_voice_score:.2f} | {status} |")
        lines.append("")
        
        # Chapter results
        if report.chapter_results:
            lines.extend([
                "## Chapter Results",
                "",
                "| Chapter | Accuracy | Voice | Status |",
                "|---------|----------|-------|--------|",
            ])
            for r in report.chapter_results:
                status = "✅" if r.passed else "❌"
                lines.append(f"| {r.identifier} | {r.accuracy:.1%} | {r.avg_voice_score:.2f} | {status} |")
            lines.append("")
        
        # Unit results
        if report.unit_results:
            lines.extend([
                "## Unit Results",
                "",
                "| Unit | Accuracy | Voice | Status |",
                "|------|----------|-------|--------|",
            ])
            for r in report.unit_results:
                status = "✅" if r.passed else "❌"
                lines.append(f"| {r.identifier} | {r.accuracy:.1%} | {r.avg_voice_score:.2f} | {status} |")
            lines.append("")
        
        # Final result
        if report.final_result:
            r = report.final_result
            status = "✅ PASSED" if r.passed else "❌ FAILED"
            lines.extend([
                "## Final Assessment",
                "",
                f"**Status:** {status}",
                f"**Accuracy:** {r.accuracy:.1%}",
                f"**Voice Fidelity:** {r.avg_voice_score:.2f}",
                "",
            ])
        
        # Recommendations
        if report.recommendations:
            lines.extend([
                "## Recommendations",
                "",
            ])
            for rec in report.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        
        # Critical violations
        if report.critical_violations:
            lines.extend([
                "## ⚠️ Critical Violations",
                "",
            ])
            for violation in report.critical_violations:
                lines.append(f"- {violation}")
            lines.append("")
        
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        
        return output_path
    
    def generate_summary(self, report: EvaluationReport) -> str:
        """Generate one-line summary for logs."""
        status = "PASS" if report.passed else "FAIL"
        return (
            f"[{status}] {report.model_name} ({report.approach}): "
            f"Acc={report.overall_accuracy:.1%}, "
            f"Voice={report.overall_voice_score:.2f}, "
            f"Violations={len(report.critical_violations)}"
        )
    
    def _summarize_level(self, results: List[LevelResult]) -> dict:
        """Summarize results for a level."""
        if not results:
            return {"count": 0, "passed": 0, "avg_accuracy": 0, "avg_voice": 0}
        
        return {
            "count": len(results),
            "passed": sum(1 for r in results if r.passed),
            "avg_accuracy": sum(r.accuracy for r in results) / len(results),
            "avg_voice": sum(r.avg_voice_score for r in results) / len(results),
        }
    
    def _summarize_single(self, result: LevelResult) -> dict:
        """Summarize single result."""
        return {
            "passed": result.passed,
            "accuracy": result.accuracy,
            "voice": result.avg_voice_score,
            "critical_passed": result.critical_distinctions_passed,
        }
```

### CLI Interface

```python
# src/evaluation/cli.py

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from .pipeline import EvaluationPipeline
from .reporting import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="Mindprint Evaluation Pipeline")
    parser.add_argument("--model", required=True, help="Base model path or HF name")
    parser.add_argument("--adapter", required=True, help="LoRA adapter path")
    parser.add_argument("--quiz-data", required=True, help="Quiz data directory")
    parser.add_argument("--output", default="./eval_results", help="Output directory")
    parser.add_argument("--approach", choices=["dpo", "ppo"], required=True)
    parser.add_argument("--no-early-termination", action="store_true")
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading base model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Load adapter
    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    
    # Run evaluation
    print(f"Running evaluation...")
    pipeline = EvaluationPipeline(
        model=model,
        tokenizer=tokenizer,
        quiz_data_path=args.quiz_data,
        early_termination=not args.no_early_termination,
    )
    
    report = pipeline.run_full_evaluation(
        model_name=Path(args.adapter).name,
        approach=args.approach,
    )
    
    # Generate reports
    reporter = ReportGenerator(args.output)
    reporter.generate_all(report)
    
    print("\n" + "═" * 60)
    print(reporter.generate_summary(report))
    print("═" * 60)
    
    return 0 if report.passed else 1


if __name__ == "__main__":
    exit(main())
```

## Evaluation Thresholds

### By Level

| Level | Accuracy | Voice | Critical | Purpose |
|-------|----------|-------|----------|---------|
| Topic | ≥90% | ≥0.75 | 100% | Granular concept mastery |
| Chapter | ≥85% | ≥0.75 | 100% | Cross-concept integration |
| Unit | ≥80% | ≥0.75 | 100% | Comprehensive understanding |
| Final | ≥85% | ≥0.80 | 100% | Holistic mindprint validation |

### By Metric

| Metric | Threshold | Weight | Rationale |
|--------|-----------|--------|-----------|
| Semantic Similarity | ≥0.70 | 50% | Core factual alignment |
| Voice Markers | ≥0.60 | 30% | Bob's communication style |
| Critical Distinctions | 100% | 10% | Non-negotiable accuracy |
| Negative Patterns | 0 | 10% | Avoid harmful content |

## Integration Points

### DPO Branch Integration

```python
# After DPO training completes
from evaluation import EvaluationPipeline, ReportGenerator

pipeline = EvaluationPipeline(
    model=dpo_trained_model,
    tokenizer=tokenizer,
    quiz_data_path="./data/bob_loukas",
)

report = pipeline.run_full_evaluation(
    model_name="bob-loukas-dpo-v1",
    approach="dpo",
)

if not report.passed:
    # Trigger additional training or human review
    handle_failure(report)
```

### PPO Branch Integration

```python
# After PPO training completes
from evaluation import EvaluationPipeline, ReportGenerator

pipeline = EvaluationPipeline(
    model=ppo_trained_model,
    tokenizer=tokenizer,
    quiz_data_path="./data/bob_loukas",
)

report = pipeline.run_full_evaluation(
    model_name="bob-loukas-ppo-v1",
    approach="ppo",
)

# Compare with DPO results if available
if dpo_report and report.overall_voice_score > dpo_report.overall_voice_score:
    print("PPO achieved better voice fidelity")
```

## Failure Handling

### Automatic Remediation

| Failure Type | Remediation |
|--------------|-------------|
| Low accuracy (<70%) | Additional SFT epochs |
| Low voice (<0.60) | Increase preference pair weight |
| Critical violation | Add targeted preference pairs |
| Unit failure | Re-train with focus on unit content |

### Manual Review Triggers

- Any critical violation (halving/cycle confusion)
- Voice score below 0.50 on any unit
- More than 50% topics failing
- Final assessment failure after 2 retraining cycles

## Dependencies

```
sentence-transformers>=2.5.0
scikit-learn>=1.4.0
transformers>=4.40.0
peft>=0.10.0
torch>=2.2.0
```

## File Structure

```
src/evaluation/
├── __init__.py
├── pipeline.py          # Core evaluation engine
├── voice_evaluator.py   # Voice fidelity (from 02-voice-evaluator.md)
├── reporting.py         # Report generation
└── cli.py              # Command-line interface

data/bob_loukas/
├── quiz_data.json       # Topic quizzes
├── chapter_tests.json   # Chapter tests (optional)
├── unit_exams.json      # Unit exams (optional)
└── final_assessment.json # Final assessment

eval_results/
├── eval_2025-12-26T10-30-00.json
├── eval_2025-12-26T10-30-00.md
└── ...
```

## Validation Checklist

- [ ] Pipeline loads all quiz levels correctly
- [ ] Voice evaluator integrates seamlessly
- [ ] Early termination triggers on critical failures
- [ ] Reports generate in all formats
- [ ] CLI works with both DPO and PPO models
- [ ] Recommendations are actionable

---

*Phase 2b - Bob Loukas Mindprint RLHF LoRA*
*Branch: shared*
*Created: December 2025*

