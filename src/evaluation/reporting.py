"""
ReportGenerator - Generate evaluation reports in multiple formats.

Produces JSON, Markdown, and summary reports from evaluation results.
"""

from pathlib import Path
from typing import List, Optional
from datetime import datetime
import json
import logging

from .pipeline import EvaluationReport, LevelResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate evaluation reports in multiple formats."""

    def __init__(self, output_dir: str):
        """
        Initialize the report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, report: EvaluationReport) -> dict:
        """
        Generate all report formats.

        Args:
            report: EvaluationReport to generate reports from

        Returns:
            Dict with paths to generated files
        """
        paths = {}

        paths["json"] = self.generate_json(report)
        paths["markdown"] = self.generate_markdown(report)
        paths["summary"] = self.generate_summary(report)

        return paths

    def generate_json(self, report: EvaluationReport) -> Path:
        """
        Export complete report as JSON.

        Args:
            report: EvaluationReport to export

        Returns:
            Path to the generated JSON file
        """
        # Sanitize timestamp for filename
        safe_timestamp = report.timestamp.replace(":", "-").replace(".", "-")
        output_path = self.output_dir / f"eval_{safe_timestamp}.json"

        report_dict = report.to_dict()

        # Add detailed topic results
        report_dict["topic_details"] = [r.to_dict() for r in report.topic_results]
        report_dict["chapter_details"] = [r.to_dict() for r in report.chapter_results]
        report_dict["unit_details"] = [r.to_dict() for r in report.unit_results]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON report saved to {output_path}")
        return output_path

    def generate_markdown(self, report: EvaluationReport) -> Path:
        """
        Generate detailed markdown report.

        Args:
            report: EvaluationReport to export

        Returns:
            Path to the generated Markdown file
        """
        safe_timestamp = report.timestamp.replace(":", "-").replace(".", "-")
        output_path = self.output_dir / f"eval_{safe_timestamp}.md"

        lines = [
            f"# Evaluation Report: {report.model_name}",
            "",
            f"**Approach:** {report.approach.upper()}",
            f"**Timestamp:** {report.timestamp}",
            f"**Overall Result:** {'PASSED' if report.passed else 'FAILED'}",
            "",
            "## Summary Metrics",
            "",
            "| Metric | Value | Threshold |",
            "|--------|-------|-----------|",
            f"| Overall Accuracy | {report.overall_accuracy:.1%} | >=80% |",
            f"| Voice Fidelity | {report.overall_voice_score:.2f} | >=0.75 |",
            f"| Critical Violations | {len(report.critical_violations)} | 0 |",
            "",
        ]

        # Topic results
        if report.topic_results:
            lines.extend(
                [
                    "## Topic Results",
                    "",
                    "| Topic | Accuracy | Voice | Status |",
                    "|-------|----------|-------|--------|",
                ]
            )
            for r in report.topic_results:
                status = "PASS" if r.passed else "FAIL"
                lines.append(
                    f"| {r.identifier} | {r.accuracy:.1%} | {r.avg_voice_score:.2f} | {status} |"
                )
            lines.append("")

        # Chapter results
        if report.chapter_results:
            lines.extend(
                [
                    "## Chapter Results",
                    "",
                    "| Chapter | Accuracy | Voice | Status |",
                    "|---------|----------|-------|--------|",
                ]
            )
            for r in report.chapter_results:
                status = "PASS" if r.passed else "FAIL"
                lines.append(
                    f"| {r.identifier} | {r.accuracy:.1%} | {r.avg_voice_score:.2f} | {status} |"
                )
            lines.append("")

        # Unit results
        if report.unit_results:
            lines.extend(
                [
                    "## Unit Results",
                    "",
                    "| Unit | Accuracy | Voice | Status |",
                    "|------|----------|-------|--------|",
                ]
            )
            for r in report.unit_results:
                status = "PASS" if r.passed else "FAIL"
                lines.append(
                    f"| {r.identifier} | {r.accuracy:.1%} | {r.avg_voice_score:.2f} | {status} |"
                )
            lines.append("")

        # Final result
        if report.final_result:
            r = report.final_result
            status = "PASSED" if r.passed else "FAILED"
            lines.extend(
                [
                    "## Final Assessment",
                    "",
                    f"**Status:** {status}",
                    f"**Accuracy:** {r.accuracy:.1%}",
                    f"**Voice Fidelity:** {r.avg_voice_score:.2f}",
                    "",
                ]
            )

        # Recommendations
        if report.recommendations:
            lines.extend(
                [
                    "## Recommendations",
                    "",
                ]
            )
            for rec in report.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Critical violations
        if report.critical_violations:
            lines.extend(
                [
                    "## Critical Violations",
                    "",
                ]
            )
            for violation in report.critical_violations:
                lines.append(f"- {violation}")
            lines.append("")

        # Failed topics
        if report.failed_topics:
            lines.extend(
                [
                    "## Failed Topics",
                    "",
                ]
            )
            for topic in report.failed_topics:
                lines.append(f"- {topic}")
            lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Markdown report saved to {output_path}")
        return output_path

    def generate_summary(self, report: EvaluationReport) -> str:
        """
        Generate one-line summary for logs.

        Args:
            report: EvaluationReport to summarize

        Returns:
            Summary string
        """
        status = "PASS" if report.passed else "FAIL"
        summary = (
            f"[{status}] {report.model_name} ({report.approach}): "
            f"Acc={report.overall_accuracy:.1%}, "
            f"Voice={report.overall_voice_score:.2f}, "
            f"Violations={len(report.critical_violations)}"
        )

        # Also print to console
        print("\n" + "=" * 60)
        print(summary)
        print("=" * 60 + "\n")

        return summary

    def generate_comparison(
        self,
        reports: List[EvaluationReport],
        output_name: str = "comparison",
    ) -> Path:
        """
        Generate a comparison report across multiple evaluations.

        Args:
            reports: List of EvaluationReport objects to compare
            output_name: Base name for output file

        Returns:
            Path to the generated comparison file
        """
        output_path = self.output_dir / f"{output_name}.md"

        lines = [
            "# Model Comparison Report",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**Models Compared:** {len(reports)}",
            "",
            "## Overview",
            "",
            "| Model | Approach | Accuracy | Voice | Status |",
            "|-------|----------|----------|-------|--------|",
        ]

        for report in reports:
            status = "PASS" if report.passed else "FAIL"
            lines.append(
                f"| {report.model_name} | {report.approach} | "
                f"{report.overall_accuracy:.1%} | {report.overall_voice_score:.2f} | {status} |"
            )

        lines.append("")

        # Per-unit comparison
        if reports:
            units = set()
            for report in reports:
                for r in report.unit_results:
                    units.add(r.identifier)

            if units:
                lines.extend(
                    [
                        "## Per-Unit Comparison",
                        "",
                    ]
                )

                for unit in sorted(units):
                    lines.append(f"### {unit}")
                    lines.append("")
                    lines.append("| Model | Accuracy | Voice |")
                    lines.append("|-------|----------|-------|")

                    for report in reports:
                        unit_result = next(
                            (r for r in report.unit_results if r.identifier == unit),
                            None,
                        )
                        if unit_result:
                            lines.append(
                                f"| {report.model_name} | "
                                f"{unit_result.accuracy:.1%} | {unit_result.avg_voice_score:.2f} |"
                            )
                        else:
                            lines.append(f"| {report.model_name} | N/A | N/A |")

                    lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Comparison report saved to {output_path}")
        return output_path
