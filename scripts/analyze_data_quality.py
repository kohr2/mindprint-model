#!/usr/bin/env python3
"""
Analyze training data quality by topic/unit.

Extracts metrics:
- Number of SFT examples per topic
- Number of preference pairs per topic
- Average output length
- Voice marker density
- Preference quality score (chosen vs rejected differentiation)
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file into list of dicts."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def analyze_output_length(text: str) -> int:
    """Get character count of output."""
    return len(text)


def analyze_voice_markers(text: str) -> float:
    """
    Compute voice marker density.

    Returns percentage of words that are voice markers.
    """
    markers = [
        "Look,", "Okay,", "Here's the thing", "I've seen",
        "The reality is", "**", "—", "right?", "fine!",
        "systematic", "discipline", "thesis", "conviction",
        "Here's", "The key", "psychology", "gambler"
    ]

    marker_count = sum(text.lower().count(m.lower()) for m in markers)
    words = len(text.split())

    # Markers per 100 words
    return (marker_count / words * 100) if words > 0 else 0.0


def analyze_preference_quality(chosen: str, rejected: str) -> float:
    """
    Compute quality differentiation score.

    Higher score = clearer quality gap between chosen and rejected.
    """
    chosen_len = len(chosen)
    rejected_len = len(rejected)

    chosen_markers = analyze_voice_markers(chosen)
    rejected_markers = analyze_voice_markers(rejected)

    # Score based on:
    # 1. Length difference (chosen should be more substantive)
    # 2. Voice marker difference (chosen should have more)

    length_ratio = chosen_len / (rejected_len + 1)  # Avoid div by zero
    marker_diff = chosen_markers - rejected_markers

    # Weighted combination
    quality_score = (min(length_ratio, 3.0) * 0.7) + (marker_diff * 0.3)

    return quality_score


def group_by_topic(data: List[Dict], field: str = "source") -> Dict[str, List[Dict]]:
    """Group data by topic/source field."""
    grouped = defaultdict(list)
    for item in data:
        topic = item.get(field, item.get("topic_id", "unknown"))
        grouped[topic].append(item)
    return dict(grouped)


def analyze_sft_data(sft_file: Path) -> Dict:
    """Analyze SFT training data."""
    data = load_jsonl(sft_file)
    by_topic = group_by_topic(data, "source")

    analysis = {}
    for topic, examples in by_topic.items():
        outputs = [ex.get("output", "") for ex in examples]
        instructions = [ex.get("instruction", ex.get("question", "")) for ex in examples]

        analysis[topic] = {
            "example_count": len(examples),
            "avg_instruction_length": sum(len(i) for i in instructions) / len(instructions) if instructions else 0,
            "avg_output_length": sum(len(o) for o in outputs) / len(outputs) if outputs else 0,
            "min_output_length": min(len(o) for o in outputs) if outputs else 0,
            "max_output_length": max(len(o) for o in outputs) if outputs else 0,
            "avg_voice_markers": sum(analyze_voice_markers(o) for o in outputs) / len(outputs) if outputs else 0,
        }

    return analysis


def analyze_preference_data(pref_file: Path) -> Dict:
    """Analyze preference pair data."""
    data = load_jsonl(pref_file)

    # Group by source/topic if available
    by_topic = group_by_topic(data, "source")

    topic_analysis = {}
    for topic, pairs in by_topic.items():
        quality_scores = []
        chosen_lengths = []
        rejected_lengths = []

        for pair in pairs:
            chosen = pair.get("chosen", "")
            rejected = pair.get("rejected", "")
            quality = analyze_preference_quality(chosen, rejected)
            quality_scores.append(quality)
            chosen_lengths.append(len(chosen))
            rejected_lengths.append(len(rejected))

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        topic_analysis[topic] = {
            "total_pairs": len(pairs),
            "avg_quality_score": avg_quality,
            "high_quality_pairs": sum(1 for q in quality_scores if q > 1.5),
            "low_quality_pairs": sum(1 for q in quality_scores if q < 0.5),
            "avg_chosen_length": sum(chosen_lengths) / len(chosen_lengths) if chosen_lengths else 0,
            "avg_rejected_length": sum(rejected_lengths) / len(rejected_lengths) if rejected_lengths else 0,
            "quality_ratio": (sum(chosen_lengths) / len(chosen_lengths)) / (sum(rejected_lengths) / len(rejected_lengths) + 1) if rejected_lengths else 0,
        }

    # Overall stats
    all_quality_scores = []
    for pairs in by_topic.values():
        for pair in pairs:
            chosen = pair.get("chosen", "")
            rejected = pair.get("rejected", "")
            quality = analyze_preference_quality(chosen, rejected)
            all_quality_scores.append(quality)

    topic_analysis["_overall"] = {
        "total_pairs": len(data),
        "avg_quality_score": sum(all_quality_scores) / len(all_quality_scores) if all_quality_scores else 0.0,
        "high_quality_pairs": sum(1 for q in all_quality_scores if q > 1.5),
        "low_quality_pairs": sum(1 for q in all_quality_scores if q < 0.5),
        "topics_count": len(by_topic),
    }

    return topic_analysis


def generate_report(sft_analysis: Dict, pref_analysis: Dict, output_path: Path) -> None:
    """Generate markdown report."""
    report = []
    report.append("# Training Data Quality Analysis\n\n")
    report.append(f"Generated from: {Path.cwd()}\n\n")

    # SFT Data Analysis
    report.append("## SFT Data by Topic\n\n")
    report.append("| Topic | Examples | Avg Instruction | Avg Output | Voice Markers (%) |\n")
    report.append("|-------|----------|-----------------|------------|-------------------|\n")

    for topic, metrics in sorted(sft_analysis.items(), key=lambda x: x[1]['example_count'], reverse=True):
        report.append(
            f"| {topic} | {metrics['example_count']} | "
            f"{metrics['avg_instruction_length']:.0f} | "
            f"{metrics['avg_output_length']:.0f} | "
            f"{metrics['avg_voice_markers']:.1f} |\n"
        )

    # Preference Data Analysis
    report.append("\n## Preference Data Quality by Topic\n\n")
    report.append("| Topic | Pairs | Avg Quality | High Quality | Quality Ratio |\n")
    report.append("|-------|-------|-------------|--------------|---------------|\n")

    for topic, metrics in sorted(pref_analysis.items()):
        if topic == "_overall":
            continue
        report.append(
            f"| {topic} | {metrics['total_pairs']} | "
            f"{metrics['avg_quality_score']:.2f} | "
            f"{metrics['high_quality_pairs']} ({metrics['high_quality_pairs']/metrics['total_pairs']*100:.1f}%) | "
            f"{metrics['quality_ratio']:.2f}x |\n"
        )

    # Overall Stats
    overall = pref_analysis.get("_overall", {})
    report.append("\n## Overall Preference Data Quality\n\n")
    report.append(f"- **Total pairs**: {overall.get('total_pairs', 0)}\n")
    report.append(f"- **Topics**: {overall.get('topics_count', 0)}\n")
    report.append(f"- **Average quality score**: {overall.get('avg_quality_score', 0):.2f}\n")
    report.append(f"- **High quality pairs (>1.5)**: {overall.get('high_quality_pairs', 0)} ({overall.get('high_quality_pairs', 0) / overall.get('total_pairs', 1) * 100:.1f}%)\n")
    report.append(f"- **Low quality pairs (<0.5)**: {overall.get('low_quality_pairs', 0)} ({overall.get('low_quality_pairs', 0) / overall.get('total_pairs', 1) * 100:.1f}%)\n")

    # Key Insights
    report.append("\n## Key Insights\n\n")

    # Find best performing topics (by example count and voice markers)
    best_topics = sorted(
        sft_analysis.items(),
        key=lambda x: (x[1]['example_count'], x[1]['avg_voice_markers']),
        reverse=True
    )[:3]

    report.append("### Top Topics (by example count and voice markers):\n\n")
    for topic, metrics in best_topics:
        report.append(
            f"- **{topic}**: {metrics['example_count']} examples, "
            f"{metrics['avg_output_length']:.0f} avg chars, "
            f"{metrics['avg_voice_markers']:.1f}% voice markers\n"
        )

    # Recommendations
    report.append("\n### Recommendations for Training\n\n")
    report.append("Based on the analysis, optimal topics should have:\n\n")
    report.append("1. **15-25 preference pairs** (concentrated signal)\n")
    report.append("2. **600-1200 character outputs** (substantive but focused)\n")
    report.append("3. **>20% voice marker density** (strong distinctive voice)\n")
    report.append("4. **>80% clear quality differentiation** (quality ratio >1.5)\n")
    report.append("\nTopics outside these ranges may benefit from:\n")
    report.append("- More training epochs (if few examples)\n")
    report.append("- Lower pass thresholds (if weak voice markers)\n")
    report.append("- Data quality improvements (if low quality ratio)\n")

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.writelines(report)

    logger.info(f"Report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze training data quality")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/bob_loukas",
        help="Data directory containing sft_data.jsonl and preference_data.jsonl"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./docs/data_quality_report.md",
        help="Output path for analysis report"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sft_file = data_dir / "sft_data.jsonl"
    pref_file = data_dir / "preference_data.jsonl"

    if not sft_file.exists():
        logger.error(f"SFT data file not found: {sft_file}")
        return 1

    if not pref_file.exists():
        logger.error(f"Preference data file not found: {pref_file}")
        return 1

    logger.info("Analyzing SFT data...")
    sft_analysis = analyze_sft_data(sft_file)

    logger.info("Analyzing preference data...")
    pref_analysis = analyze_preference_data(pref_file)

    logger.info("Generating report...")
    output_path = Path(args.output)
    generate_report(sft_analysis, pref_analysis, output_path)

    logger.info("Analysis complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
