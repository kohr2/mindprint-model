#!/usr/bin/env python3
"""
Comprehensive training results analysis.

Combines checkpoint analysis and log analysis to generate:
- Complete training summary report
- Root cause analysis
- Actionable recommendations
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def analyze_thresholds(checkpoint_data: Dict, config_path: Optional[Path] = None) -> Dict:
    """Analyze threshold configuration and score distributions."""
    import yaml
    
    # Load thresholds from config if available
    thresholds = {
        "accuracy_threshold": 0.70,
        "dpo_trigger_threshold": 0.70,
        "topic_pass_threshold": 0.70,
    }
    
    if config_path and config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
            if "thresholds" in config:
                thresholds.update(config["thresholds"])
    
    # Extract scores from checkpoint
    topics = []
    if "result" in checkpoint_data and "units" in checkpoint_data["result"]:
        for unit in checkpoint_data["result"]["units"]:
            for chapter in unit.get("chapters", []):
                for topic in chapter.get("topics", []):
                    topics.append({
                        "accuracy": topic.get("accuracy_score", 0.0),
                        "voice": topic.get("voice_score", 0.0),
                    })
    
    # Analyze distribution
    if topics:
        acc_scores = [t["accuracy"] for t in topics]
        voice_scores = [t["voice"] for t in topics]
        
        analysis = {
            "thresholds": thresholds,
            "accuracy_distribution": {
                "min": min(acc_scores),
                "max": max(acc_scores),
                "mean": sum(acc_scores) / len(acc_scores),
                "below_threshold": sum(1 for a in acc_scores if a < thresholds["accuracy_threshold"]),
                "above_threshold": sum(1 for a in acc_scores if a >= thresholds["accuracy_threshold"]),
            },
            "voice_distribution": {
                "min": min(voice_scores),
                "max": max(voice_scores),
                "mean": sum(voice_scores) / len(voice_scores),
                "below_threshold": sum(1 for v in voice_scores if v < thresholds["dpo_trigger_threshold"]),
                "above_threshold": sum(1 for v in voice_scores if v >= thresholds["dpo_trigger_threshold"]),
            },
        }
        
        # Check if thresholds are too strict
        pass_rate = checkpoint_data.get("result", {}).get("pass_rate", 0.0)
        analysis["threshold_assessment"] = {
            "pass_rate": pass_rate,
            "too_strict": pass_rate < 0.30,
            "appropriate": 0.30 <= pass_rate <= 0.70,
            "too_loose": pass_rate > 0.70,
        }
        
        return analysis
    
    return {"thresholds": thresholds, "error": "No topic data found"}


def analyze_data_quality(data_dir: Path) -> Dict:
    """Analyze training data quality."""
    import json
    
    analysis = {
        "sft_examples": 0,
        "preference_pairs": 0,
        "topics_covered": set(),
        "examples_per_topic": {},
    }
    
    sft_file = data_dir / "transcripts" / "sft_data.jsonl"
    pref_file = data_dir / "transcripts" / "preference_data.jsonl"
    
    if sft_file.exists():
        with open(sft_file) as f:
            for line in f:
                if line.strip():
                    example = json.loads(line)
                    analysis["sft_examples"] += 1
                    topic = example.get("source", example.get("topic_id", "unknown"))
                    analysis["topics_covered"].add(topic)
                    analysis["examples_per_topic"][topic] = analysis["examples_per_topic"].get(topic, 0) + 1
    
    if pref_file.exists():
        with open(pref_file) as f:
            for line in f:
                if line.strip():
                    analysis["preference_pairs"] += 1
    
    analysis["topics_covered"] = len(analysis["topics_covered"])
    
    # Assess quality
    min_examples = min(analysis["examples_per_topic"].values()) if analysis["examples_per_topic"] else 0
    max_examples = max(analysis["examples_per_topic"].values()) if analysis["examples_per_topic"] else 0
    avg_examples = sum(analysis["examples_per_topic"].values()) / len(analysis["examples_per_topic"]) if analysis["examples_per_topic"] else 0
    
    analysis["quality_assessment"] = {
        "sufficient_data": min_examples >= 5,
        "min_examples_per_topic": min_examples,
        "max_examples_per_topic": max_examples,
        "avg_examples_per_topic": avg_examples,
        "needs_more_data": min_examples < 5,
    }
    
    return analysis


def generate_recommendations(
    checkpoint_analysis: Dict,
    log_analysis: Dict,
    threshold_analysis: Dict,
    data_analysis: Dict,
) -> List[Dict]:
    """Generate actionable recommendations."""
    recommendations = []
    
    pass_rate = checkpoint_analysis.get("overall_metrics", {}).get("pass_rate", 0.0)
    
    # Low pass rate recommendations
    if pass_rate < 0.30:
        recommendations.append({
            "priority": "HIGH",
            "category": "Thresholds",
            "issue": f"Very low pass rate ({pass_rate:.1%})",
            "recommendation": "Consider lowering thresholds temporarily to 0.65 to identify if thresholds are the issue",
            "impact": "High - May reveal if thresholds are blocking progress",
        })
    
    # Data quality recommendations
    if data_analysis.get("quality_assessment", {}).get("needs_more_data", False):
        min_examples = data_analysis.get("quality_assessment", {}).get("min_examples_per_topic", 0)
        recommendations.append({
            "priority": "HIGH",
            "category": "Data",
            "issue": f"Insufficient data: minimum {min_examples} examples per topic",
            "recommendation": "Add more training examples (target: 10+ SFT examples, 10+ preference pairs per topic)",
            "impact": "High - Data quantity directly affects model performance",
        })
    
    # DPO issues
    negative_dpo = len(checkpoint_analysis.get("patterns", {}).get("negative_dpo_loss", []))
    if negative_dpo > 0:
        recommendations.append({
            "priority": "MEDIUM",
            "category": "DPO",
            "issue": f"{negative_dpo} topics with negative DPO loss",
            "recommendation": "Review DPO loss computation and consider lowering DPO learning rate",
            "impact": "Medium - May indicate training instability",
        })
    
    # High accuracy, low voice pattern
    high_acc_low_voice = len(checkpoint_analysis.get("patterns", {}).get("high_accuracy_low_voice", []))
    if high_acc_low_voice > 0:
        recommendations.append({
            "priority": "MEDIUM",
            "category": "DPO",
            "issue": f"{high_acc_low_voice} topics with high accuracy but low voice",
            "recommendation": "Review DPO trigger threshold and preference data quality for these topics",
            "impact": "Medium - DPO may not be triggering effectively",
        })
    
    # SFT convergence issues
    if log_analysis.get("sft_loss_progressions"):
        converged = sum(1 for p in log_analysis["sft_loss_progressions"].values() if p.get("converged", False))
        total = len(log_analysis["sft_loss_progressions"])
        if converged / total < 0.5:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Hyperparameters",
                "issue": f"Only {converged}/{total} topics showing SFT convergence",
                "recommendation": "Consider increasing SFT epochs or adjusting learning rate",
                "impact": "Medium - Poor convergence may indicate insufficient training",
            })
    
    return recommendations


def generate_markdown_report(
    checkpoint_analysis: Dict,
    log_analysis: Dict,
    threshold_analysis: Dict,
    data_analysis: Dict,
    recommendations: List[Dict],
    output_path: Path,
):
    """Generate comprehensive markdown report."""
    report = []
    
    report.append("# Training Results Analysis Report\n\n")
    report.append(f"Generated: {Path.cwd()}\n\n")
    
    # Executive Summary
    overall = checkpoint_analysis.get("overall_metrics", {})
    report.append("## Executive Summary\n\n")
    report.append(f"- **Pass Rate**: {overall.get('pass_rate', 0.0):.1%}\n")
    report.append(f"- **Total Topics**: {overall.get('total_topics', 0)}\n")
    report.append(f"- **Passed Topics**: {overall.get('passed_topics', 0)}\n")
    report.append(f"- **Training Time**: {overall.get('total_training_time_hours', 0):.2f} hours\n\n")
    
    # Threshold Analysis
    report.append("## Threshold Analysis\n\n")
    thresholds = threshold_analysis.get("thresholds", {})
    report.append(f"- Accuracy Threshold: {thresholds.get('accuracy_threshold', 0.70)}\n")
    report.append(f"- DPO Trigger Threshold: {thresholds.get('dpo_trigger_threshold', 0.70)}\n")
    report.append(f"- Topic Pass Threshold: {thresholds.get('topic_pass_threshold', 0.70)}\n\n")
    
    if "threshold_assessment" in threshold_analysis:
        assessment = threshold_analysis["threshold_assessment"]
        report.append(f"**Assessment**: Pass rate is {assessment.get('pass_rate', 0.0):.1%}\n")
        if assessment.get("too_strict"):
            report.append("- ⚠️ Thresholds may be too strict\n")
        elif assessment.get("too_loose"):
            report.append("- ✅ Thresholds may be too loose (good performance)\n")
        else:
            report.append("- ✅ Thresholds appear appropriate\n")
    
    # Data Quality
    report.append("\n## Data Quality Analysis\n\n")
    report.append(f"- **SFT Examples**: {data_analysis.get('sft_examples', 0)}\n")
    report.append(f"- **Preference Pairs**: {data_analysis.get('preference_pairs', 0)}\n")
    report.append(f"- **Topics Covered**: {data_analysis.get('topics_covered', 0)}\n")
    
    if "quality_assessment" in data_analysis:
        qa = data_analysis["quality_assessment"]
        report.append(f"- **Min Examples/Topic**: {qa.get('min_examples_per_topic', 0)}\n")
        report.append(f"- **Avg Examples/Topic**: {qa.get('avg_examples_per_topic', 0):.1f}\n")
        if qa.get("needs_more_data"):
            report.append("- ⚠️ **Insufficient data** - Need at least 5 examples per topic\n")
        else:
            report.append("- ✅ **Sufficient data**\n")
    
    # Patterns
    report.append("\n## Key Patterns\n\n")
    patterns = checkpoint_analysis.get("patterns", {})
    report.append(f"- High Accuracy, Low Voice: {len(patterns.get('high_accuracy_low_voice', []))} topics\n")
    report.append(f"- Low Accuracy, High Voice: {len(patterns.get('low_accuracy_high_voice', []))} topics\n")
    report.append(f"- Both Low: {len(patterns.get('both_low', []))} topics\n")
    report.append(f"- DPO Triggered: {len(patterns.get('dpo_triggered', []))} topics\n")
    report.append(f"- Negative DPO Loss: {len(patterns.get('negative_dpo_loss', []))} topics\n")
    
    # Recommendations
    report.append("\n## Recommendations\n\n")
    for i, rec in enumerate(recommendations, 1):
        report.append(f"### {i}. [{rec['priority']}] {rec['category']}: {rec['issue']}\n\n")
        report.append(f"**Recommendation**: {rec['recommendation']}\n\n")
        report.append(f"**Impact**: {rec['impact']}\n\n")
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.writelines(report)
    
    logger.info(f"Markdown report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive training results analysis")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/latest.json",
        help="Path to checkpoint JSON file"
    )
    parser.add_argument(
        "--checkpoint-analysis",
        type=str,
        help="Path to checkpoint analysis JSON (if already generated)"
    )
    parser.add_argument(
        "--log",
        type=str,
        default="./logs/training.log",
        help="Path to training log file"
    )
    parser.add_argument(
        "--log-analysis",
        type=str,
        help="Path to log analysis JSON (if already generated)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/training_pipeline.yaml",
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/bob_loukas",
        help="Path to data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./analysis/training_results_analysis",
        help="Output directory for reports"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint analysis
    if args.checkpoint_analysis:
        logger.info(f"Loading checkpoint analysis from {args.checkpoint_analysis}")
        checkpoint_analysis = load_json(Path(args.checkpoint_analysis))
    else:
        logger.info("Running checkpoint analysis...")
        import sys
        
        # Import functions from analyze_checkpoint module
        sys.path.insert(0, str(Path(__file__).parent))
        from analyze_checkpoint import (
            analyze_overall_metrics,
            extract_topic_metrics,
            analyze_by_unit,
            identify_patterns,
            load_checkpoint,
        )
        
        checkpoint_data = load_checkpoint(Path(args.checkpoint))
        overall = analyze_overall_metrics(checkpoint_data["result"])
        topics = extract_topic_metrics(checkpoint_data["result"])
        by_unit = analyze_by_unit(topics)
        patterns = identify_patterns(topics)
        
        checkpoint_analysis = {
            "overall_metrics": overall,
            "topics": topics,
            "by_unit": by_unit,
            "patterns": patterns,
        }
    
    # Load log analysis
    if args.log_analysis:
        logger.info(f"Loading log analysis from {args.log_analysis}")
        log_analysis = load_json(Path(args.log_analysis))
    else:
        logger.info("Running log analysis...")
        import sys
        
        # Import function from analyze_training_logs module
        sys.path.insert(0, str(Path(__file__).parent))
        from analyze_training_logs import analyze_logs
        
        log_analysis = analyze_logs(Path(args.log))
    
    # Analyze thresholds
    logger.info("Analyzing thresholds...")
    threshold_analysis = analyze_thresholds(
        load_json(Path(args.checkpoint)),
        Path(args.config) if Path(args.config).exists() else None
    )
    
    # Analyze data quality
    logger.info("Analyzing data quality...")
    data_analysis = analyze_data_quality(Path(args.data_dir))
    
    # Generate recommendations
    logger.info("Generating recommendations...")
    recommendations = generate_recommendations(
        checkpoint_analysis,
        log_analysis,
        threshold_analysis,
        data_analysis,
    )
    
    # Save JSON report
    json_report = {
        "checkpoint_analysis": checkpoint_analysis,
        "log_analysis": log_analysis,
        "threshold_analysis": threshold_analysis,
        "data_analysis": data_analysis,
        "recommendations": recommendations,
    }
    
    json_path = output_dir / "analysis.json"
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    logger.info(f"JSON report saved to {json_path}")
    
    # Generate markdown report
    md_path = output_dir / "analysis_report.md"
    generate_markdown_report(
        checkpoint_analysis,
        log_analysis,
        threshold_analysis,
        data_analysis,
        recommendations,
        md_path,
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nReports saved to: {output_dir}")
    print(f"  - JSON: {json_path}")
    print(f"  - Markdown: {md_path}")
    print(f"\nRecommendations: {len(recommendations)}")
    for rec in recommendations:
        print(f"  [{rec['priority']}] {rec['category']}: {rec['issue']}")
    print("\n" + "=" * 80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
