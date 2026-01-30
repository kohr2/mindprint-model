#!/usr/bin/env python3
"""
Validate dataset quality before training.

Checks that dataset meets quality thresholds:
- Examples per topic >= 10
- Voice marker density >= 20%
- Answer length 600-1200 chars
- Preference pairs mapped to topics (not "unknown")
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import logging
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.voice_markers import DEFAULT_VOICE_MARKERS
from src.data_prep.voice_aware_processor import VoiceAwareProcessor

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


def validate_sft_data(sft_file: Path, min_examples_per_topic: int = 10) -> Dict:
    """
    Validate SFT data quality.

    Args:
        sft_file: Path to sft_data.jsonl
        min_examples_per_topic: Minimum examples per topic

    Returns:
        Validation results dictionary
    """
    data = load_jsonl(sft_file)
    
    # Group by topic/source
    by_topic = defaultdict(list)
    for item in data:
        topic = item.get("source", "unknown")
        by_topic[topic].append(item)
    
    # Validate each topic
    validation_results = {
        "total_examples": len(data),
        "total_topics": len(by_topic),
        "topics_meeting_threshold": 0,
        "topics_below_threshold": 0,
        "topic_details": {},
        "overall_pass": True,
    }
    
    voice_processor = VoiceAwareProcessor()
    
    for topic, examples in by_topic.items():
        topic_results = {
            "example_count": len(examples),
            "meets_count_threshold": len(examples) >= min_examples_per_topic,
            "avg_voice_density": 0.0,
            "avg_length": 0.0,
            "meets_voice_threshold": False,
            "meets_length_threshold": False,
            "pass": False,
        }
        
        # Calculate average voice density and length
        voice_densities = []
        lengths = []
        
        for example in examples:
            output = example.get("output", "")
            metrics = voice_processor.validate_quality(output)
            voice_densities.append(metrics.voice_marker_density)
            lengths.append(metrics.length)
        
        if voice_densities:
            topic_results["avg_voice_density"] = sum(voice_densities) / len(voice_densities)
            topic_results["meets_voice_threshold"] = topic_results["avg_voice_density"] >= 20.0
        
        if lengths:
            topic_results["avg_length"] = sum(lengths) / len(lengths)
            topic_results["meets_length_threshold"] = (
                600 <= topic_results["avg_length"] <= 1200
            )
        
        # Topic passes if all thresholds met
        topic_results["pass"] = (
            topic_results["meets_count_threshold"]
            and topic_results["meets_voice_threshold"]
            and topic_results["meets_length_threshold"]
        )
        
        validation_results["topic_details"][topic] = topic_results
        
        if topic_results["pass"]:
            validation_results["topics_meeting_threshold"] += 1
        else:
            validation_results["topics_below_threshold"] += 1
            validation_results["overall_pass"] = False
    
    return validation_results


def validate_preference_data(pref_file: Path) -> Dict:
    """
    Validate preference pair data quality.

    Args:
        pref_file: Path to preference_data.jsonl

    Returns:
        Validation results dictionary
    """
    data = load_jsonl(pref_file)
    
    # Group by source/topic
    by_topic = defaultdict(list)
    unknown_count = 0
    
    for item in data:
        source = item.get("source", "unknown")
        if not source or source == "unknown":
            unknown_count += 1
        by_topic[source].append(item)
    
    validation_results = {
        "total_pairs": len(data),
        "total_topics": len(by_topic),
        "unknown_pairs": unknown_count,
        "topics_with_sufficient_pairs": 0,
        "topics_below_threshold": 0,
        "overall_pass": True,
    }
    
    # Check each topic has sufficient pairs (target: 10+)
    min_pairs_per_topic = 10
    for topic, pairs in by_topic.items():
        if topic == "unknown":
            validation_results["overall_pass"] = False
            validation_results["topics_below_threshold"] += 1
        elif len(pairs) >= min_pairs_per_topic:
            validation_results["topics_with_sufficient_pairs"] += 1
        else:
            validation_results["overall_pass"] = False
            validation_results["topics_below_threshold"] += 1
    
    # Overall fails if too many unknown pairs
    if unknown_count > len(data) * 0.1:  # More than 10% unknown
        validation_results["overall_pass"] = False
    
    return validation_results


def print_validation_report(sft_results: Dict, pref_results: Dict):
    """Print validation report."""
    print("\n" + "=" * 80)
    print("DATASET QUALITY VALIDATION REPORT")
    print("=" * 80)
    
    # SFT Data Results
    print(f"\nSFT Data Validation:")
    print(f"  Total Examples: {sft_results['total_examples']}")
    print(f"  Total Topics: {sft_results['total_topics']}")
    print(f"  Topics Meeting Threshold: {sft_results['topics_meeting_threshold']}")
    print(f"  Topics Below Threshold: {sft_results['topics_below_threshold']}")
    
    if sft_results['topics_below_threshold'] > 0:
        print(f"\n  ⚠️  Topics Below Threshold:")
        for topic, details in sft_results['topic_details'].items():
            if not details['pass']:
                issues = []
                if not details['meets_count_threshold']:
                    issues.append(f"count ({details['example_count']} < 10)")
                if not details['meets_voice_threshold']:
                    issues.append(f"voice ({details['avg_voice_density']:.1f}% < 20%)")
                if not details['meets_length_threshold']:
                    issues.append(f"length ({details['avg_length']:.0f} not in 600-1200)")
                print(f"    - {topic}: {', '.join(issues)}")
    
    # Preference Data Results
    print(f"\nPreference Data Validation:")
    print(f"  Total Pairs: {pref_results['total_pairs']}")
    print(f"  Total Topics: {pref_results['total_topics']}")
    print(f"  Unknown Pairs: {pref_results['unknown_pairs']}")
    print(f"  Topics with Sufficient Pairs: {pref_results['topics_with_sufficient_pairs']}")
    print(f"  Topics Below Threshold: {pref_results['topics_below_threshold']}")
    
    if pref_results['unknown_pairs'] > 0:
        print(f"\n  ⚠️  Found {pref_results['unknown_pairs']} pairs with 'unknown' source")
        print(f"     Preference pairs should be mapped to specific topics")
    
    # Overall Result
    print("\n" + "=" * 80)
    overall_pass = sft_results['overall_pass'] and pref_results['overall_pass']
    if overall_pass:
        print("✅ DATASET PASSES QUALITY VALIDATION")
    else:
        print("❌ DATASET FAILS QUALITY VALIDATION")
        print("\nPlease address the issues above before training.")
    print("=" * 80 + "\n")
    
    return overall_pass


def main():
    parser = argparse.ArgumentParser(description="Validate dataset quality")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to data directory containing sft_data.jsonl and preference_data.jsonl"
    )
    parser.add_argument(
        "--min-examples",
        type=int,
        default=10,
        help="Minimum examples per topic (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save validation report JSON (optional)"
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
    
    logger.info("Validating SFT data...")
    sft_results = validate_sft_data(sft_file, args.min_examples)
    
    logger.info("Validating preference data...")
    pref_results = validate_preference_data(pref_file)
    
    # Print report
    overall_pass = print_validation_report(sft_results, pref_results)
    
    # Save JSON report if requested
    if args.output:
        report = {
            "sft_validation": sft_results,
            "preference_validation": pref_results,
            "overall_pass": overall_pass,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Validation report saved to {output_path}")
    
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
