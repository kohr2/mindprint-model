#!/usr/bin/env python3
"""
Test script for transcript processing.

Verifies that transcript processing works correctly with downloaded files.
Tests both with and without episode summaries.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_prep.transcript_processor import TranscriptProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test transcript processing with downloaded files"
    )
    parser.add_argument(
        "--transcript-dir",
        type=str,
        default="data/bob_loukas/transcripts",
        help="Directory containing transcripts (default: data/bob_loukas/transcripts)",
    )
    parser.add_argument(
        "--summaries-dir",
        type=str,
        default=None,
        help="Directory containing summaries.json (optional)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of transcripts to process (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for test results (default: print to stdout)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize processor
    try:
        processor = TranscriptProcessor(
            transcripts_dir=args.transcript_dir,
            summaries_dir=args.summaries_dir,
        )
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        return 1

    # Process transcripts
    logger.info(f"Processing up to {args.count} transcripts...")
    questions = processor.process_all_transcripts()

    # Limit to requested count
    if len(questions) > args.count:
        questions = questions[:args.count]
        logger.info(f"Limited to {args.count} questions for testing")

    # Prepare output
    results = {
        "total_questions": len(questions),
        "questions": [],
    }

    for q in questions:
        results["questions"].append({
            "question": q.question,
            "reference_answer": q.reference_answer[:200] + "..." if len(q.reference_answer) > 200 else q.reference_answer,
            "source": q.source,
            "key_concepts": q.key_concepts,
        })

    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")
    else:
        print("\n" + "=" * 60)
        print("TRANSCRIPT PROCESSING TEST RESULTS")
        print("=" * 60)
        print(f"Total questions generated: {len(questions)}")
        print("\nSample Questions:")
        print("-" * 60)
        for i, q in enumerate(questions[:5], 1):
            print(f"\n{i}. Source: {q.source}")
            print(f"   Question: {q.question}")
            print(f"   Answer (first 150 chars): {q.reference_answer[:150]}...")
            if q.key_concepts:
                print(f"   Key concepts: {', '.join(q.key_concepts)}")
        print("=" * 60 + "\n")

    # Verification checks
    print("\nVERIFICATION CHECKS:")
    print("-" * 60)
    
    # Check 1: All questions have source
    sources_without = [q for q in questions if not q.source]
    if sources_without:
        print(f"❌ {len(sources_without)} questions missing source identifier")
    else:
        print(f"✅ All {len(questions)} questions have source identifiers")

    # Check 2: All questions have answers
    answers_without = [q for q in questions if not q.reference_answer]
    if answers_without:
        print(f"❌ {len(answers_without)} questions missing reference answers")
    else:
        print(f"✅ All {len(questions)} questions have reference answers")

    # Check 3: Answer length check
    short_answers = [q for q in questions if len(q.reference_answer) < 50]
    if short_answers:
        print(f"⚠️  {len(short_answers)} questions have very short answers (<50 chars)")
    else:
        print(f"✅ All answers are substantial (>=50 chars)")

    # Check 4: Source format check
    valid_sources = [q for q in questions if q.source.startswith("episode-")]
    if len(valid_sources) != len(questions):
        print(f"⚠️  {len(questions) - len(valid_sources)} questions have unexpected source format")
    else:
        print(f"✅ All sources follow expected format (episode-YYYY-MM-DD)")

    print("-" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
