#!/usr/bin/env python3
"""
Data Preparation Script for Bob Loukas Mindprint Training.

Converts the Bob Loukas textbook into training-ready data:
- SFT training pairs (instruction/output)
- Preference pairs for DPO (chosen/rejected)
- Quiz data for evaluation
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_prep.pipeline import DataPipeline, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare training data from Bob Loukas textbook",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with question augmentation
  python scripts/run_data_prep.py \\
    --textbook ../omnia/projects/bob_loukas/textbook \\
    --output ./data/bob_loukas

  # Run without question augmentation (faster, for testing)
  python scripts/run_data_prep.py \\
    --textbook ../omnia/projects/bob_loukas/textbook \\
    --output ./data/bob_loukas \\
    --no-augment

  # Specify target questions per topic
  python scripts/run_data_prep.py \\
    --textbook ../omnia/projects/bob_loukas/textbook \\
    --output ./data/bob_loukas \\
    --target-questions 15
""",
    )

    parser.add_argument(
        "--textbook",
        default="../omnia/projects/bob_loukas/textbook",
        help="Path to Bob Loukas textbook directory (default: ../omnia/projects/bob_loukas/textbook)",
    )
    parser.add_argument(
        "--output",
        default="./data/bob_loukas",
        help="Output directory for generated data (default: ./data/bob_loukas)",
    )
    parser.add_argument(
        "--target-questions",
        type=int,
        default=10,
        help="Target number of questions per topic (default: 10)",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Skip question augmentation (use existing questions only)",
    )
    parser.add_argument(
        "--no-critical",
        action="store_true",
        help="Skip critical distinction pairs",
    )
    parser.add_argument(
        "--api-key",
        help="Anthropic API key for question generation (default: ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and show statistics without generating output",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate paths
    textbook_path = Path(args.textbook)
    if not textbook_path.exists():
        logger.error(f"Textbook directory not found: {textbook_path}")
        logger.error("Make sure the path to the Bob Loukas textbook is correct.")
        return 1

    curriculum_path = textbook_path / "curriculum.yaml"
    if not curriculum_path.exists():
        logger.error(f"curriculum.yaml not found in {textbook_path}")
        return 1

    # Show configuration
    print("\n" + "=" * 60)
    print("DATA PREPARATION CONFIGURATION")
    print("=" * 60)
    print(f"Textbook path:     {textbook_path.absolute()}")
    print(f"Output path:       {Path(args.output).absolute()}")
    print(f"Target questions:  {args.target_questions}")
    print(f"Augment questions: {not args.no_augment}")
    print(f"Critical pairs:    {not args.no_critical}")
    print(f"Dry run:           {args.dry_run}")
    print("=" * 60 + "\n")

    # Dry run - just show statistics
    if args.dry_run:
        from src.data_prep.textbook_parser import TextbookParser

        parser = TextbookParser(str(textbook_path))
        stats = parser.get_statistics()

        print("TEXTBOOK STATISTICS")
        print("-" * 40)
        print(f"Topics:               {stats['topics']}")
        print(f"Chapters:             {stats['chapters']}")
        print(f"Units:                {stats['units']}")
        print(f"Topic questions:      {stats['topic_questions']}")
        print(f"Chapter questions:    {stats['chapter_questions']}")
        print(f"Unit questions:       {stats['unit_questions']}")
        print(f"Total questions:      {stats['total_questions']}")
        print(f"Avg per topic:        {stats['avg_questions_per_topic']:.1f}")
        print("-" * 40)

        if stats['avg_questions_per_topic'] < args.target_questions:
            questions_needed = int(
                (args.target_questions - stats['avg_questions_per_topic']) * stats['topics']
            )
            print(f"\nNote: ~{questions_needed} questions need to be generated")
            print("to reach the target of {args.target_questions} per topic.")

        return 0

    # Run the pipeline
    try:
        config = PipelineConfig(
            textbook_path=str(textbook_path),
            output_path=args.output,
            target_questions_per_topic=args.target_questions,
            augment_questions=not args.no_augment,
            include_critical_distinctions=not args.no_critical,
            api_key=args.api_key,
        )

        pipeline = DataPipeline(config)
        stats = pipeline.run()

        print("\nData preparation complete!")
        print(f"Output saved to: {Path(args.output).absolute()}")
        return 0

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
