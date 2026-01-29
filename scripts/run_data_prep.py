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
        default=None,
        help="Path to Bob Loukas textbook directory (required for textbook or combined mode)",
    )
    parser.add_argument(
        "--transcript-dir",
        default=None,
        help="Path to transcripts directory (required for transcript or combined mode)",
    )
    parser.add_argument(
        "--summaries-dir",
        default=None,
        help="Path to episode summaries directory (optional, from mindprint-agent)",
    )
    parser.add_argument(
        "--output",
        default="./data/bob_loukas",
        help="Output directory for generated data (default: ./data/bob_loukas)",
    )
    parser.add_argument(
        "--mode",
        choices=["textbook", "transcripts", "combined"],
        default="textbook",
        help="Data preparation mode (default: textbook)",
    )
    parser.add_argument(
        "--textbook-ratio",
        type=float,
        default=0.6,
        help="Ratio of textbook data when combining (default: 0.6 = 60%% textbook, 40%% transcripts)",
    )
    parser.add_argument(
        "--target-questions",
        type=int,
        default=10,
        help="Target number of questions per topic (default: 10)",
    )
    parser.add_argument(
        "--target-episode-questions",
        type=int,
        default=15,
        help="Target number of questions per episode (default: 15)",
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

    # Validate paths based on mode
    use_textbook = args.mode in ["textbook", "combined"]
    use_transcripts = args.mode in ["transcripts", "combined"]

    textbook_path = None
    if use_textbook:
        if not args.textbook:
            logger.error("--textbook required for textbook or combined mode")
            return 1
        textbook_path = Path(args.textbook)
        if not textbook_path.exists():
            logger.error(f"Textbook directory not found: {textbook_path}")
            return 1
        curriculum_path = textbook_path / "curriculum.yaml"
        if not curriculum_path.exists():
            logger.error(f"curriculum.yaml not found in {textbook_path}")
            return 1

    transcript_dir = None
    if use_transcripts:
        if not args.transcript_dir:
            logger.error("--transcript-dir required for transcripts or combined mode")
            return 1
        transcript_dir = args.transcript_dir
        transcript_path = Path(transcript_dir)
        if not transcript_path.exists():
            logger.error(f"Transcript directory not found: {transcript_path}")
            return 1

    # Show configuration
    print("\n" + "=" * 60)
    print("DATA PREPARATION CONFIGURATION")
    print("=" * 60)
    print(f"Mode:              {args.mode}")
    if textbook_path:
        print(f"Textbook path:     {textbook_path.absolute()}")
    if transcript_dir:
        print(f"Transcript dir:    {Path(transcript_dir).absolute()}")
    print(f"Output path:       {Path(args.output).absolute()}")
    print(f"Target questions:  {args.target_questions}")
    print(f"Target episodes:   {args.target_episode_questions}")
    print(f"Augment questions: {not args.no_augment}")
    print(f"Critical pairs:    {not args.no_critical}")
    if args.mode == "combined":
        print(f"Textbook ratio:    {args.textbook_ratio}")
    print(f"Dry run:           {args.dry_run}")
    print("=" * 60 + "\n")

    # Dry run - just show statistics
    if args.dry_run:
        if use_textbook:
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

        if use_transcripts:
            from src.data_prep.transcript_processor import TranscriptProcessor
            processor = TranscriptProcessor(
                transcripts_dir=transcript_dir,
                summaries_dir=args.summaries_dir,
            )
            raw_dir = Path(transcript_dir) / "raw"
            transcript_files = list(raw_dir.glob("*.txt")) if raw_dir.exists() else []
            print(f"\nTRANSCRIPT STATISTICS")
            print("-" * 40)
            print(f"Transcript files:     {len(transcript_files)}")
            print("-" * 40)

        return 0

    # Run the pipeline
    try:
        config = PipelineConfig(
            textbook_path=str(textbook_path) if textbook_path else None,
            output_path=args.output,
            transcript_dir=transcript_dir,
            summaries_dir=args.summaries_dir,
            use_transcripts=use_transcripts,
            combine_with_textbook=args.mode == "combined",
            target_questions_per_topic=args.target_questions,
            target_questions_per_episode=args.target_episode_questions,
            augment_questions=not args.no_augment,
            include_critical_distinctions=not args.no_critical,
            api_key=args.api_key,
            textbook_ratio=args.textbook_ratio,
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
