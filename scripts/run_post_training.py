#!/usr/bin/env python3
"""
Post-Training Pipeline CLI - Orchestrate merge, evaluate, and export.

Usage:
  python scripts/run_post_training.py \
    --base-model google/gemma-3-12b-it \
    --adapter ./checkpoints/bob-loukas-dpo \
    --quiz-data ./data/bob_loukas \
    --output ./output/bob-loukas-v1

Modes:
  Full pipeline (default): merge -> evaluate -> export
  Merge only: --merge-only
  Export only: --export-only --model-path /path/to/merged
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.post_training.pipeline import (
    PostTrainingConfig,
    PostTrainingPipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """CLI entry point for post-training pipeline."""
    parser = argparse.ArgumentParser(
        description="Post-training pipeline for Bob Loukas mindprint models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (merge -> evaluate -> export)
  python scripts/run_post_training.py \\
    --base-model google/gemma-3-12b-it \\
    --adapter ./checkpoints/bob-loukas-dpo \\
    --quiz-data ./data/bob_loukas \\
    --output ./output/bob-loukas-v1

  # Merge only (skip evaluation and export)
  python scripts/run_post_training.py \\
    --base-model google/gemma-3-12b-it \\
    --adapter ./checkpoints/bob-loukas-dpo \\
    --output ./output/bob-loukas-v1 \\
    --merge-only

  # Export only (from pre-merged model)
  python scripts/run_post_training.py \\
    --output ./output/bob-loukas-v1 \\
    --export-only \\
    --model-path ./output/merged

  # Skip evaluation (useful for quick iteration)
  python scripts/run_post_training.py \\
    --base-model google/gemma-3-12b-it \\
    --adapter ./checkpoints/bob-loukas-dpo \\
    --output ./output/bob-loukas-v1 \\
    --skip-evaluation
""",
    )

    # Required arguments for full pipeline
    parser.add_argument(
        "--base-model",
        help="Base model path or HuggingFace repo ID (e.g., google/gemma-3-12b-it)",
    )
    parser.add_argument(
        "--adapter",
        help="Path to trained LoRA adapter directory",
    )
    parser.add_argument(
        "--quiz-data",
        help="Path to quiz data directory for evaluation",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for merged model and exports",
    )

    # Mode selection
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only run merge phase (skip evaluation and export)",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only run export phase (requires --model-path)",
    )
    parser.add_argument(
        "--model-path",
        help="Path to pre-merged model (required for --export-only)",
    )

    # Options
    parser.add_argument(
        "--approach",
        choices=["dpo", "sft", "ppo"],
        default="dpo",
        help="Training approach used (default: dpo)",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation phase",
    )
    parser.add_argument(
        "--no-safetensors",
        action="store_true",
        help="Skip safetensors export",
    )
    parser.add_argument(
        "--no-gguf",
        action="store_true",
        help="Skip GGUF export",
    )
    parser.add_argument(
        "--gguf-quantization",
        default="Q5_K_M",
        choices=["Q4_K_M", "Q5_K_M", "Q8_0", "F16"],
        help="GGUF quantization format (default: Q5_K_M)",
    )

    args = parser.parse_args()

    # Validate argument combinations
    if args.export_only:
        if not args.model_path:
            parser.error("--export-only requires --model-path")
    elif not args.merge_only:
        if not args.base_model:
            parser.error("--base-model is required for full pipeline or merge-only")
        if not args.adapter:
            parser.error("--adapter is required for full pipeline or merge-only")
        if not args.skip_evaluation and not args.quiz_data:
            parser.error("--quiz-data is required unless using --skip-evaluation")

    # Print header
    print("\n" + "=" * 60)
    print("POST-TRAINING PIPELINE")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {args.output}")

    if args.merge_only:
        print("Mode: MERGE ONLY")
    elif args.export_only:
        print("Mode: EXPORT ONLY")
    else:
        print("Mode: FULL PIPELINE")
    print("=" * 60 + "\n")

    try:
        if args.export_only:
            # Export-only mode
            from src.export.exporter import ExportConfig, ModelExporter

            export_config = ExportConfig(
                model_path=args.model_path,
                output_dir=args.output,
                export_safetensors=not args.no_safetensors,
                export_gguf=not args.no_gguf,
                gguf_quantization=args.gguf_quantization,
            )

            exporter = ModelExporter(export_config)
            result = exporter.export_all()

            if result.success:
                print("\n" + "=" * 60)
                print("EXPORT COMPLETED SUCCESSFULLY")
                print("=" * 60)
                for fmt, path in result.exports.items():
                    print(f"  {fmt}: {path}")
                return 0
            else:
                print(f"\nExport failed: {result.error_message}")
                return 1

        else:
            # Full pipeline or merge-only
            config = PostTrainingConfig(
                base_model=args.base_model,
                adapter_path=args.adapter,
                quiz_data_path=args.quiz_data or "",
                output_dir=args.output,
                approach=args.approach,
                skip_evaluation=args.skip_evaluation or args.merge_only,
                export_safetensors=not args.no_safetensors and not args.merge_only,
                export_gguf=not args.no_gguf and not args.merge_only,
                gguf_quantization=args.gguf_quantization,
            )

            pipeline = PostTrainingPipeline(config)

            if args.merge_only:
                result = pipeline.run_merge_only()
                if result.success:
                    print("\n" + "=" * 60)
                    print("MERGE COMPLETED SUCCESSFULLY")
                    print("=" * 60)
                    print(f"  Merged model: {result.output_path}")
                    print(f"  Time: {result.merge_time_seconds:.1f}s")
                    return 0
                else:
                    print(f"\nMerge failed: {result.error_message}")
                    return 1
            else:
                result = pipeline.run()

                print("\n" + "=" * 60)
                if result.success:
                    print("PIPELINE COMPLETED SUCCESSFULLY")
                else:
                    print("PIPELINE COMPLETED WITH ISSUES")
                print("=" * 60)

                print(f"\nMerged model: {result.merged_model_path}")

                if result.evaluation_passed is not None:
                    status = "PASSED" if result.evaluation_passed else "FAILED"
                    print(f"Evaluation: {status}")
                else:
                    print("Evaluation: SKIPPED")

                if result.exports:
                    print("\nExports:")
                    for fmt, path in result.exports.items():
                        print(f"  {fmt}: {path}")

                if result.warnings:
                    print("\nWarnings:")
                    for warning in result.warnings:
                        print(f"  - {warning}")

                print(f"\nTotal time: {result.total_time_seconds:.1f}s")

                return 0 if result.success else 1

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
