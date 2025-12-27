"""
Evaluation CLI - Command-line interface for running mindprint evaluations.
"""

import argparse
import sys
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from .pipeline import EvaluationPipeline
from .reporting import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Mindprint Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a DPO-trained model
  python -m src.evaluation.cli \\
    --model google/gemma-3-12b \\
    --adapter ./checkpoints/bob-loukas-dpo-v1 \\
    --quiz-data ./data/bob_loukas \\
    --approach dpo

  # Evaluate without LoRA adapter (base model)
  python -m src.evaluation.cli \\
    --model ./models/gemma-3-12b-bob-merged \\
    --quiz-data ./data/bob_loukas \\
    --approach dpo \\
    --no-adapter
""",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Base model path or HuggingFace model name",
    )
    parser.add_argument(
        "--adapter",
        help="LoRA adapter path (optional if using merged model)",
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Evaluate without loading an adapter",
    )
    parser.add_argument(
        "--quiz-data",
        required=True,
        help="Path to quiz data directory",
    )
    parser.add_argument(
        "--output",
        default="./eval_results",
        help="Output directory for reports (default: ./eval_results)",
    )
    parser.add_argument(
        "--approach",
        choices=["dpo", "ppo"],
        required=True,
        help="Training approach (for labeling)",
    )
    parser.add_argument(
        "--name",
        help="Model name for reports (default: derived from adapter/model path)",
    )
    parser.add_argument(
        "--no-early-termination",
        action="store_true",
        help="Disable early termination on critical failures",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on (default: cuda if available)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code (required for some models like Qwen)",
    )

    return parser.parse_args()


def get_torch_dtype(dtype_str: str):
    """Convert dtype string to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def main():
    """Main entry point for evaluation CLI."""
    args = parse_args()

    # Validate arguments
    if not args.no_adapter and not args.adapter:
        logger.error("Either --adapter or --no-adapter must be specified")
        sys.exit(1)

    # Determine model name
    if args.name:
        model_name = args.name
    elif args.adapter:
        model_name = Path(args.adapter).name
    else:
        model_name = Path(args.model).name

    logger.info(f"Model name: {model_name}")
    logger.info(f"Base model: {args.model}")
    logger.info(f"Adapter: {args.adapter if args.adapter else 'None'}")
    logger.info(f"Quiz data: {args.quiz_data}")
    logger.info(f"Approach: {args.approach}")
    logger.info(f"Device: {args.device}")

    # Prepare quantization config
    quantization_config = None
    if args.load_in_4bit or args.load_in_8bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=get_torch_dtype(args.dtype),
        )
        logger.info(f"Using {'4-bit' if args.load_in_4bit else '8-bit'} quantization")

    # Load model
    logger.info("Loading base model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=get_torch_dtype(args.dtype),
            device_map=args.device if args.device != "cpu" else None,
            quantization_config=quantization_config,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        sys.exit(1)

    # Load adapter if specified
    if args.adapter and not args.no_adapter:
        logger.info(f"Loading adapter: {args.adapter}")
        try:
            model = PeftModel.from_pretrained(model, args.adapter)
        except Exception as e:
            logger.error(f"Failed to load adapter: {e}")
            sys.exit(1)

    model.eval()

    # Verify quiz data exists
    quiz_data_path = Path(args.quiz_data)
    if not quiz_data_path.exists():
        logger.error(f"Quiz data directory not found: {quiz_data_path}")
        sys.exit(1)

    # Run evaluation
    logger.info("Running evaluation...")
    try:
        pipeline = EvaluationPipeline(
            model=model,
            tokenizer=tokenizer,
            quiz_data_path=str(quiz_data_path),
            early_termination=not args.no_early_termination,
        )

        report = pipeline.run_full_evaluation(
            model_name=model_name,
            approach=args.approach,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

    # Generate reports
    logger.info("Generating reports...")
    reporter = ReportGenerator(args.output)
    paths = reporter.generate_all(report)

    logger.info(f"JSON report: {paths['json']}")
    logger.info(f"Markdown report: {paths['markdown']}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Approach: {args.approach.upper()}")
    print(f"Result: {'PASSED' if report.passed else 'FAILED'}")
    print(f"Overall Accuracy: {report.overall_accuracy:.1%}")
    print(f"Voice Fidelity: {report.overall_voice_score:.2f}")
    print(f"Critical Violations: {len(report.critical_violations)}")
    print("=" * 60)
    print(f"Reports saved to: {args.output}")
    print("=" * 60 + "\n")

    # Exit with appropriate code
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
