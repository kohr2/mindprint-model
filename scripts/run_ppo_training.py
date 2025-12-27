#!/usr/bin/env python3
"""
PPO Training Pipeline CLI.

Runs the full SFT + Reward Model + PPO training pipeline for Bob Loukas mindprint.
Optimized for Mac Studio M2 Ultra (64GB unified memory, MPS).

Usage:
    python scripts/run_ppo_training.py --config configs/training_pipeline.yaml
    python scripts/run_ppo_training.py --resume ./checkpoints/latest.json
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import (
    PipelineConfig,
    PPOPipeline,
)
from src.training.mps_utils import get_mps_device, mps_empty_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> PipelineConfig:
    """Load pipeline configuration from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Map YAML structure to PipelineConfig
    return PipelineConfig(
        # SFT settings
        sft_epochs_per_topic=config_dict.get("sft", {}).get("epochs_per_topic", 3),
        sft_learning_rate=config_dict.get("sft", {}).get("learning_rate", 3e-4),
        sft_batch_size=config_dict.get("sft", {}).get("batch_size", 4),
        # Reward model settings
        reward_model_epochs=config_dict.get("reward_model", {}).get("epochs", 1),
        reward_learning_rate=config_dict.get("reward_model", {}).get("learning_rate", 1e-5),
        reward_batch_size=config_dict.get("reward_model", {}).get("batch_size", 4),
        # PPO settings
        ppo_steps_per_topic=config_dict.get("ppo", {}).get("steps_per_topic", 100),
        ppo_learning_rate=config_dict.get("ppo", {}).get("learning_rate", 1e-5),
        ppo_batch_size=config_dict.get("ppo", {}).get("batch_size", 4),
        ppo_kl_penalty=config_dict.get("ppo", {}).get("kl_penalty", 0.2),
        # Thresholds
        topic_pass_threshold=config_dict.get("thresholds", {}).get("topic_pass_threshold", 0.85),
        # Pipeline control
        merge_after_unit=config_dict.get("pipeline", {}).get("merge_after_unit", True),
        max_retries_per_topic=config_dict.get("pipeline", {}).get("max_retries_per_topic", 2),
        # Paths
        data_dir=config_dict.get("paths", {}).get("data_dir", "./data"),
        output_dir=config_dict.get("paths", {}).get("output_dir", "./output"),
        checkpoint_dir=config_dict.get("paths", {}).get("checkpoint_dir", "./checkpoints"),
    )


def load_model_and_tokenizer(
    model_name: str,
    device: str = "mps",
    dtype: str = "float16",
):
    """
    Load the base model and tokenizer.

    Args:
        model_name: Model identifier (e.g., google/gemma-3-12b)
        device: Target device (mps, cuda, cpu)
        dtype: Data type (float16, bfloat16, float32)

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Device: {device}, dtype: {dtype}")

    # Determine dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    # Load model - for MPS, load without device_map
    if device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        model = model.to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Model loaded: {model.config.model_type}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, tokenizer


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run SFT + Reward Model + PPO training pipeline for Bob Loukas mindprint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_pipeline.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-12b",
        help="Model name or path",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Override data directory from config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without training",
    )

    args = parser.parse_args()

    # Check MPS availability
    device = get_mps_device()
    logger.info(f"Using device: {device}")

    # Load configuration
    if Path(args.config).exists():
        config = load_config(args.config)
        logger.info(f"Loaded config from: {args.config}")
    else:
        config = PipelineConfig()
        logger.warning(f"Config file not found: {args.config}, using defaults")

    # Override paths if specified
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir

    # Dry run - just print config
    if args.dry_run:
        logger.info("=== DRY RUN - Configuration ===")
        logger.info(f"Model: {args.model}")
        logger.info(f"Device: {device}")
        logger.info(f"SFT epochs/topic: {config.sft_epochs_per_topic}")
        logger.info(f"Reward model epochs: {config.reward_model_epochs}")
        logger.info(f"PPO steps/topic: {config.ppo_steps_per_topic}")
        logger.info(f"PPO KL penalty: {config.ppo_kl_penalty}")
        logger.info(f"Topic pass threshold: {config.topic_pass_threshold}")
        logger.info(f"Data dir: {config.data_dir}")
        logger.info(f"Output dir: {config.output_dir}")
        logger.info(f"Checkpoint dir: {config.checkpoint_dir}")
        return 0

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device=device.type,
        dtype="float16",
    )

    # Create pipeline
    pipeline = PPOPipeline(model, tokenizer, config)

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            progress = pipeline.resume_from_checkpoint(checkpoint_path)
            logger.info(f"Resumed from checkpoint: {args.resume}")
        else:
            logger.error(f"Checkpoint not found: {args.resume}")
            return 1

    # Run training
    logger.info("Starting PPO training pipeline...")
    logger.info("Phase 1: SFT training")
    logger.info("Phase 2: Reward model training")
    logger.info("Phase 3: PPO training")
    result = pipeline.train_curriculum()

    # Report results
    logger.info("=== Training Complete ===")
    logger.info(f"Success: {result.success}")
    logger.info(f"Total topics: {result.total_topics}")
    logger.info(f"Passed topics: {result.passed_topics}")
    logger.info(f"Failed topics: {len(result.failed_topics)}")
    logger.info(f"Training time: {result.total_training_time_hours:.2f} hours")

    if result.failed_topics:
        logger.warning(f"Failed topics: {result.failed_topics}")

    # Save final checkpoint
    pipeline.save_checkpoint({
        "result": result.to_dict(),
        "status": "complete",
    })

    # Clear cache
    mps_empty_cache()

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
