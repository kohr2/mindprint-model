#!/usr/bin/env python3
"""
DPO Training Pipeline CLI.

Runs the full SFT + DPO training pipeline for Bob Loukas mindprint.
Supports both PyTorch (CUDA/CPU) and MLX (Apple Silicon) backends.

Usage:
    # With MLX backend (Mac Studio)
    python scripts/run_dpo_training.py --config configs/training_pipeline.yaml --backend mlx

    # With PyTorch backend (Cloud GPU)
    python scripts/run_dpo_training.py --config configs/training_pipeline.yaml --backend pytorch

    # Legacy mode (direct PyTorch)
    python scripts/run_dpo_training.py --config configs/training_pipeline.yaml --backend null

    # Resume from checkpoint
    python scripts/run_dpo_training.py --resume ./checkpoints/latest.json
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import (
    PipelineConfig,
    DPOPipeline,
)

# Try to import backends (optional)
try:
    from src.backends import create_backend, BackendProtocol
    BACKENDS_AVAILABLE = True
except ImportError:
    BACKENDS_AVAILABLE = False
    BackendProtocol = None

# Try to import PyTorch (for legacy mode)
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.training.mps_utils import get_mps_device, mps_empty_cache
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> PipelineConfig:
    """Load pipeline configuration from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Extract backend configuration
    backend_config = config_dict.get("backend", {})
    backend_type = backend_config.get("type")
    backend_device = backend_config.get("device", "auto")
    backend_dtype = backend_config.get("dtype", "float16")

    # Map YAML structure to PipelineConfig
    return PipelineConfig(
        # Backend settings (for new backend system)
        backend_type=backend_type,
        backend_device=backend_device,
        backend_dtype=backend_dtype,
        # SFT settings
        sft_epochs_per_topic=config_dict.get("sft", {}).get("epochs_per_topic", 3),
        sft_learning_rate=config_dict.get("sft", {}).get("learning_rate", 3e-4),
        sft_batch_size=config_dict.get("sft", {}).get("batch_size", 4),
        # DPO settings
        dpo_steps_per_topic=config_dict.get("dpo", {}).get("steps_per_topic", 100),
        dpo_learning_rate=config_dict.get("dpo", {}).get("learning_rate", 5e-7),
        dpo_batch_size=config_dict.get("dpo", {}).get("batch_size", 2),
        dpo_beta=config_dict.get("dpo", {}).get("beta", 0.1),
        # Thresholds
        accuracy_threshold=config_dict.get("thresholds", {}).get("accuracy_threshold", 0.70),
        dpo_trigger_threshold=config_dict.get("thresholds", {}).get("dpo_trigger_threshold", 0.75),
        topic_pass_threshold=config_dict.get("thresholds", {}).get("topic_pass_threshold", 0.90),
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
        description="Run SFT + DPO training pipeline for Bob Loukas mindprint"
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
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--backend",
        type=str,
        help="Override backend from config (pytorch, mlx, or null for legacy)",
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

    # Load configuration
    if Path(args.config).exists():
        config = load_config(args.config)
        logger.info(f"Loaded config from: {args.config}")
    else:
        config = PipelineConfig()
        logger.warning(f"Config file not found: {args.config}, using defaults")

    # Override backend if specified
    if args.backend:
        if args.backend.lower() == "null":
            config.backend_type = None
        else:
            config.backend_type = args.backend.lower()
        logger.info(f"Backend overridden to: {config.backend_type}")

    # Override paths if specified
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir

    # Determine mode
    use_backend = config.backend_type is not None and BACKENDS_AVAILABLE

    # Dry run - just print config
    if args.dry_run:
        logger.info("=== DRY RUN - Configuration ===")
        logger.info(f"Model: {args.model}")
        logger.info(f"Backend mode: {use_backend}")
        if use_backend:
            logger.info(f"Backend type: {config.backend_type}")
            logger.info(f"Backend device: {config.backend_device}")
            logger.info(f"Backend dtype: {config.backend_dtype}")
        logger.info(f"SFT epochs/topic: {config.sft_epochs_per_topic}")
        logger.info(f"DPO steps/topic: {config.dpo_steps_per_topic}")
        logger.info(f"Accuracy threshold: {config.accuracy_threshold}")
        logger.info(f"DPO trigger threshold: {config.dpo_trigger_threshold}")
        logger.info(f"Topic pass threshold: {config.topic_pass_threshold}")
        logger.info(f"Data dir: {config.data_dir}")
        logger.info(f"Output dir: {config.output_dir}")
        logger.info(f"Checkpoint dir: {config.checkpoint_dir}")
        return 0

    # Initialize pipeline based on mode
    backend = None
    model = None
    tokenizer = None

    if use_backend:
        # Backend mode
        logger.info(f"Using backend: {config.backend_type}")
        logger.info(f"Backend device: {config.backend_device}")
        logger.info(f"Backend dtype: {config.backend_dtype}")

        # Create backend
        backend = create_backend(
            config.backend_type,
            device=config.backend_device,
            dtype=config.backend_dtype,
        )
        logger.info(f"Backend created: {backend.name}")

        # Load model name from config
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
            model_name = config_dict.get("model", {}).get("name", args.model)

        logger.info(f"Loading model via backend: {model_name}")

        # Load model via backend
        model_interface = backend.load_model(model_name)
        logger.info(f"Model loaded successfully")
        logger.info(f"Model parameters: {model_interface.num_parameters:,}")

        # Create pipeline with backend
        pipeline = DPOPipeline(
            model=model_interface.get_underlying_model(),  # Pass underlying model
            tokenizer=model_interface.tokenizer,
            config=config,
            backend=backend,
        )
    else:
        # Legacy mode (direct PyTorch)
        if not PYTORCH_AVAILABLE:
            logger.error("PyTorch not available for legacy mode")
            return 1

        logger.info("Using legacy mode (direct PyTorch)")

        # Check MPS availability
        device = get_mps_device()
        logger.info(f"Using device: {device}")

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            args.model,
            device=device.type,
            dtype="float16",
        )

        # Create pipeline
        pipeline = DPOPipeline(model, tokenizer, config)

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
    logger.info("Starting training pipeline...")
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
    if use_backend and backend is not None:
        # Use backend device manager
        backend.get_device_manager().empty_cache()
        logger.info("Cleared backend device cache")
    elif PYTORCH_AVAILABLE:
        # Use MPS utils
        mps_empty_cache()
        logger.info("Cleared MPS cache")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
