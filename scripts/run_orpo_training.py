#!/usr/bin/env python3
"""
ORPO Training Pipeline CLI.

Runs the ORPO training pipeline for Bob Loukas mindprint.
Supports both PyTorch (CUDA/CPU) and MLX (Apple Silicon) backends.

Usage:
    # With MLX backend (Mac Studio)
    python scripts/run_orpo_training.py --config configs/training_pipeline.yaml --backend mlx

    # With PyTorch backend (Cloud GPU)
    python scripts/run_orpo_training.py --config configs/training_pipeline.yaml --backend pytorch

    # Resume from checkpoint
    python scripts/run_orpo_training.py --resume ./checkpoints/latest.json
"""

import argparse
import json
import logging
import sys
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Any, Dict

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import (
    PipelineConfig,
    ORPOPipeline,
)
from src.infrastructure import (
    set_seed,
    hash_config,
    get_reproducibility_info,
)

# Try to import backends (optional)
try:
    from src.backends import create_backend, BackendProtocol, AdapterConfig
    # Import backend implementations to trigger registration
    try:
        import src.backends.pytorch  # noqa
    except ImportError as e:
        print(f"Warning: PyTorch backend import failed: {e}")
    try:
        import src.backends.mlx  # noqa
    except ImportError as e:
        print(f"Warning: MLX backend import failed: {e}")
    BACKENDS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Backend framework not available: {e}")
    BACKENDS_AVAILABLE = False
    BackendProtocol = None
    AdapterConfig = None

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


def extract_dataset_name(config_path: str) -> str:
    """
    Extract dataset name from config filename.
    
    Examples:
        "bob_loukas_textbook.yaml" -> "textbook"
        "bob_loukas_transcripts.yaml" -> "transcripts"
        "bob_loukas_combined.yaml" -> "combined"
        "training_pipeline.yaml" -> "default"
    """
    config_stem = Path(config_path).stem
    
    # Remove common prefix
    if "bob_loukas_" in config_stem:
        dataset_name = config_stem.replace("bob_loukas_", "")
    elif "training_pipeline" in config_stem:
        dataset_name = "default"
    else:
        # Try to extract from filename pattern
        dataset_name = config_stem
    
    return dataset_name


def load_config(config_path: str) -> Tuple[PipelineConfig, str]:
    """
    Load pipeline configuration from YAML file.
    
    Returns:
        Tuple of (PipelineConfig, dataset_name)
    """
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Extract dataset name from config filename
    dataset_name = extract_dataset_name(config_path)

    # Extract backend configuration
    backend_config = config_dict.get("backend", {})
    backend_type = backend_config.get("type")
    backend_device = backend_config.get("device", "auto")
    backend_dtype = backend_config.get("dtype", "float16")

    # Map YAML structure to PipelineConfig
    orpo_config = config_dict.get("orpo", {})

    # Early stopping settings
    pipeline_config = config_dict.get("pipeline", {})
    
    config = PipelineConfig(
        # Backend settings
        backend_type=backend_type,
        backend_device=backend_device,
        backend_dtype=backend_dtype,
        # ORPO settings
        orpo_steps_per_topic=orpo_config.get("steps_per_topic", 100),
        orpo_learning_rate=orpo_config.get("learning_rate", 3e-4),
        orpo_batch_size=orpo_config.get("batch_size", 4),
        orpo_max_length=orpo_config.get("max_length", 512),
        orpo_lambda=orpo_config.get("lambda_orpo", 0.1),
        orpo_lora_rank=orpo_config.get("lora_rank", 8),
        orpo_lora_alpha=orpo_config.get("lora_alpha", 16),
        orpo_lora_dropout=orpo_config.get("lora_dropout", 0.05),
        orpo_target_modules=orpo_config.get(
            "target_modules",
            ["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        ),
        # Thresholds
        accuracy_threshold=config_dict.get("thresholds", {}).get("accuracy_threshold", 0.70),
        topic_pass_threshold=config_dict.get("thresholds", {}).get("topic_pass_threshold", 0.90),
        # Pipeline control
        merge_after_unit=pipeline_config.get("merge_after_unit", True),
        max_retries_per_topic=pipeline_config.get("max_retries_per_topic", 2),
        # Early stopping configuration
        max_topics=pipeline_config.get("max_topics"),
        early_stopping_enabled=pipeline_config.get("early_stopping_enabled", False),
        early_stopping_patience=pipeline_config.get("early_stopping_patience", 3),
        early_stopping_cv_threshold=pipeline_config.get("early_stopping_cv_threshold", 15.0),
        early_stopping_min_topics=pipeline_config.get("early_stopping_min_topics", 10),
        # Deterministic split + gating
        split_seed=pipeline_config.get("split_seed", 42),
        holdout_ratio=pipeline_config.get("holdout_ratio", 0.2),
        holdout_min_examples_per_topic=pipeline_config.get(
            "holdout_min_examples_per_topic", 1
        ),
        require_holdout_for_gate=pipeline_config.get("require_holdout_for_gate", True),
        # Paths
        data_dir=config_dict.get("paths", {}).get("data_dir", "./data"),
        output_dir=config_dict.get("paths", {}).get("output_dir", "./output"),
        checkpoint_dir=config_dict.get("paths", {}).get("checkpoint_dir", "./checkpoints"),
        # Store config filename for checkpoint naming
        config_filename=config_path,
    )
    
    return config, dataset_name


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


def validate_training_dataset(data_dir: str) -> Tuple[int, int]:
    """
    Fail-fast validation for ORPO preference data.

    Returns:
        Tuple of (total_records, valid_records)
    """
    pref_path = Path(data_dir) / "preference_data.jsonl"
    if not pref_path.exists():
        raise FileNotFoundError(f"Missing training file: {pref_path}")

    total_records = 0
    valid_records = 0
    with open(pref_path) as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {pref_path} at line {line_num}: {exc.msg}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Invalid record type in {pref_path} at line {line_num}: "
                    f"expected object, got {type(record).__name__}"
                )

            total_records += 1
            if record.get("prompt") and record.get("chosen") and record.get("rejected"):
                valid_records += 1

    if total_records == 0:
        raise ValueError(f"Training file is empty: {pref_path}")

    if valid_records == 0:
        raise ValueError(
            "Training file has no valid preference pairs with prompt/chosen/rejected: "
            f"{pref_path}"
        )

    return total_records, valid_records


def hash_file_sha256(path: Path) -> str:
    """Compute SHA256 for a file path."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def load_pipeline_stats(data_dir: str) -> Optional[Dict[str, Any]]:
    """Load optional data-prep stats if available."""
    stats_path = Path(data_dir) / "pipeline_stats.json"
    if not stats_path.exists():
        return None
    try:
        with open(stats_path) as f:
            stats = json.load(f)
        if isinstance(stats, dict):
            return stats
    except Exception as exc:
        logger.warning(f"Could not parse pipeline stats at {stats_path}: {exc}")
    return None


def write_run_manifest(
    *,
    pipeline: ORPOPipeline,
    config: PipelineConfig,
    config_path: str,
    model_name: str,
    total_pairs: int,
    valid_pairs: int,
    result: Any,
    gate_result: Dict[str, Any],
) -> Path:
    """
    Persist full run lineage and gating metadata for reproducibility.

    Manifest includes config/data hashes, split assignment manifest,
    environment metadata, and promotion gate verdict.
    """
    data_path = Path(config.data_dir) / "preference_data.jsonl"
    data_hash = hash_file_sha256(data_path) if data_path.exists() else "missing"
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run": {
            "config_path": config_path,
            "model_name": model_name,
            "backend_type": config.backend_type,
            "backend_device": config.backend_device,
            "backend_dtype": config.backend_dtype,
        },
        "reproducibility": {
            "seed": config.split_seed,
            "config_hash": hash_config(config),
            "git_sha": get_git_sha(),
            "environment": get_reproducibility_info(),
        },
        "data": {
            "data_dir": config.data_dir,
            "preference_data_path": str(data_path),
            "preference_data_sha256": data_hash,
            "total_pairs": total_pairs,
            "valid_pairs": valid_pairs,
            "split_manifest": pipeline.get_split_manifest(),
            "pipeline_stats": load_pipeline_stats(config.data_dir),
        },
        "results": {
            "training": result.to_dict(),
            "promotion_gate": gate_result,
        },
    }

    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run ORPO training pipeline for Bob Loukas mindprint"
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
        config, dataset_name = load_config(args.config)
        logger.info(f"Loaded config from: {args.config}")
        logger.info(f"Dataset: {dataset_name}")
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

    set_seed(config.split_seed)
    logger.info(f"Deterministic seed set to {config.split_seed}")

    # Determine mode
    # ORPO training requires backend mode - enforce it
    use_backend = config.backend_type is not None and BACKENDS_AVAILABLE

    # If ORPO is configured but backends not available, that's an error
    if not use_backend and config.backend_type is not None:
        logger.error("ORPO training requires backend support, but backends are not available")
        logger.error("Please ensure MLX or PyTorch backend dependencies are installed")
        return 1

    # Dry run - just print config
    if args.dry_run:
        logger.info("=== DRY RUN - Configuration ===")
        logger.info(f"Model: {args.model}")
        logger.info(f"Backend mode: {use_backend}")
        if use_backend:
            logger.info(f"Backend type: {config.backend_type}")
            logger.info(f"Backend device: {config.backend_device}")
            logger.info(f"Backend dtype: {config.backend_dtype}")
        logger.info(f"ORPO steps/topic: {config.orpo_steps_per_topic}")
        logger.info(f"ORPO learning rate: {config.orpo_learning_rate}")
        logger.info(f"ORPO batch size: {config.orpo_batch_size}")
        logger.info(f"ORPO lambda: {config.orpo_lambda}")
        logger.info(f"ORPO LoRA rank: {config.orpo_lora_rank}")
        logger.info(f"ORPO LoRA alpha: {config.orpo_lora_alpha}")
        logger.info(f"ORPO target modules: {config.orpo_target_modules}")
        logger.info(f"Accuracy threshold: {config.accuracy_threshold}")
        logger.info(f"Topic pass threshold: {config.topic_pass_threshold}")
        logger.info(f"Split seed: {config.split_seed}")
        logger.info(f"Holdout ratio: {config.holdout_ratio}")
        logger.info(
            "Holdout min examples/topic: "
            f"{config.holdout_min_examples_per_topic}"
        )
        logger.info(f"Require holdout for gate: {config.require_holdout_for_gate}")
        logger.info(f"Data dir: {config.data_dir}")
        logger.info(f"Output dir: {config.output_dir}")
        logger.info(f"Checkpoint dir: {config.checkpoint_dir}")
        return 0

    # Fail fast on empty/malformed training data before loading a large model
    try:
        total_pairs, valid_pairs = validate_training_dataset(config.data_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"Training preflight failed: {exc}")
        return 1
    logger.info(
        f"Training dataset preflight passed: {valid_pairs}/{total_pairs} valid preference pairs"
    )

    # Load resume progress first so we can load model with adapter if resuming
    resume_progress = None
    if args.resume:
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {args.resume}")
            return 1
        with open(checkpoint_path) as f:
            resume_progress = json.load(f)
        logger.info(f"Will resume from checkpoint: {args.resume}")

    # Initialize pipeline based on mode
    backend = None
    model = None
    tokenizer = None
    model_name = args.model

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
            seed=config.split_seed,
        )
        logger.info(f"Backend created: {backend.name}")

        # Load model name from config
        if Path(args.config).exists():
            with open(args.config) as f:
                config_dict = yaml.safe_load(f)
                model_name = config_dict.get("model", {}).get("name", args.model)

        logger.info(f"Loading model via backend: {model_name}")

        # Load adapter from checkpoint if resuming
        adapter_path = None
        if resume_progress is not None:
            adapter_path = resume_progress.get("adapter_path")
            if adapter_path:
                logger.info(f"Loading adapter from checkpoint: {adapter_path}")

        # Load model via backend (with optional adapter for resume)
        model_interface = backend.load_model(
            model_name,
            adapter_path=adapter_path,
        )
        if not model_interface.has_adapter():
            adapter_cfg = AdapterConfig(
                r=config.orpo_lora_rank,
                alpha=config.orpo_lora_alpha,
                dropout=config.orpo_lora_dropout,
                target_modules=config.orpo_target_modules,
            )
            model_interface = backend.get_adapter_manager().add_adapter(
                model_interface,
                adapter_cfg,
                adapter_name="orpo",
            )
            logger.info(
                "Attached ORPO LoRA adapter: "
                f"rank={config.orpo_lora_rank}, alpha={config.orpo_lora_alpha}, "
                f"targets={config.orpo_target_modules}"
            )
        else:
            logger.info("Using pre-loaded adapter from resume checkpoint")
        logger.info(f"Model loaded successfully")
        logger.info(f"Model parameters: {model_interface.num_parameters:,}")

        # Create pipeline with backend
        pipeline = ORPOPipeline(
            model=model_interface,  # Pass ModelInterface directly
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
        pipeline = ORPOPipeline(model, tokenizer, config)

    # Run training (pass resume progress so completed topics are skipped)
    logger.info("Starting training pipeline...")
    result = pipeline.train_curriculum(
        initial_progress=resume_progress if resume_progress else None,
    )

    # Report results
    logger.info("=== Training Complete ===")
    logger.info(f"Success: {result.success}")
    logger.info(f"Total topics: {result.total_topics}")
    logger.info(f"Passed topics: {result.passed_topics}")
    logger.info(f"Failed topics: {len(result.failed_topics)}")
    logger.info(f"Training time: {result.total_training_time_hours:.2f} hours")

    if result.failed_topics:
        logger.warning(f"Failed topics: {result.failed_topics}")

    gate_result = pipeline.build_promotion_gate(result)
    logger.info(
        "Promotion gate verdict: "
        f"{'PASS' if gate_result.get('passed') else 'FAIL'}"
    )
    if gate_result.get("failed_conditions"):
        logger.warning(
            "Promotion gate failed conditions: "
            f"{gate_result['failed_conditions']}"
        )

    # Save final checkpoint
    checkpoint_path = pipeline.save_checkpoint({
        "result": result.to_dict(),
        "status": "complete",
        "promotion_gate": gate_result,
        "split_summary": pipeline.get_split_manifest().get("summary", {}),
    })
    logger.info(f"Saved final checkpoint: {checkpoint_path}")

    try:
        manifest_path = write_run_manifest(
            pipeline=pipeline,
            config=config,
            config_path=args.config,
            model_name=model_name,
            total_pairs=total_pairs,
            valid_pairs=valid_pairs,
            result=result,
            gate_result=gate_result,
        )
        logger.info(f"Wrote run manifest: {manifest_path}")
    except Exception as exc:
        logger.warning(f"Failed to write run manifest: {exc}")

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
