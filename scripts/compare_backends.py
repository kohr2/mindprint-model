#!/usr/bin/env python3
"""
Backend Performance Comparison Script.

Compares PyTorch and MLX backends on the same training task.

Usage:
    python scripts/compare_backends.py --model Qwen/Qwen2.5-7B-Instruct
    python scripts/compare_backends.py --model Qwen/Qwen2.5-7B-Instruct --backends pytorch mlx
    python scripts/compare_backends.py --help
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_test_data() -> tuple[List[Dict], List[Dict]]:
    """Create minimal test data for comparison."""
    sft_data = [
        {
            "question": "What is a bull market?",
            "answer": "A bull market is characterized by rising prices and investor optimism, typically lasting 18-24 months in Bitcoin's 4-year cycle.",
        },
        {
            "question": "What is a bear market?",
            "answer": "A bear market is characterized by falling prices and investor pessimism, typically following a cycle top and lasting 12-18 months.",
        },
        {
            "question": "What is the Bitcoin halving?",
            "answer": "The Bitcoin halving is an event that reduces the block reward by 50%, occurring approximately every 4 years and reducing supply inflation.",
        },
    ]

    dpo_data = [
        {
            "prompt": "Explain Bitcoin cycles:",
            "chosen": "Bitcoin follows 4-year cycles driven by the halving event, which reduces supply inflation and typically triggers bull markets.",
            "rejected": "Bitcoin price goes up and down randomly over time.",
        },
    ]

    return sft_data, dpo_data


def train_with_backend(
    backend_type: str,
    model_name: str,
    sft_data: List[Dict],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Train with a specific backend and return results.

    Args:
        backend_type: "pytorch" or "mlx"
        model_name: HuggingFace model name
        sft_data: Training data
        output_dir: Output directory for this backend

    Returns:
        Dictionary with timing and results
    """
    from src.backends import create_backend

    logger.info(f"\n{'='*60}")
    logger.info(f"Training with {backend_type.upper()} backend")
    logger.info('='*60)

    results = {
        "backend": backend_type,
        "model": model_name,
        "success": False,
    }

    try:
        # Create backend
        device = "cuda" if backend_type == "pytorch" else "auto"
        backend = create_backend(backend_type, device=device, dtype="float16")
        logger.info(f"✓ Created {backend_type} backend (device={device})")

        # Load model
        logger.info(f"Loading model: {model_name}")
        load_start = time.time()
        model = backend.load_model(model_name)
        load_time = time.time() - load_start
        logger.info(f"✓ Model loaded in {load_time:.2f}s")

        results["model_load_time"] = load_time
        results["num_parameters"] = model.num_parameters
        results["num_trainable_parameters"] = model.num_trainable_parameters

        # Create SFT trainer
        sft_config = {
            "learning_rate": 3e-4,
            "num_epochs": 1,  # Single epoch for comparison
            "per_device_batch_size": 2,
            "max_seq_length": 512,
            "lora_r": 8,
            "lora_alpha": 16,
            "output_dir": str(output_dir),
        }

        logger.info("Creating SFT trainer...")
        trainer = backend.create_sft_trainer(model, sft_config)
        logger.info("✓ SFT trainer created")

        # Train
        logger.info(f"Training on {len(sft_data)} samples...")
        train_start = time.time()
        train_result = trainer.train(sft_data)
        train_time = time.time() - train_start

        results["training_time"] = train_time
        results["final_loss"] = train_result.final_loss
        results["samples_trained"] = train_result.samples_trained
        results["success"] = train_result.success

        logger.info(f"✓ Training complete in {train_time:.2f}s")
        logger.info(f"  Final loss: {train_result.final_loss:.4f}")
        logger.info(f"  Samples/sec: {len(sft_data) / train_time:.2f}")

        # Save adapter
        adapter_path = output_dir / "adapter"
        trainer.save_adapter(adapter_path)
        results["adapter_path"] = str(adapter_path)

        # Clean up
        backend.get_device_manager().empty_cache()

        return results

    except Exception as e:
        logger.error(f"✗ {backend_type} backend failed: {e}")
        results["error"] = str(e)
        return results


def compare_results(results: Dict[str, Dict[str, Any]]) -> None:
    """Print comparison of results."""
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    if len(results) < 2:
        print("Need at least 2 backends to compare")
        return

    backends = list(results.keys())

    # Model loading time
    print("\n📊 Model Loading Time:")
    for backend in backends:
        if "model_load_time" in results[backend]:
            print(f"  {backend:10s}: {results[backend]['model_load_time']:6.2f}s")

    # Training time
    print("\n📊 Training Time (1 epoch, 3 samples):")
    for backend in backends:
        if "training_time" in results[backend]:
            time_val = results[backend]['training_time']
            samples_per_sec = results[backend].get('samples_trained', 0) / time_val
            print(f"  {backend:10s}: {time_val:6.2f}s  ({samples_per_sec:.2f} samples/sec)")

    # Final loss
    print("\n📊 Final Loss:")
    losses = {}
    for backend in backends:
        if "final_loss" in results[backend]:
            loss = results[backend]['final_loss']
            losses[backend] = loss
            print(f"  {backend:10s}: {loss:.4f}")

    # Loss difference
    if len(losses) == 2:
        loss_diff = abs(losses[backends[0]] - losses[backends[1]])
        loss_pct = (loss_diff / losses[backends[0]]) * 100
        print(f"\n  Loss difference: {loss_diff:.4f} ({loss_pct:.1f}%)")

        if loss_pct < 5:
            print("  ✅ Results are equivalent (< 5% difference)")
        elif loss_pct < 10:
            print("  ⚠️  Results differ slightly (5-10% difference)")
        else:
            print("  ❌ Results differ significantly (> 10% difference)")

    # Model parameters
    print("\n📊 Model Size:")
    for backend in backends:
        if "num_parameters" in results[backend]:
            total = results[backend]['num_parameters']
            trainable = results[backend]['num_trainable_parameters']
            pct = (trainable / total) * 100
            print(f"  {backend:10s}: {total:,} total, {trainable:,} trainable ({pct:.2f}%)")

    # Success status
    print("\n📊 Status:")
    for backend in backends:
        status = "✅ Success" if results[backend].get("success") else "❌ Failed"
        print(f"  {backend:10s}: {status}")
        if "error" in results[backend]:
            print(f"               Error: {results[backend]['error']}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and MLX backend performance"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["pytorch", "mlx"],
        choices=["pytorch", "mlx"],
        help="Backends to compare (default: pytorch mlx)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./backend_comparison"),
        help="Output directory (default: ./backend_comparison)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, only test initialization",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create test data
    sft_data, dpo_data = create_test_data()

    # Train with each backend
    results = {}
    for backend_type in args.backends:
        backend_output = args.output_dir / backend_type
        backend_output.mkdir(exist_ok=True)

        if args.skip_training:
            logger.info(f"Skipping training for {backend_type} (--skip-training)")
            results[backend_type] = {"backend": backend_type, "skipped": True}
        else:
            results[backend_type] = train_with_backend(
                backend_type,
                args.model,
                sft_data,
                backend_output,
            )

    # Save results
    results_file = args.output_dir / "comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✓ Results saved to {results_file}")

    # Print comparison
    if not args.skip_training:
        compare_results(results)


if __name__ == "__main__":
    main()
