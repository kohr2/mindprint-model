"""
Model download utility for Bob Loukas mindprint training.

Downloads and sets up Gemma-3-12B or Qwen2.5-7B from HuggingFace Hub.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import get_model_config, load_model_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def detect_platform() -> dict:
    """Detect the current platform and available resources."""
    platform_info = {
        "device": "cpu",
        "has_cuda": torch.cuda.is_available(),
        "has_mps": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
        "gpu_memory_gb": 0,
        "recommended_dtype": "float32",
    }

    if platform_info["has_cuda"]:
        platform_info["device"] = "cuda"
        # Get GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        platform_info["gpu_memory_gb"] = round(gpu_memory, 1)
        platform_info["recommended_dtype"] = "bfloat16" if gpu_memory >= 24 else "float16"
    elif platform_info["has_mps"]:
        platform_info["device"] = "mps"
        # Mac unified memory - estimate based on system
        import subprocess

        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
            )
            total_memory_gb = int(result.stdout.strip()) / (1024**3)
            platform_info["gpu_memory_gb"] = round(total_memory_gb, 1)
            platform_info["recommended_dtype"] = "float16"  # MPS works well with fp16
        except Exception:
            platform_info["gpu_memory_gb"] = 64  # Assume Mac Studio

    return platform_info


def download_model(
    model_name: str,
    cache_dir: Optional[str] = None,
    revision: str = "main",
    verify: bool = True,
) -> Path:
    """
    Download a model from HuggingFace Hub.

    Args:
        model_name: "gemma" or "qwen" (or full HF path)
        cache_dir: Local cache directory (default: HF cache)
        revision: Git revision to download
        verify: Whether to verify the download

    Returns:
        Path to the downloaded model
    """
    # Get model config
    try:
        config = get_model_config(model_name)
        hf_path = config.hf_path
    except KeyError:
        # Assume it's a direct HF path
        hf_path = model_name

    logger.info(f"Downloading model: {hf_path}")
    logger.info(f"Cache directory: {cache_dir or 'default HF cache'}")

    # Download using snapshot_download for better reliability
    try:
        local_path = snapshot_download(
            repo_id=hf_path,
            cache_dir=cache_dir,
            revision=revision,
            resume_download=True,
        )
        logger.info(f"Downloaded to: {local_path}")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise

    # Verify the download
    if verify:
        logger.info("Verifying download...")
        try:
            # Try to load tokenizer (fast, verifies file integrity)
            AutoTokenizer.from_pretrained(local_path)
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            raise

    return Path(local_path)


def load_model_for_training(
    model_path: str,
    dtype: str = "auto",
    device: str = "auto",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    trust_remote_code: bool = False,
):
    """
    Load a model configured for training.

    Args:
        model_path: Path to model or HF repo ID
        dtype: Data type ("auto", "float16", "bfloat16", "float32")
        device: Device to load to ("auto", "cuda", "mps", "cpu")
        load_in_4bit: Use 4-bit quantization
        load_in_8bit: Use 8-bit quantization
        trust_remote_code: Trust remote code (for Qwen)

    Returns:
        Tuple of (model, tokenizer)
    """
    platform = detect_platform()

    # Determine device
    if device == "auto":
        device = platform["device"]

    # Determine dtype
    if dtype == "auto":
        dtype = platform["recommended_dtype"]

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    logger.info(f"Loading model with dtype={dtype}, device={device}")

    # Prepare quantization config
    quantization_config = None
    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        logger.info(f"Using {'4-bit' if load_in_4bit else '8-bit'} quantization")

    # Handle device_map for different platforms
    if device == "mps":
        # MPS doesn't support device_map, load to CPU then move
        device_map = None
        logger.info("MPS detected - loading to CPU first, then moving to MPS")
    elif device == "cuda":
        device_map = "auto"
    else:
        device_map = None

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quantization_config,
        trust_remote_code=trust_remote_code,
    )

    # Move to MPS if needed
    if device == "mps" and device_map is None:
        model = model.to("mps")
        logger.info("Model moved to MPS")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def main():
    """CLI entry point for model download."""
    parser = argparse.ArgumentParser(
        description="Download models for Bob Loukas mindprint training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Gemma-3-12B (primary recommendation)
  python -m src.models.download --model gemma

  # Download Qwen2.5-7B (alternative)
  python -m src.models.download --model qwen

  # Download to specific directory
  python -m src.models.download --model gemma --cache-dir ./models

  # Download and verify
  python -m src.models.download --model gemma --verify
""",
    )

    parser.add_argument(
        "--model",
        choices=["gemma", "qwen", "gemma-3-12b", "qwen2.5-7b"],
        default="gemma",
        help="Model to download (default: gemma)",
    )
    parser.add_argument(
        "--cache-dir",
        help="Local cache directory (default: HuggingFace cache)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Git revision to download (default: main)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify download by loading tokenizer",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show platform info and exit",
    )

    args = parser.parse_args()

    # Show platform info
    platform = detect_platform()
    print("\n" + "=" * 50)
    print("PLATFORM INFORMATION")
    print("=" * 50)
    print(f"Device: {platform['device'].upper()}")
    print(f"CUDA available: {platform['has_cuda']}")
    print(f"MPS available: {platform['has_mps']}")
    print(f"GPU/Unified memory: {platform['gpu_memory_gb']} GB")
    print(f"Recommended dtype: {platform['recommended_dtype']}")
    print("=" * 50 + "\n")

    if args.info:
        # Recommend model based on memory
        if platform["gpu_memory_gb"] >= 24:
            print("Recommendation: Use Gemma-3-12B (fp16)")
        elif platform["gpu_memory_gb"] >= 16:
            print("Recommendation: Use Gemma-3-12B (4-bit) or Qwen2.5-7B (fp16)")
        else:
            print("Recommendation: Use Qwen2.5-7B (4-bit)")
        return 0

    # Download model
    try:
        path = download_model(
            model_name=args.model,
            cache_dir=args.cache_dir,
            revision=args.revision,
            verify=args.verify,
        )
        print(f"\nModel downloaded to: {path}")
        print("\nNext steps:")
        print("1. Run data preparation: python scripts/run_data_prep.py")
        print("2. Start training (see DPO/PPO branch for training scripts)")
        return 0
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
