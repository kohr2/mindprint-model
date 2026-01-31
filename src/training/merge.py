"""
LoRA Merging Module - Merge LoRA adapters into base models.

Handles merging trained LoRA adapters back into the base Gemma-3-12B model.
Optimized for Mac Studio M2 Ultra (64GB unified memory).
"""

import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.models.download import detect_platform

logger = logging.getLogger(__name__)


@dataclass
class MergeConfig:
    """Configuration for LoRA merging."""

    base_model_path: str
    adapter_path: str
    output_path: str
    dtype: str = "float16"
    device: str = "auto"
    verify_merge: bool = True

    def __post_init__(self):
        """Validate configuration."""
        # Validate adapter path exists
        if not Path(self.adapter_path).exists():
            raise ValueError(f"Adapter path does not exist: {self.adapter_path}")


@dataclass
class MergeResult:
    """Result of a merge operation."""

    success: bool
    output_path: str
    base_model: str
    adapter_path: str
    merge_time_seconds: float
    verification_passed: bool = True
    verification_details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class LoRAMerger:
    """Merges LoRA adapters into base models."""

    def __init__(self, config: MergeConfig):
        """
        Initialize the merger.

        Args:
            config: MergeConfig with merge parameters
        """
        self.config = config

    def merge(self) -> MergeResult:
        """
        Merge LoRA adapter into base model.

        Returns:
            MergeResult with merge outcome
        """
        start_time = time.time()

        try:
            # Detect platform for device selection
            if self.config.device == "auto":
                platform = detect_platform()
                device = platform["device"]
            else:
                device = self.config.device

            # Determine dtype
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(self.config.dtype, torch.float16)

            logger.info(f"Loading base model: {self.config.base_model_path}")
            logger.info(f"Device: {device}, dtype: {self.config.dtype}")

            # Load base model - for MPS, load to CPU first
            if device == "mps":
                device_map = None
            else:
                device_map = "auto" if device == "cuda" else None

            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            logger.info(f"Loading adapter: {self.config.adapter_path}")

            # Load LoRA adapter
            model_with_adapter = PeftModel.from_pretrained(
                base_model,
                self.config.adapter_path,
            )

            logger.info("Merging adapter into base model...")

            # Merge and unload - this creates a standalone model
            merged_model = model_with_adapter.merge_and_unload()

            # Move to MPS if needed for verification
            if device == "mps" and self.config.verify_merge:
                merged_model = merged_model.to("mps")

            # Create output directory
            output_path = Path(self.config.output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Saving merged model to: {output_path}")

            # Save merged model
            merged_model.save_pretrained(
                output_path,
                safe_serialization=True,
            )
            tokenizer.save_pretrained(output_path)

            merge_time = time.time() - start_time

            # Verify merge if requested
            verification_passed = True
            verification_details = None

            if self.config.verify_merge:
                logger.info("Verifying merge with inference test...")
                try:
                    verification_details = self._verify_merge(
                        merged_model, tokenizer, device
                    )
                    verification_passed = True
                    logger.info("Verification passed")
                except Exception as e:
                    logger.warning(f"Verification failed: {e}")
                    verification_passed = False
                    verification_details = {"error": str(e)}

            return MergeResult(
                success=True,
                output_path=str(output_path),
                base_model=self.config.base_model_path,
                adapter_path=self.config.adapter_path,
                merge_time_seconds=merge_time,
                verification_passed=verification_passed,
                verification_details=verification_details,
            )

        except Exception as e:
            merge_time = time.time() - start_time
            logger.error(f"Merge failed: {e}")
            return MergeResult(
                success=False,
                output_path="",
                base_model=self.config.base_model_path,
                adapter_path=self.config.adapter_path,
                merge_time_seconds=merge_time,
                verification_passed=False,
                error_message=str(e),
            )

    def merge_incremental(self, adapter_paths: List[str]) -> MergeResult:
        """
        Merge multiple adapters incrementally.

        This is useful for combining adapters trained on different phases
        (e.g., ORPO adapter from different units).

        Args:
            adapter_paths: List of adapter paths to merge in order

        Returns:
            MergeResult with final merge outcome
        """
        start_time = time.time()

        try:
            # Detect platform for device selection
            if self.config.device == "auto":
                platform = detect_platform()
                device = platform["device"]
            else:
                device = self.config.device

            # Determine dtype
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(self.config.dtype, torch.float16)

            logger.info(f"Incremental merge of {len(adapter_paths)} adapters")

            # Load base model
            if device == "mps":
                device_map = None
            else:
                device_map = "auto" if device == "cuda" else None

            current_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )

            tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Merge each adapter in sequence
            for i, adapter_path in enumerate(adapter_paths):
                logger.info(f"Merging adapter {i+1}/{len(adapter_paths)}: {adapter_path}")

                model_with_adapter = PeftModel.from_pretrained(
                    current_model,
                    adapter_path,
                )
                current_model = model_with_adapter.merge_and_unload()

            # Save final merged model
            output_path = Path(self.config.output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            current_model.save_pretrained(
                output_path,
                safe_serialization=True,
            )
            tokenizer.save_pretrained(output_path)

            merge_time = time.time() - start_time

            return MergeResult(
                success=True,
                output_path=str(output_path),
                base_model=self.config.base_model_path,
                adapter_path=",".join(adapter_paths),
                merge_time_seconds=merge_time,
                verification_passed=True,
            )

        except Exception as e:
            merge_time = time.time() - start_time
            logger.error(f"Incremental merge failed: {e}")
            return MergeResult(
                success=False,
                output_path="",
                base_model=self.config.base_model_path,
                adapter_path=",".join(adapter_paths),
                merge_time_seconds=merge_time,
                verification_passed=False,
                error_message=str(e),
            )

    def _verify_merge(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str,
    ) -> Dict[str, Any]:
        """
        Verify merge by running a quick inference test.

        Args:
            model: Merged model
            tokenizer: Tokenizer
            device: Device to run on

        Returns:
            Dict with verification details
        """
        test_prompt = "What drives the 4-year cycle in Bitcoin?"

        inputs = tokenizer(test_prompt, return_tensors="pt")

        if device == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        elif device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "test_prompt": test_prompt,
            "test_output": response,
            "output_length": len(outputs[0]),
        }
