#!/usr/bin/env python3
"""
Merge a LoRA adapter into a base model and save a standalone fine-tuned model.

Works for any dataset run (textbook, transcripts, combined) as long as you
provide either:
- a checkpoint JSON with adapter_path, or
- an explicit --adapter-path.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.backends import create_backend


logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_checkpoint(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def resolve_inputs(
    config_path: Optional[Path],
    checkpoint_path: Optional[Path],
    adapter_path: Optional[Path],
    model_name: Optional[str],
    backend: Optional[str],
    device: Optional[str],
    dtype: Optional[str],
) -> Dict[str, str]:
    cfg: Dict[str, Any] = {}
    if config_path is not None:
        cfg = load_yaml(config_path)

    backend_cfg = cfg.get("backend", {})
    model_cfg = cfg.get("model", {})

    resolved_model = model_name or model_cfg.get("name")
    resolved_backend = (backend or backend_cfg.get("type") or "mlx").lower()
    resolved_device = device or backend_cfg.get("device", "auto")
    resolved_dtype = dtype or backend_cfg.get("dtype", "float16")

    resolved_adapter: Optional[str] = str(adapter_path) if adapter_path else None
    if resolved_adapter is None and checkpoint_path is not None:
        ckpt = load_checkpoint(checkpoint_path)
        if ckpt.get("adapter_path"):
            resolved_adapter = ckpt["adapter_path"]

    if not resolved_model:
        raise ValueError("Could not resolve model name. Pass --model-name or --config with model.name.")
    if not resolved_adapter:
        raise ValueError("Could not resolve adapter path. Pass --adapter-path or --checkpoint with adapter_path.")

    return {
        "model_name": resolved_model,
        "backend": resolved_backend,
        "device": resolved_device,
        "dtype": resolved_dtype,
        "adapter_path": resolved_adapter,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge LoRA adapter into standalone model")
    p.add_argument("--config", type=Path, default=None, help="YAML config (optional)")
    p.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint JSON (optional)")
    p.add_argument("--adapter-path", type=Path, default=None, help="Adapter directory (optional)")
    p.add_argument("--model-name", type=str, default=None, help="Base model name/path override")
    p.add_argument("--backend", type=str, default=None, choices=["mlx", "pytorch"], help="Backend override")
    p.add_argument("--device", type=str, default=None, help="Backend device override")
    p.add_argument("--dtype", type=str, default=None, help="Backend dtype override")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory for merged model")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        resolved = resolve_inputs(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            adapter_path=args.adapter_path,
            model_name=args.model_name,
            backend=args.backend,
            device=args.device,
            dtype=args.dtype,
        )
    except Exception as e:
        logger.error(str(e))
        return 1

    model_name = resolved["model_name"]
    backend_name = resolved["backend"]
    adapter_path = resolved["adapter_path"]
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Backend: {backend_name} (device={resolved['device']}, dtype={resolved['dtype']})")
    logger.info(f"Base model: {model_name}")
    logger.info(f"Adapter path: {adapter_path}")
    logger.info(f"Output dir: {out_dir}")

    backend = create_backend(backend_name, device=resolved["device"], dtype=resolved["dtype"])
    model = backend.load_model(model_name, adapter_path=adapter_path)
    model = backend.get_adapter_manager().merge_adapter(model, adapter_name="orpo")
    model.save_pretrained(out_dir)

    logger.info("Merged model saved successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

