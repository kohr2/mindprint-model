"""
Model Exporter - Export models to various formats.

Exports merged models as safetensors and GGUF for deployment.
"""

import time
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for model export."""

    model_path: str
    output_dir: str
    export_safetensors: bool = True
    export_gguf: bool = True
    gguf_quantization: str = "Q5_K_M"

    def __post_init__(self):
        """Validate configuration."""
        if not Path(self.model_path).exists():
            raise ValueError(f"Model path does not exist: {self.model_path}")


@dataclass
class ExportResult:
    """Result of an export operation."""

    success: bool
    exports: Dict[str, str]
    model_path: str
    export_time_seconds: float
    error_message: Optional[str] = None
    warnings: list = field(default_factory=list)


class ModelExporter:
    """Exports models to various formats."""

    def __init__(self, config: ExportConfig):
        """
        Initialize the exporter.

        Args:
            config: ExportConfig with export parameters
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(self) -> ExportResult:
        """
        Export model to all configured formats.

        Returns:
            ExportResult with export outcomes
        """
        start_time = time.time()
        exports = {}
        warnings = []

        try:
            # Export safetensors
            if self.config.export_safetensors:
                logger.info("Exporting to safetensors format...")
                safetensors_path = self.export_safetensors()
                exports["safetensors"] = safetensors_path
                logger.info(f"Safetensors exported to: {safetensors_path}")

            # Export GGUF
            if self.config.export_gguf:
                logger.info("Exporting to GGUF format...")
                try:
                    gguf_path = self.export_gguf()
                    exports["gguf"] = gguf_path
                    logger.info(f"GGUF exported to: {gguf_path}")
                except RuntimeError as e:
                    warnings.append(f"GGUF export failed: {e}")
                    logger.warning(f"GGUF export failed: {e}")

            export_time = time.time() - start_time

            return ExportResult(
                success=True,
                exports=exports,
                model_path=self.config.model_path,
                export_time_seconds=export_time,
                warnings=warnings,
            )

        except Exception as e:
            export_time = time.time() - start_time
            logger.error(f"Export failed: {e}")
            return ExportResult(
                success=False,
                exports=exports,
                model_path=self.config.model_path,
                export_time_seconds=export_time,
                error_message=str(e),
            )

    def export_safetensors(self) -> str:
        """
        Export model to safetensors format.

        Returns:
            Path to the exported model directory
        """
        output_path = self.output_dir / "safetensors"
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading model from: {self.config.model_path}")

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

        # Save with safetensors
        logger.info(f"Saving to: {output_path}")
        model.save_pretrained(
            output_path,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(output_path)

        return str(output_path)

    def export_gguf(self) -> str:
        """
        Export model to GGUF format using llama.cpp.

        Returns:
            Path to the exported GGUF file
        """
        output_path = self.output_dir / "gguf"
        output_path.mkdir(parents=True, exist_ok=True)

        # Output filename based on quantization
        gguf_file = output_path / f"model-{self.config.gguf_quantization}.gguf"

        logger.info(f"Converting to GGUF with quantization: {self.config.gguf_quantization}")

        # Call llama.cpp converter
        # The converter script path may vary by installation
        converter_paths = [
            "convert-hf-to-gguf.py",
            "llama.cpp/convert-hf-to-gguf.py",
            Path.home() / "llama.cpp" / "convert-hf-to-gguf.py",
        ]

        converter = None
        for path in converter_paths:
            if Path(path).exists():
                converter = str(path)
                break

        if converter is None:
            # Try to run as module (if llama-cpp-python installed)
            converter = "convert-hf-to-gguf.py"

        cmd = [
            "python",
            converter,
            self.config.model_path,
            "--outfile",
            str(gguf_file),
            "--outtype",
            self.config.gguf_quantization,
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"GGUF conversion failed (exit {result.returncode}): {result.stderr}"
            )

        return str(gguf_file)

    def generate_model_card(
        self,
        model_name: str,
        base_model: str,
        approach: str,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """
        Generate MODEL_CARD.md for the exported model.

        Args:
            model_name: Name of the fine-tuned model
            base_model: Base model used for fine-tuning
            approach: Training approach (dpo, sft, etc.)
            metrics: Optional evaluation metrics

        Returns:
            Path to the generated model card
        """
        card_path = self.output_dir / "MODEL_CARD.md"

        lines = [
            f"# {model_name}",
            "",
            "## Model Details",
            "",
            f"- **Base Model:** {base_model}",
            f"- **Fine-tuning Approach:** {approach.upper()}",
            f"- **Created:** {datetime.now().strftime('%Y-%m-%d')}",
            "",
            "## Description",
            "",
            "This model has been fine-tuned to capture the voice and style of ",
            "Bob Loukas, focusing on Bitcoin cycle analysis and market psychology.",
            "",
            "## Training Data",
            "",
            "- Source: Bob Loukas textbook content",
            "- Preference pairs generated via voice stripping",
            "- Critical distinctions for halving vs. cycle causation",
            "",
        ]

        if metrics:
            lines.extend([
                "## Evaluation Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
            ])
            for name, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"| {name} | {value:.3f} |")
                else:
                    lines.append(f"| {name} | {value} |")
            lines.append("")

        lines.extend([
            "## Usage",
            "",
            "```python",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            f'model = AutoModelForCausalLM.from_pretrained("{model_name}")',
            f'tokenizer = AutoTokenizer.from_pretrained("{model_name}")',
            "",
            '# Example prompt',
            'prompt = "Explain the 4-year Bitcoin cycle."',
            'inputs = tokenizer(prompt, return_tensors="pt")',
            'outputs = model.generate(**inputs, max_new_tokens=200)',
            'print(tokenizer.decode(outputs[0]))',
            "```",
            "",
            "## License",
            "",
            "This model is for research and educational purposes only.",
            "",
            "## Acknowledgments",
            "",
            "Based on the teaching and analysis methodology of Bob Loukas.",
            "",
        ])

        with open(card_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Model card saved to: {card_path}")
        return card_path
