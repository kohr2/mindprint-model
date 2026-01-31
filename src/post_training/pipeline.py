"""
Post-Training Pipeline - Orchestrate merge, evaluate, and export.

Coordinates the post-training workflow:
1. Merge LoRA adapter into base model
2. Evaluate merged model with voice fidelity checks
3. Export to safetensors and GGUF formats
"""

import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.training.merge import MergeConfig, MergeResult, LoRAMerger
from src.export.exporter import ExportConfig, ExportResult, ModelExporter
from src.evaluation.pipeline import EvaluationPipeline
from src.evaluation.reporting import ReportGenerator

logger = logging.getLogger(__name__)


@dataclass
class PostTrainingConfig:
    """Configuration for the post-training pipeline."""

    base_model: str
    adapter_path: str
    quiz_data_path: str
    output_dir: str
    approach: str = "orpo"
    skip_evaluation: bool = False
    export_safetensors: bool = True
    export_gguf: bool = True
    gguf_quantization: str = "Q5_K_M"

    def __post_init__(self):
        """Validate configuration."""
        if not Path(self.adapter_path).exists():
            raise ValueError(f"Adapter path does not exist: {self.adapter_path}")


@dataclass
class PostTrainingResult:
    """Result of the post-training pipeline."""

    success: bool
    merged_model_path: str
    exports: Dict[str, str]
    evaluation_passed: Optional[bool]
    total_time_seconds: float
    error_message: Optional[str] = None
    evaluation_report_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class PostTrainingPipeline:
    """Orchestrates the post-training workflow."""

    def __init__(self, config: PostTrainingConfig):
        """
        Initialize the pipeline.

        Args:
            config: PostTrainingConfig with pipeline parameters
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> PostTrainingResult:
        """
        Run the full post-training pipeline.

        Executes: Merge -> Evaluate -> Export

        Returns:
            PostTrainingResult with pipeline outcome
        """
        start_time = time.time()
        warnings = []

        # Phase 1: Merge
        logger.info("=" * 60)
        logger.info("PHASE 1: Merging LoRA adapter into base model")
        logger.info("=" * 60)

        merge_result = self._run_merge()

        if not merge_result.success:
            return PostTrainingResult(
                success=False,
                merged_model_path="",
                exports={},
                evaluation_passed=False,
                total_time_seconds=time.time() - start_time,
                error_message=f"Merge failed: {merge_result.error_message}",
            )

        merged_model_path = merge_result.output_path

        # Phase 2: Evaluate (optional)
        evaluation_passed = None
        evaluation_report_path = None

        if not self.config.skip_evaluation:
            logger.info("=" * 60)
            logger.info("PHASE 2: Evaluating merged model")
            logger.info("=" * 60)

            try:
                eval_result = self._run_evaluation(merged_model_path)
                evaluation_passed = eval_result.passed
                evaluation_report_path = self._save_evaluation_report(eval_result)

                if not evaluation_passed:
                    warnings.append(
                        "Evaluation did not pass. Model may not meet quality thresholds."
                    )
                    logger.warning("Evaluation did not pass, but continuing with export...")
            except Exception as e:
                warnings.append(f"Evaluation failed: {e}")
                logger.warning(f"Evaluation failed: {e}. Continuing with export...")
                evaluation_passed = False

        # Phase 3: Export
        logger.info("=" * 60)
        logger.info("PHASE 3: Exporting model")
        logger.info("=" * 60)

        export_result = self._run_export(merged_model_path)

        if not export_result.success:
            warnings.append(f"Export partially failed: {export_result.error_message}")

        total_time = time.time() - start_time

        return PostTrainingResult(
            success=True,
            merged_model_path=merged_model_path,
            exports=export_result.exports,
            evaluation_passed=evaluation_passed,
            total_time_seconds=total_time,
            evaluation_report_path=evaluation_report_path,
            warnings=warnings,
        )

    def run_merge_only(self) -> MergeResult:
        """
        Run only the merge phase.

        Returns:
            MergeResult from the merge operation
        """
        logger.info("Running merge-only mode")
        return self._run_merge()

    def run_export_only(self, model_path: str) -> ExportResult:
        """
        Run only the export phase.

        Args:
            model_path: Path to the model to export

        Returns:
            ExportResult from the export operation
        """
        logger.info("Running export-only mode")
        return self._run_export(model_path)

    def _run_merge(self) -> MergeResult:
        """Execute the merge phase."""
        merge_config = MergeConfig(
            base_model_path=self.config.base_model,
            adapter_path=self.config.adapter_path,
            output_path=str(self.output_dir / "merged"),
        )

        merger = LoRAMerger(merge_config)
        return merger.merge()

    def _run_evaluation(self, model_path: str):
        """Execute the evaluation phase."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model for evaluation: {model_path}")

        # Load the merged model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Create evaluation pipeline
        eval_pipeline = EvaluationPipeline(
            model=model,
            tokenizer=tokenizer,
            quiz_data_path=self.config.quiz_data_path,
        )

        # Run full evaluation
        return eval_pipeline.run_full_evaluation()

    def _run_export(self, model_path: str) -> ExportResult:
        """Execute the export phase."""
        export_config = ExportConfig(
            model_path=model_path,
            output_dir=str(self.output_dir / "exports"),
            export_safetensors=self.config.export_safetensors,
            export_gguf=self.config.export_gguf,
            gguf_quantization=self.config.gguf_quantization,
        )

        exporter = ModelExporter(export_config)
        return exporter.export_all()

    def _save_evaluation_report(self, eval_result) -> str:
        """Save evaluation report and return path."""
        report_dir = self.output_dir / "reports"
        report_generator = ReportGenerator(str(report_dir))
        paths = report_generator.generate_all(eval_result)
        return str(paths["json"])
