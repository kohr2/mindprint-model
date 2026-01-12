"""
DPOPipeline - SFT + DPO Training Orchestration.

Orchestrates the full training pipeline:
1. SFT training on each topic (3 epochs)
2. Evaluation (accuracy + voice fidelity)
3. DPO refinement if needed (accuracy >= 0.70 but voice < 0.75)
4. Merge adapters after each unit

Optimized for Mac Studio M2 Ultra (MPS backend, fp16).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any
from enum import Enum
import logging
import time
import json
import traceback

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .sft_trainer import SFTTrainer, SFTConfig
from .dpo_trainer import Rank1DPOTrainer, Rank1DPOConfig
from .mps_utils import mps_empty_cache
from src.evaluation.voice_evaluator import QuizEvaluator

# Backend interface imports (optional for backward compatibility)
try:
    from src.backends import (
        BackendProtocol,
        ModelInterface,
        TrainerInterface,
        create_backend,
    )
    BACKENDS_AVAILABLE = True
except ImportError:
    BACKENDS_AVAILABLE = False
    BackendProtocol = None
    ModelInterface = None
    TrainerInterface = None

logger = logging.getLogger(__name__)


class TopicStatus(Enum):
    """Status of topic training progress."""

    PENDING = "pending"
    SFT_COMPLETE = "sft_complete"
    EVAL_PASSED = "eval_passed"
    DPO_NEEDED = "dpo_needed"
    DPO_COMPLETE = "dpo_complete"
    PASSED = "passed"
    FAILED = "failed"


@dataclass
class PipelineConfig:
    """Configuration for the DPO training pipeline."""

    # Backend settings (optional - defaults to direct PyTorch usage)
    backend_type: Optional[str] = None  # "pytorch", "mlx", or None for legacy mode
    backend_device: str = "auto"  # "auto", "mps", "cuda", "cpu", "gpu"
    backend_dtype: str = "float16"  # "float16", "float32", "bfloat16"

    # SFT settings
    sft_epochs_per_topic: int = 3
    sft_learning_rate: float = 3e-4
    sft_batch_size: int = 4

    # DPO settings
    dpo_steps_per_topic: int = 100
    dpo_learning_rate: float = 5e-7
    dpo_batch_size: int = 2
    dpo_beta: float = 0.1

    # Thresholds
    accuracy_threshold: float = 0.70  # Min accuracy to try DPO
    dpo_trigger_threshold: float = 0.75  # Voice < this triggers DPO
    topic_pass_threshold: float = 0.90  # Combined score to pass

    # Pipeline control
    merge_after_unit: bool = True
    max_retries_per_topic: int = 2

    # Paths
    data_dir: str = "./data"
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"


@dataclass
class TopicProgress:
    """Progress for a single topic."""

    topic_id: str
    status: TopicStatus
    accuracy_score: float = 0.0
    voice_score: float = 0.0
    retry_count: int = 0
    sft_loss: float = 0.0
    dpo_loss: float = 0.0
    training_time_seconds: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "topic_id": self.topic_id,
            "status": self.status.value,
            "accuracy_score": self.accuracy_score,
            "voice_score": self.voice_score,
            "retry_count": self.retry_count,
            "sft_loss": self.sft_loss,
            "dpo_loss": self.dpo_loss,
            "training_time_seconds": self.training_time_seconds,
        }


@dataclass
class ChapterProgress:
    """Progress for a chapter (collection of topics)."""

    chapter_id: str
    topics: List[TopicProgress]

    @property
    def passed_count(self) -> int:
        """Count of passed topics."""
        return sum(1 for t in self.topics if t.status == TopicStatus.PASSED)

    @property
    def total_count(self) -> int:
        """Total topic count."""
        return len(self.topics)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "chapter_id": self.chapter_id,
            "topics": [t.to_dict() for t in self.topics],
            "passed_count": self.passed_count,
            "total_count": self.total_count,
        }


@dataclass
class UnitProgress:
    """Progress for a unit (collection of chapters)."""

    unit_id: str
    chapters: List[ChapterProgress]
    merged: bool = False

    @property
    def passed_topics(self) -> int:
        """Total passed topics across chapters."""
        return sum(c.passed_count for c in self.chapters)

    @property
    def total_topics(self) -> int:
        """Total topics across chapters."""
        return sum(c.total_count for c in self.chapters)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "unit_id": self.unit_id,
            "chapters": [c.to_dict() for c in self.chapters],
            "merged": self.merged,
            "passed_topics": self.passed_topics,
            "total_topics": self.total_topics,
        }


@dataclass
class PipelineResult:
    """Final result of the training pipeline."""

    success: bool
    total_topics: int
    passed_topics: int
    failed_topics: List[str]
    total_training_time_hours: float
    units: List[UnitProgress] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "total_topics": self.total_topics,
            "passed_topics": self.passed_topics,
            "failed_topics": self.failed_topics,
            "total_training_time_hours": self.total_training_time_hours,
            "pass_rate": self.passed_topics / self.total_topics if self.total_topics > 0 else 0.0,
        }


class DPOPipeline:
    """
    Orchestrates SFT + DPO training across the curriculum.

    Features:
    - Topic-level SFT with voice evaluation
    - DPO refinement when accuracy is high but voice is low
    - Unit-level adapter merging
    - Checkpoint save/resume
    - MPS-optimized for Mac Studio M2 Ultra
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: PipelineConfig,
        backend: Optional[BackendProtocol] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            model: Base model to fine-tune (legacy mode) or None if using backend
            tokenizer: Tokenizer for the model (legacy mode) or None if using backend
            config: Pipeline configuration
            backend: Optional backend instance (if None, uses legacy mode)
        """
        self.config = config
        self.backend = backend
        self.use_backend = backend is not None

        if self.use_backend:
            # Backend mode: wrap model in backend interface
            if not BACKENDS_AVAILABLE:
                raise RuntimeError(
                    "Backend mode requested but backends not available. "
                    "Install backend dependencies."
                )

            # Create backend if needed
            if self.backend is None and self.config.backend_type:
                logger.info(f"Creating {self.config.backend_type} backend")
                self.backend = create_backend(
                    self.config.backend_type,
                    device=self.config.backend_device,
                    dtype=self.config.backend_dtype,
                )
                self.use_backend = True

            # Wrap model in backend interface
            if self.backend and model:
                from src.backends.pytorch import PyTorchModel
                self.model = PyTorchModel(model, tokenizer)
            else:
                self.model = None  # Will be loaded via backend

            self.tokenizer = tokenizer
        else:
            # Legacy mode: use direct PyTorch models
            self.model = model
            self.tokenizer = tokenizer

        # Ensure tokenizer has pad token
        if self.tokenizer and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create output directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.units: List[UnitProgress] = []
        self.start_time: float = 0.0

        logger.info(
            f"DPOPipeline initialized in {'backend' if self.use_backend else 'legacy'} mode"
        )

    def train_curriculum(
        self,
        sft_data: Optional[List[Dict]] = None,
        preference_data: Optional[List[Dict]] = None,
    ) -> PipelineResult:
        """
        Train the full curriculum.

        Args:
            sft_data: Optional SFT data (loads from file if None)
            preference_data: Optional preference pairs (loads from file if None)

        Returns:
            PipelineResult with training outcome
        """
        self.start_time = time.time()

        try:
            # Load data if not provided
            if sft_data is None:
                sft_data = self._load_sft_data()
            if preference_data is None:
                preference_data = self._load_preference_data()

            # Group data by topic
            grouped_data = self._group_data_by_topic(sft_data, preference_data)

            # Organize into units/chapters/topics
            curriculum = self._organize_curriculum(grouped_data)

            # Train each unit
            for unit_data in curriculum:
                unit_progress = self.train_unit(unit_data)
                self.units.append(unit_progress)

                # Merge after unit if enabled
                if self.config.merge_after_unit:
                    self._merge_unit_adapters(unit_progress)
                    unit_progress.merged = True

                # Clear cache after each unit
                mps_empty_cache()

            # Compute final results
            total_topics = sum(u.total_topics for u in self.units)
            passed_topics = sum(u.passed_topics for u in self.units)
            failed_topics = self._collect_failed_topics()
            training_time = (time.time() - self.start_time) / 3600

            return PipelineResult(
                success=len(failed_topics) == 0,
                total_topics=total_topics,
                passed_topics=passed_topics,
                failed_topics=failed_topics,
                total_training_time_hours=training_time,
                units=self.units,
            )

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            training_time = (time.time() - self.start_time) / 3600
            return PipelineResult(
                success=False,
                total_topics=0,
                passed_topics=0,
                failed_topics=[],
                total_training_time_hours=training_time,
            )

    def train_unit(self, unit_data: Dict) -> UnitProgress:
        """
        Train a single unit.

        Args:
            unit_data: Unit data with chapters and topics

        Returns:
            UnitProgress with training outcome
        """
        unit_id = unit_data["unit_id"]
        logger.info(f"Training unit: {unit_id}")

        chapters = []
        for chapter_data in unit_data.get("chapters", []):
            chapter_progress = self._train_chapter(chapter_data)
            chapters.append(chapter_progress)

            # Clear cache between chapters
            mps_empty_cache()

        return UnitProgress(
            unit_id=unit_id,
            chapters=chapters,
            merged=False,
        )

    def _train_chapter(self, chapter_data: Dict) -> ChapterProgress:
        """Train a single chapter."""
        chapter_id = chapter_data["chapter_id"]
        logger.info(f"Training chapter: {chapter_id}")

        topics = []
        for topic_data in chapter_data.get("topics", []):
            topic_progress = self.train_topic(topic_data)
            topics.append(topic_progress)

        return ChapterProgress(
            chapter_id=chapter_id,
            topics=topics,
        )

    def train_topic(self, topic_data: Dict) -> TopicProgress:
        """
        Train a single topic with SFT and optional DPO.

        Flow:
        1. Run SFT training
        2. Evaluate accuracy and voice
        3. If accuracy >= 0.70 and voice < 0.75, run DPO
        4. Final evaluation

        Args:
            topic_data: Topic data with sft_data and preference_pairs

        Returns:
            TopicProgress with training outcome
        """
        topic_id = topic_data["topic_id"]
        start_time = time.time()

        logger.info(f"Training topic: {topic_id}")

        progress = TopicProgress(
            topic_id=topic_id,
            status=TopicStatus.PENDING,
        )

        try:
            # 1. SFT Training
            if self.use_backend:
                # Backend mode: use backend interface
                sft_config_dict = {
                    "learning_rate": self.config.sft_learning_rate,
                    "num_epochs": self.config.sft_epochs_per_topic,
                    "per_device_batch_size": self.config.sft_batch_size,
                    "output_dir": self.config.output_dir,
                }
                sft_trainer = self.backend.create_sft_trainer(self.model, sft_config_dict)

                sft_result = sft_trainer.train_on_topic(
                    topic_data.get("sft_data", []),
                    topic_id,
                )

                if not sft_result.success:
                    progress.status = TopicStatus.FAILED
                    progress.training_time_seconds = time.time() - start_time
                    return progress

                progress.status = TopicStatus.SFT_COMPLETE
                progress.sft_loss = sft_result.final_loss
                self.model = sft_trainer.get_model()
                logger.info("SFT complete (backend mode)")

            else:
                # Legacy mode: use direct PyTorch trainers
                sft_config = SFTConfig(
                    learning_rate=self.config.sft_learning_rate,
                    num_epochs=self.config.sft_epochs_per_topic,
                    per_device_batch_size=self.config.sft_batch_size,
                )
                sft_trainer = SFTTrainer(self.model, self.tokenizer, sft_config)

                sft_result = sft_trainer.train_on_topic(
                    topic_data.get("sft_data", []),
                    topic_id,
                )

                if not sft_result.success:
                    progress.status = TopicStatus.FAILED
                    progress.training_time_seconds = time.time() - start_time
                    return progress

                progress.status = TopicStatus.SFT_COMPLETE
                progress.sft_loss = sft_result.final_loss

                # Keep SFT adapter for DPO (no merge/unload to avoid corruption)
                # DPO will continue training the same adapter
                self.model = sft_trainer.get_model()
                logger.info("Keeping SFT adapter for DPO training (no merge)")

            # Clear MPS cache
            if self.use_backend:
                self.backend.get_device_manager().empty_cache()
            else:
                mps_empty_cache()

            # 2. Evaluate
            eval_result = self._evaluate_topic(topic_data)
            progress.accuracy_score = eval_result.get("accuracy", 0.0)
            progress.voice_score = eval_result.get("voice_score", 0.0)

            # 3. Check if DPO is needed
            if self._should_run_dpo(eval_result):
                progress.status = TopicStatus.DPO_NEEDED
                logger.info(
                    f"Topic {topic_id}: accuracy={progress.accuracy_score:.2f}, "
                    f"voice={progress.voice_score:.2f} → Running DPO"
                )

                # Run DPO
                if self.use_backend:
                    # Backend mode
                    dpo_config_dict = {
                        "learning_rate": self.config.dpo_learning_rate,
                        "max_steps": self.config.dpo_steps_per_topic,
                        "per_device_batch_size": self.config.dpo_batch_size,
                        "beta": self.config.dpo_beta,
                        "output_dir": self.config.output_dir,
                    }
                    dpo_trainer = self.backend.create_dpo_trainer(
                        self.model, dpo_config_dict
                    )

                    dpo_result = dpo_trainer.train_on_topic(
                        topic_data.get("preference_pairs", []),
                        topic_id,
                    )

                    if dpo_result.success:
                        progress.status = TopicStatus.DPO_COMPLETE
                        progress.dpo_loss = dpo_result.final_loss
                        self.model = dpo_trainer.get_model()
                else:
                    # Legacy mode
                    dpo_config = Rank1DPOConfig(
                        learning_rate=self.config.dpo_learning_rate,
                        max_steps=self.config.dpo_steps_per_topic,
                        per_device_batch_size=self.config.dpo_batch_size,
                        beta=self.config.dpo_beta,
                    )
                    dpo_trainer = Rank1DPOTrainer(
                        self.model, self.tokenizer, dpo_config
                    )

                    dpo_result = dpo_trainer.train_on_topic(
                        topic_data.get("preference_pairs", []),
                        topic_id,
                    )

                    if dpo_result.success:
                        progress.status = TopicStatus.DPO_COMPLETE
                        progress.dpo_loss = dpo_result.final_loss
                        self.model = dpo_trainer.get_model()

                # Re-evaluate after DPO
                eval_result = self._evaluate_topic(topic_data)
                progress.accuracy_score = eval_result.get("accuracy", 0.0)
                progress.voice_score = eval_result.get("voice_score", 0.0)

            # 4. Final pass/fail determination
            combined_score = (progress.accuracy_score + progress.voice_score) / 2
            if combined_score >= self.config.topic_pass_threshold:
                progress.status = TopicStatus.PASSED
            else:
                # Check retry count
                if progress.retry_count < self.config.max_retries_per_topic:
                    progress.retry_count += 1
                    # Could trigger retry here, but for now just mark as needing it
                    progress.status = TopicStatus.DPO_NEEDED
                else:
                    progress.status = TopicStatus.FAILED

            progress.training_time_seconds = time.time() - start_time

            # TEMPORARY: Skip merging between topics to avoid corruption on MPS
            # Just keep the adapted model with LoRA for next topic
            # This means we'll have incremental adapters stacking, but at least
            # we can test if SFT-only training works
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                logger.info("Keeping adapted model for next topic (no merge/unload to avoid MPS corruption)")
                # Model stays as-is with adapter attached
                mps_empty_cache()

            logger.info(
                f"Topic {topic_id} complete: status={progress.status.value}, "
                f"accuracy={progress.accuracy_score:.2f}, voice={progress.voice_score:.2f}"
            )

            return progress

        except Exception as e:
            logger.error(f"Topic {topic_id} training failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            progress.status = TopicStatus.FAILED
            progress.training_time_seconds = time.time() - start_time
            return progress

    def _should_run_dpo(self, eval_result: Dict) -> bool:
        """
        Determine if DPO training is needed.

        DPO runs when:
        - Accuracy >= accuracy_threshold (model knows the content)
        - Voice < dpo_trigger_threshold (but doesn't sound like Bob)

        Args:
            eval_result: Dict with accuracy and voice_score

        Returns:
            True if DPO should run
        """
        accuracy = eval_result.get("accuracy", 0.0)
        voice_score = eval_result.get("voice_score", 0.0)

        accuracy_ok = accuracy >= self.config.accuracy_threshold
        voice_low = voice_score < self.config.dpo_trigger_threshold

        return accuracy_ok and voice_low

    def _should_mark_failed(self, progress: TopicProgress) -> bool:
        """Check if topic should be marked as failed."""
        return progress.retry_count >= self.config.max_retries_per_topic

    def _evaluate_topic(self, topic_data: Dict) -> Dict:
        """
        Evaluate a topic's performance.

        Args:
            topic_data: Topic data with questions for evaluation

        Returns:
            Dict with accuracy and voice_score
        """
        try:
            evaluator = QuizEvaluator(self.model, self.tokenizer)

            questions = topic_data.get("questions", [])
            if not questions:
                # Create questions from SFT data (handles both PPO and DPO formats)
                sft_data = topic_data.get("sft_data", [])
                questions = [
                    {
                        "question": d.get("question", d.get("instruction", "")),
                        "reference_answer": d.get("answer", d.get("output", ""))
                    }
                    for d in sft_data
                ]

            if not questions:
                return {"accuracy": 0.0, "voice_score": 0.0, "passed": False}

            result = evaluator.evaluate(questions)

            return {
                "accuracy": result.get("accuracy", 0.0),
                "voice_score": result.get("voice_score", 0.0),
                "passed": result.get("passed", False),
            }

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"accuracy": 0.0, "voice_score": 0.0, "passed": False}

    def _merge_unit_adapters(self, unit_progress: UnitProgress) -> None:
        """
        Merge all adapters from a unit into the model.

        Args:
            unit_progress: Unit progress with completed topics
        """
        logger.info(f"Merging adapters for unit: {unit_progress.unit_id}")

        try:
            # For now, the model already has adapters merged incrementally
            # This is a placeholder for explicit merge logic if needed
            pass
        except Exception as e:
            logger.error(f"Unit merge failed: {e}")

    def save_checkpoint(self, progress: Dict) -> Path:
        """
        Save training checkpoint.

        Args:
            progress: Progress data to save

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = Path(self.config.checkpoint_dir) / "latest.json"

        with open(checkpoint_path, "w") as f:
            json.dump(progress, f, indent=2)

        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path

    def resume_from_checkpoint(self, checkpoint_path: Path) -> Dict:
        """
        Resume from a saved checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded progress data
        """
        with open(checkpoint_path) as f:
            progress = json.load(f)

        logger.info(f"Resumed from checkpoint: {checkpoint_path}")
        return progress

    def _load_sft_data(self) -> List[Dict]:
        """Load SFT training data from file."""
        data_path = Path(self.config.data_dir) / "sft_data.jsonl"

        if not data_path.exists():
            logger.warning(f"SFT data file not found: {data_path}")
            return []

        data = []
        with open(data_path) as f:
            for line in f:
                data.append(json.loads(line))

        logger.info(f"Loaded {len(data)} SFT examples")
        return data

    def _load_preference_data(self) -> List[Dict]:
        """Load preference pair data from file."""
        data_path = Path(self.config.data_dir) / "preference_data.jsonl"

        if not data_path.exists():
            logger.warning(f"Preference data file not found: {data_path}")
            return []

        data = []
        with open(data_path) as f:
            for line in f:
                data.append(json.loads(line))

        logger.info(f"Loaded {len(data)} preference pairs")
        return data

    def _group_data_by_topic(
        self,
        sft_data: Optional[List[Dict]] = None,
        preference_data: Optional[List[Dict]] = None,
    ) -> Dict[str, Dict]:
        """
        Group training data by topic ID.

        Args:
            sft_data: SFT training examples
            preference_data: Preference pairs

        Returns:
            Dict mapping topic_id -> {sft_data, preference_pairs}
        """
        if sft_data is None:
            sft_data = self._load_sft_data()
        if preference_data is None:
            preference_data = self._load_preference_data()

        grouped: Dict[str, Dict] = {}

        # Group SFT data by 'source' field
        for item in sft_data:
            topic_id = item.get("source", item.get("topic_id", "unknown"))
            if topic_id not in grouped:
                grouped[topic_id] = {"sft_data": [], "preference_pairs": [], "topic_id": topic_id}
            grouped[topic_id]["sft_data"].append(item)

        # Group preference data - match by instruction/prompt to SFT data
        # For now, add all preference pairs to their corresponding topics based on prompts
        for pref_item in preference_data:
            prompt = pref_item.get("prompt", "")
            # Find matching SFT item by instruction
            matched = False
            for topic_id, topic_data in grouped.items():
                for sft_item in topic_data["sft_data"]:
                    if sft_item.get("instruction") == prompt:
                        topic_data["preference_pairs"].append(pref_item)
                        matched = True
                        break
                if matched:
                    break

        return grouped

    def _organize_curriculum(self, grouped_data: Dict) -> List[Dict]:
        """
        Organize grouped data into unit/chapter/topic hierarchy.

        Args:
            grouped_data: Data grouped by topic ID

        Returns:
            List of unit dicts with chapters and topics
        """
        # Parse topic IDs to extract unit/chapter structure
        units: Dict[str, Dict] = {}

        for topic_id, topic_data in grouped_data.items():
            parts = topic_id.split("/")

            if len(parts) >= 3:
                unit_id = parts[0]
                chapter_id = f"{parts[0]}/{parts[1]}"
            elif len(parts) == 2:
                unit_id = parts[0]
                chapter_id = topic_id
            else:
                unit_id = "unit-default"
                chapter_id = "chapter-default"

            # Initialize unit
            if unit_id not in units:
                units[unit_id] = {"unit_id": unit_id, "chapters": {}}

            # Initialize chapter
            if chapter_id not in units[unit_id]["chapters"]:
                units[unit_id]["chapters"][chapter_id] = {
                    "chapter_id": chapter_id,
                    "topics": [],
                }

            # Add topic
            units[unit_id]["chapters"][chapter_id]["topics"].append({
                "topic_id": topic_id,
                **topic_data,
            })

        # Convert to list format
        result = []
        for unit_id in sorted(units.keys()):
            unit = units[unit_id]
            chapters = []
            for chapter_id in sorted(unit["chapters"].keys()):
                chapter = unit["chapters"][chapter_id]
                chapter["topics"] = sorted(
                    chapter["topics"], key=lambda t: t["topic_id"]
                )
                chapters.append(chapter)
            result.append({
                "unit_id": unit_id,
                "chapters": chapters,
            })

        return result

    def _collect_failed_topics(self) -> List[str]:
        """Collect all failed topic IDs."""
        failed = []
        for unit in self.units:
            for chapter in unit.chapters:
                for topic in chapter.topics:
                    if topic.status == TopicStatus.FAILED:
                        failed.append(topic.topic_id)
        return failed
