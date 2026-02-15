"""
DPOPipeline (ORPO-only) - ORPO Training Orchestration.

Orchestrates the ORPO training pipeline:
1. ORPO training on each topic (combined SFT + preference alignment)
2. Evaluation (accuracy + voice fidelity)
3. Merge adapters after each unit

Optimized for Mac Studio M2 Ultra (MPS backend, fp16) and PyTorch backends.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime
import logging
import re
import time
import json
import traceback

try:
    from transformers import PreTrainedModel, PreTrainedTokenizer
except ImportError:
    PreTrainedModel = Any  # type: ignore[misc,assignment]
    PreTrainedTokenizer = Any  # type: ignore[misc,assignment]

try:
    from .mps_utils import mps_empty_cache
except ImportError:
    def mps_empty_cache() -> None:
        """No-op cache clear when torch/MPS utilities are unavailable."""
        return None
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
    ORPO_COMPLETE = "orpo_complete"  # ORPO training complete
    EVAL_PASSED = "eval_passed"  # Evaluation passed
    PASSED = "passed"  # Topic passed all checks
    FAILED = "failed"  # Topic failed


@dataclass
class PipelineConfig:
    """Configuration for the ORPO training pipeline."""

    # Backend settings
    backend_type: Optional[str] = None  # "pytorch", "mlx", or None for legacy mode
    backend_device: str = "auto"  # "auto", "mps", "cuda", "cpu", "gpu"
    backend_dtype: str = "float16"  # "float16", "float32", "bfloat16"

    # ORPO settings - Odds Ratio Preference Optimization
    orpo_steps_per_topic: int = 100
    orpo_learning_rate: float = 3e-4
    orpo_batch_size: int = 4
    orpo_max_length: int = 512
    orpo_lambda: float = 0.1  # Weight for preference term
    orpo_lora_rank: int = 8
    orpo_lora_alpha: float = 16.0
    orpo_lora_dropout: float = 0.05
    orpo_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
    )

    # Thresholds
    accuracy_threshold: float = 0.70  # Min accuracy to pass
    topic_pass_threshold: float = 0.90  # Combined score to pass

    # Pipeline control
    merge_after_unit: bool = True
    max_retries_per_topic: int = 2

    # Early stopping configuration
    max_topics: Optional[int] = None  # Limit total topics to train
    early_stopping_enabled: bool = False  # Enable loss-based early stopping
    early_stopping_patience: int = 3  # Stop if loss CV < threshold for N topics
    early_stopping_cv_threshold: float = 15.0  # Coefficient of variation threshold (%)
    early_stopping_min_topics: int = 10  # Minimum topics before early stopping applies

    # Paths
    data_dir: str = "./data"
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"
    config_filename: Optional[str] = None  # Path to config file for dataset extraction


@dataclass
class TopicProgress:
    """Progress for a single topic."""

    topic_id: str
    status: TopicStatus
    accuracy_score: float = 0.0
    voice_score: float = 0.0
    retry_count: int = 0
    orpo_loss: float = 0.0  # ORPO training loss
    training_time_seconds: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "topic_id": self.topic_id,
            "status": self.status.value,
            "accuracy_score": self.accuracy_score,
            "voice_score": self.voice_score,
            "retry_count": self.retry_count,
            "orpo_loss": self.orpo_loss,
            "training_time_seconds": self.training_time_seconds,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TopicProgress":
        """Reconstruct from dictionary (e.g. checkpoint)."""
        return cls(
            topic_id=d["topic_id"],
            status=TopicStatus(d.get("status", "failed")),
            accuracy_score=float(d.get("accuracy_score", 0.0)),
            voice_score=float(d.get("voice_score", 0.0)),
            retry_count=int(d.get("retry_count", 0)),
            orpo_loss=float(d.get("orpo_loss", 0.0)),
            training_time_seconds=float(d.get("training_time_seconds", 0.0)),
        )


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

    @classmethod
    def from_dict(cls, d: Dict) -> "ChapterProgress":
        """Reconstruct from dictionary (e.g. checkpoint)."""
        topics = [TopicProgress.from_dict(t) for t in d.get("topics", [])]
        return cls(chapter_id=d["chapter_id"], topics=topics)


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

    @classmethod
    def from_dict(cls, d: Dict) -> "UnitProgress":
        """Reconstruct from dictionary (e.g. checkpoint)."""
        chapters = [ChapterProgress.from_dict(c) for c in d.get("chapters", [])]
        return cls(
            unit_id=d["unit_id"],
            chapters=chapters,
            merged=bool(d.get("merged", False)),
        )


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
            "units": [u.to_dict() for u in self.units],
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

            # Check if model is already wrapped as ModelInterface
            if self.backend and model:
                # Check if already a ModelInterface
                if hasattr(model, 'get_tokenizer') and hasattr(model, 'forward'):
                    # Already wrapped
                    self.model = model
                    self.tokenizer = model.get_tokenizer() if hasattr(model, 'get_tokenizer') else tokenizer
                else:
                    # Need to wrap
                    from src.backends.pytorch import PyTorchModel
                    self.model = PyTorchModel(model, tokenizer)
                    self.tokenizer = tokenizer
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

        # Modify output_dir to include dataset name and timestamp
        # This ensures adapter files don't overwrite each other
        dataset_name = self._extract_dataset_name()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = Path(self.config.output_dir)
        self.config.output_dir = str(base_output_dir / f"{dataset_name}_{timestamp}")
        
        logger.info(f"Output directory: {self.config.output_dir}")

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
        preference_data: Optional[List[Dict]] = None,
        initial_progress: Optional[Dict] = None,
    ) -> PipelineResult:
        """
        Train the full curriculum with early stopping support.

        Args:
            preference_data: Optional preference pairs (loads from file if None)
            initial_progress: Optional checkpoint dict from resume; completed topics
                are skipped and adapter state is assumed already loaded.

        Returns:
            PipelineResult with training outcome
        """
        self.start_time = time.time()
        self.units = []

        # Restore from checkpoint so we skip completed topics (do not pre-fill self.units)
        progress_map: Dict[str, TopicProgress] = {}
        passed_loss_history: List[float] = []
        topics_trained = 0
        if initial_progress is not None:
            result_data = initial_progress.get("result", {})
            units_data = result_data.get("units", [])
            if units_data:
                for u in units_data:
                    unit = UnitProgress.from_dict(u)
                    for chapter in unit.chapters:
                        for topic in chapter.topics:
                            progress_map[topic.topic_id] = topic
                            if (
                                topic.status == TopicStatus.PASSED
                                and topic.orpo_loss is not None
                            ):
                                passed_loss_history.append(topic.orpo_loss)
                topics_trained = sum(
                    UnitProgress.from_dict(u).total_topics for u in units_data
                )
                logger.info(
                    f"Resuming: {len(units_data)} units, {topics_trained} topics already done"
                )

        try:
            # Load data if not provided
            if preference_data is None:
                preference_data = self._load_preference_data()

            # Group data by topic
            grouped_data = self._group_data_by_topic(preference_data)

            # Organize into units/chapters/topics
            curriculum = self._organize_curriculum(grouped_data)

            early_stopped = False

            # Train each unit (self.units may already be partially filled from resume)
            for unit_data in curriculum:
                unit_id = unit_data["unit_id"]
                logger.info(f"Training unit: {unit_id}")

                chapters = []
                for chapter_data in unit_data.get("chapters", []):
                    chapter_id = chapter_data["chapter_id"]
                    logger.info(f"Training chapter: {chapter_id}")

                    topics = []
                    for topic_data in chapter_data.get("topics", []):
                        topic_id = topic_data.get("topic_id", "")
                        # Skip if already completed (resume)
                        if topic_id in progress_map:
                            topic_progress = progress_map[topic_id]
                            topics.append(topic_progress)
                            topics_trained += 1
                            if (
                                topic_progress.status == TopicStatus.PASSED
                                and topic_progress.orpo_loss is not None
                            ):
                                passed_loss_history.append(topic_progress.orpo_loss)
                            # Check early stopping
                            if (self.config.early_stopping_enabled
                                and topics_trained >= self.config.early_stopping_min_topics
                                and len(passed_loss_history) >= self.config.early_stopping_patience):
                                recent_losses = passed_loss_history[-self.config.early_stopping_patience:]
                                cv = self._calculate_cv(recent_losses)
                                if cv < self.config.early_stopping_cv_threshold:
                                    logger.info(
                                        f"Early stopping triggered: Loss CV ({cv:.1f}%) < threshold "
                                        f"({self.config.early_stopping_cv_threshold}%) for "
                                        f"{self.config.early_stopping_patience} topics"
                                    )
                                    early_stopped = True
                                    break
                            continue

                        # Check max_topics limit
                        if self.config.max_topics and topics_trained >= self.config.max_topics:
                            logger.info(f"Reached max_topics limit ({self.config.max_topics}). Stopping training.")
                            early_stopped = True
                            break

                        # Train topic
                        topic_progress = self.train_topic(topic_data)
                        topics.append(topic_progress)
                        topics_trained += 1

                        # Track loss for early stopping
                        if (
                            topic_progress.status == TopicStatus.PASSED
                            and topic_progress.orpo_loss is not None
                        ):
                            passed_loss_history.append(topic_progress.orpo_loss)

                        # Check early stopping
                        if (self.config.early_stopping_enabled 
                            and topics_trained >= self.config.early_stopping_min_topics
                            and len(passed_loss_history) >= self.config.early_stopping_patience):
                            
                            recent_losses = passed_loss_history[-self.config.early_stopping_patience:]
                            cv = self._calculate_cv(recent_losses)
                            
                            if cv < self.config.early_stopping_cv_threshold:
                                logger.info(
                                    f"Early stopping triggered: Loss CV ({cv:.1f}%) < threshold "
                                    f"({self.config.early_stopping_cv_threshold}%) for "
                                    f"{self.config.early_stopping_patience} topics"
                                )
                                early_stopped = True
                                break

                    if early_stopped:
                        break

                    # Create chapter progress
                    chapter_progress = ChapterProgress(
                        chapter_id=chapter_id,
                        topics=topics,
                    )
                    chapters.append(chapter_progress)

                    # Clear cache between chapters
                    mps_empty_cache()

                if early_stopped:
                    break

                # Create unit progress
                unit_progress = UnitProgress(
                    unit_id=unit_id,
                    chapters=chapters,
                    merged=False,
                )
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

            if early_stopped:
                logger.info(f"Training stopped early after {topics_trained} topics")

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
        Train a single topic with ORPO (single-stage training).

        Flow:
        1. Run ORPO training (combines SFT + preference alignment)
        2. Evaluate accuracy and voice
        3. Pass/Fail determination

        Args:
            topic_data: Topic data with preference_pairs (ORPO only needs preference pairs)

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
            # 1. ORPO Training (single-stage, replaces SFT+DPO)
            preference_pairs = topic_data.get("preference_pairs", [])
            
            if not preference_pairs:
                logger.warning(f"No preference pairs found for topic {topic_id}, skipping")
                progress.status = TopicStatus.FAILED
                progress.training_time_seconds = time.time() - start_time
                return progress

            if self.use_backend:
                # Backend mode: use backend interface
                orpo_config_dict = {
                    "learning_rate": self.config.orpo_learning_rate,
                    "max_steps": self.config.orpo_steps_per_topic,
                    "per_device_batch_size": self.config.orpo_batch_size,
                    "lambda_orpo": self.config.orpo_lambda,
                    "output_dir": self.config.output_dir,
                    "max_length": self.config.orpo_max_length,
                }
                orpo_trainer = self.backend.create_orpo_trainer(self.model, orpo_config_dict)

                orpo_result = orpo_trainer.train_on_topic(
                    preference_pairs,
                    topic_id,
                )

                if not orpo_result.success:
                    progress.status = TopicStatus.FAILED
                    progress.training_time_seconds = time.time() - start_time
                    return progress

                progress.status = TopicStatus.ORPO_COMPLETE
                progress.orpo_loss = orpo_result.final_loss
                self.model = orpo_trainer.get_model()
                logger.info(f"ORPO complete (backend mode): loss={orpo_result.final_loss:.4f}")

            else:
                # Legacy mode not supported for ORPO - ORPO requires backend
                logger.error("ORPO training requires backend mode. Legacy PyTorch mode not supported.")
                progress.status = TopicStatus.FAILED
                progress.training_time_seconds = time.time() - start_time
                return progress

            # Clear cache
            if self.use_backend:
                self.backend.get_device_manager().empty_cache()
            else:
                mps_empty_cache()

            # 2. Evaluate
            eval_result = self._evaluate_topic(topic_data)
            progress.accuracy_score = eval_result.get("accuracy", 0.0)
            progress.voice_score = eval_result.get("voice_score", 0.0)

            # 3. Final pass/fail determination
            combined_score = (progress.accuracy_score + progress.voice_score) / 2
            if combined_score >= self.config.topic_pass_threshold:
                progress.status = TopicStatus.PASSED
            else:
                # Check retry count
                if progress.retry_count < self.config.max_retries_per_topic:
                    progress.retry_count += 1
                    # Mark as failed but allow retry
                    progress.status = TopicStatus.FAILED
                else:
                    progress.status = TopicStatus.FAILED

            progress.training_time_seconds = time.time() - start_time

            logger.info(
                f"Topic {topic_id} complete: status={progress.status.value}, "
                f"accuracy={progress.accuracy_score:.2f}, voice={progress.voice_score:.2f}, "
                f"orpo_loss={progress.orpo_loss:.4f}"
            )

            return progress

        except Exception as e:
            logger.error(f"Topic {topic_id} training failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            progress.status = TopicStatus.FAILED
            progress.training_time_seconds = time.time() - start_time
            return progress

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
                # Create questions from preference pairs (prompt=question, chosen=reference)
                preference_pairs = topic_data.get("preference_pairs", [])
                questions = [
                    {
                        "question": p.get("prompt", ""),
                        "reference_answer": p.get("chosen", ""),
                    }
                    for p in preference_pairs
                    if p.get("prompt") and p.get("chosen")
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

    def _extract_dataset_name(self) -> str:
        """
        Extract dataset name from config filename or data_dir.
        
        Returns:
            Dataset name (textbook, transcripts, combined, or default)
        """
        dataset_name = "default"
        if self.config.config_filename:
            config_stem = Path(self.config.config_filename).stem
            if "textbook" in config_stem:
                dataset_name = "textbook"
            elif "transcripts" in config_stem:
                dataset_name = "transcripts"
            elif "combined" in config_stem:
                dataset_name = "combined"
            else:
                # Fallback: try to extract from data_dir
                data_dir = Path(self.config.data_dir)
                if data_dir.name in ["textbook", "transcripts", "combined"]:
                    dataset_name = data_dir.name
        else:
            # Fallback: try to extract from data_dir if config_filename not set
            data_dir = Path(self.config.data_dir)
            if data_dir.name in ["textbook", "transcripts", "combined"]:
                dataset_name = data_dir.name
        
        return dataset_name

    def save_checkpoint(self, progress: Dict) -> Path:
        """
        Save training checkpoint with timestamped filename.
        
        Saves both:
        - Timestamped file: checkpoint_{dataset}_{timestamp}.json
        - Latest file: latest.json (for backward compatibility)

        Args:
            progress: Progress data to save

        Returns:
            Path to saved timestamped checkpoint
        """
        # Ensure checkpoint directory exists
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save adapter to checkpoint dir so resume can load it
        if self.use_backend and self.model is not None:
            if getattr(self.model, "has_adapter", lambda: False)():
                adapter_dir = checkpoint_dir / "adapters_latest"
                adapter_dir.mkdir(parents=True, exist_ok=True)
                try:
                    self.model.save_adapter(adapter_dir)
                    progress["adapter_path"] = str(adapter_dir)
                    logger.info(f"Saved adapter to {adapter_dir}")
                except Exception as e:
                    logger.warning(f"Could not save adapter for checkpoint: {e}")
        
        # Extract dataset name (reuse helper method)
        dataset_name = self._extract_dataset_name()
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create timestamped filename
        timestamped_filename = f"checkpoint_{dataset_name}_{timestamp}.json"
        timestamped_path = checkpoint_dir / timestamped_filename
        
        # Save timestamped checkpoint
        with open(timestamped_path, "w") as f:
            json.dump(progress, f, indent=2)
        
        logger.info(f"Saved checkpoint to {timestamped_path}")
        
        # Also save latest.json for backward compatibility
        latest_path = checkpoint_dir / "latest.json"
        with open(latest_path, "w") as f:
            json.dump(progress, f, indent=2)
        
        logger.info(f"Saved latest checkpoint to {latest_path}")
        
        return timestamped_path

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
        preference_data: Optional[List[Dict]] = None,
    ) -> Dict[str, Dict]:
        """
        Group preference data by topic ID.

        Extracts topic IDs from preference prompts using date patterns
        (e.g. "on 2019-02-19?" -> "episode-2019-02-19") or falls back
        to a sequential topic ID.

        Args:
            preference_data: Preference pairs with prompt/chosen/rejected

        Returns:
            Dict mapping topic_id -> {preference_pairs, topic_id}
        """
        if preference_data is None:
            preference_data = self._load_preference_data()

        grouped: Dict[str, Dict] = {}

        for item in preference_data:
            topic_id = self._extract_topic_id(item)
            if topic_id not in grouped:
                grouped[topic_id] = {"preference_pairs": [], "topic_id": topic_id}
            grouped[topic_id]["preference_pairs"].append(item)

        return grouped

    @staticmethod
    def _extract_topic_id(item: Dict) -> str:
        """Extract topic ID from a preference pair.

        Priority:
        1. Explicit 'source' or 'topic_id' field (textbook dataset)
        2. Date pattern in 'prompt' field (transcript dataset)
        3. Fallback to 'unknown'
        """
        source = item.get("source") or item.get("topic_id")
        if source:
            return source
        prompt = item.get("prompt", "")
        match = re.search(r"(\d{4}-\d{2}-\d{2})", prompt)
        if match:
            return f"episode-{match.group(1)}"
        return "general"

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

    def _calculate_cv(self, values: List[float]) -> float:
        """Calculate coefficient of variation (CV) as percentage."""
        if not values or len(values) < 2:
            return 100.0  # High CV if insufficient data
        
        mean = sum(values) / len(values)
        if mean == 0:
            return 100.0
        
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        cv = (std_dev / mean) * 100
        
        return cv
