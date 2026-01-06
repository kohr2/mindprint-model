"""
PPOPipeline - SFT + Reward Model + PPO Training Orchestration.

Three-phase training:
1. SFT training on each topic (3 epochs)
2. Reward model training on preference pairs
3. PPO refinement with learned reward signal

Optimized for Mac Studio M2 Ultra (MPS backend, fp16).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any
from enum import Enum
import logging
import time
import json

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .sft_trainer import SFTTrainer, SFTConfig
from .reward_model import RewardModelTrainer, RewardConfig
from .ppo_trainer import PPOTrainer, PPOConfig
from .mps_utils import mps_empty_cache

logger = logging.getLogger(__name__)


class TopicStatus(Enum):
    """Status of topic training progress."""

    PENDING = "pending"
    SFT_COMPLETE = "sft_complete"
    REWARD_TRAINING = "reward_training"
    PPO_TRAINING = "ppo_training"
    PASSED = "passed"
    FAILED = "failed"


@dataclass
class PipelineConfig:
    """Configuration for the PPO training pipeline."""

    # SFT settings
    sft_epochs_per_topic: int = 3
    sft_learning_rate: float = 3e-4
    sft_batch_size: int = 4

    # Reward model settings
    reward_model_epochs: int = 1
    reward_learning_rate: float = 1e-5
    reward_batch_size: int = 4

    # PPO settings
    ppo_steps_per_topic: int = 100
    ppo_learning_rate: float = 1e-5
    ppo_batch_size: int = 4
    ppo_kl_penalty: float = 0.2

    # Thresholds
    topic_pass_threshold: float = 0.85  # Reward score to pass

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
    sft_loss: float = 0.0
    reward_accuracy: float = 0.0
    reward_score: float = 0.0
    ppo_reward_mean: float = 0.0
    training_time_seconds: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "topic_id": self.topic_id,
            "status": self.status.value,
            "sft_loss": self.sft_loss,
            "reward_accuracy": self.reward_accuracy,
            "reward_score": self.reward_score,
            "ppo_reward_mean": self.ppo_reward_mean,
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


class PPOPipeline:
    """
    Orchestrates three-phase training: SFT → Reward Model → PPO.

    Features:
    - Topic-level SFT for knowledge transfer
    - Reward model training on preference pairs
    - PPO with learned reward signal
    - Unit-level adapter merging
    - Checkpoint save/resume
    - MPS-optimized for Mac Studio M2 Ultra
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: PipelineConfig,
    ):
        """
        Initialize the pipeline.

        Args:
            model: Base model to fine-tune
            tokenizer: Tokenizer for the model
            config: Pipeline configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_model: Optional[Any] = None

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create output directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.units: List[UnitProgress] = []
        self.start_time: float = 0.0

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
        Train a single topic with three-phase training and adaptive config.

        Flow:
        1. Compute data quality metrics
        2. Generate adaptive training config
        3. SFT training on Q&A pairs (with adaptive epochs/batch size)
        4. Reward model training on preference pairs (with adaptive epochs)
        5. PPO training with reward signal (with adaptive steps)

        Args:
            topic_data: Topic data with sft_data, preference_pairs, prompts

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
            # NEW: Compute data quality and generate adaptive config
            from .adaptive_config import AdaptiveConfigGenerator

            config_gen = AdaptiveConfigGenerator(
                baseline_sft_epochs=self.config.sft_epochs_per_topic,
                baseline_sft_lr=self.config.sft_learning_rate,
                baseline_sft_batch_size=self.config.sft_batch_size,
                baseline_reward_epochs=self.config.reward_model_epochs,
                baseline_reward_lr=self.config.reward_learning_rate,
                baseline_reward_batch_size=self.config.reward_batch_size,
                baseline_ppo_steps=self.config.ppo_steps_per_topic,
                baseline_ppo_lr=self.config.ppo_learning_rate,
                baseline_pass_threshold=self.config.topic_pass_threshold,
            )

            metrics = config_gen.compute_data_quality(
                sft_data=topic_data.get("sft_data", []),
                preference_pairs=topic_data.get("preference_pairs", [])
            )

            # Check if data is trainable
            if not metrics.is_trainable:
                logger.warning(
                    f"Topic {topic_id} has insufficient data quality:\n"
                    f"  Examples: {metrics.example_count}\n"
                    f"  Avg output length: {metrics.avg_output_length:.0f}\n"
                    f"  Voice density: {metrics.voice_marker_density:.1f}%"
                )
                progress.status = TopicStatus.FAILED
                progress.training_time_seconds = time.time() - start_time
                return progress

            adaptive_config = config_gen.generate_config(metrics)

            logger.info(f"Adaptive config for {topic_id}:")
            logger.info(f"  Data quality:")
            logger.info(f"    - Examples: {metrics.example_count}")
            logger.info(f"    - Avg output length: {metrics.avg_output_length:.0f}")
            logger.info(f"    - Voice density: {metrics.voice_marker_density:.1f}%")
            logger.info(f"    - Preference quality: {metrics.preference_quality_score:.2f}")
            logger.info(f"  Training config:")
            logger.info(f"    - SFT: {adaptive_config.sft_epochs} epochs, "
                       f"lr={adaptive_config.sft_learning_rate:.1e}, "
                       f"batch={adaptive_config.sft_batch_size}")
            logger.info(f"    - Reward: {adaptive_config.reward_epochs} epochs")
            logger.info(f"    - PPO: {adaptive_config.ppo_steps} steps")
            logger.info(f"    - Pass threshold: {adaptive_config.pass_threshold}")
            logger.info(f"  Rationale: {adaptive_config.rationale}")

            # Phase 1: SFT Training (with adaptive config)
            sft_config = SFTConfig(
                learning_rate=adaptive_config.sft_learning_rate,
                num_epochs=adaptive_config.sft_epochs,
                per_device_batch_size=adaptive_config.sft_batch_size,
                output_dir=self.config.output_dir,  # Enable adapter saving
                save_adapters=True,
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
            self.model = sft_trainer.get_model()

            # Phase 2: Reward Model Training (with adaptive config)
            progress.status = TopicStatus.REWARD_TRAINING
            reward_config = RewardConfig(
                learning_rate=adaptive_config.reward_learning_rate,
                num_epochs=adaptive_config.reward_epochs,
                per_device_batch_size=adaptive_config.reward_batch_size,
            )
            reward_trainer = RewardModelTrainer(
                self.model, self.tokenizer, reward_config
            )

            reward_result = reward_trainer.train(
                topic_data.get("preference_pairs", [])
            )

            if not reward_result.success:
                progress.status = TopicStatus.FAILED
                progress.training_time_seconds = time.time() - start_time
                return progress

            progress.reward_accuracy = reward_result.final_accuracy
            self.reward_model = reward_trainer.get_reward_model()

            # Phase 3: PPO Training (with adaptive config)
            progress.status = TopicStatus.PPO_TRAINING
            ppo_config = PPOConfig(
                learning_rate=adaptive_config.ppo_learning_rate,
                max_steps=adaptive_config.ppo_steps,
                per_device_batch_size=self.config.ppo_batch_size,
                kl_penalty=self.config.ppo_kl_penalty,
                output_dir=self.config.output_dir,  # Enable adapter saving
                save_adapters=True,
            )
            ppo_trainer = PPOTrainer(
                self.model, self.tokenizer, self.reward_model, ppo_config
            )

            prompts = topic_data.get("prompts", [])
            if not prompts:
                # Extract prompts from SFT data
                prompts = [d["question"] for d in topic_data.get("sft_data", [])]

            ppo_result = ppo_trainer.train_on_topic(prompts, topic_id)

            if ppo_result.success:
                progress.ppo_reward_mean = ppo_result.final_reward_mean
                progress.reward_score = ppo_result.final_reward_mean
                self.model = ppo_trainer.get_model()

                # Check if passed adaptive threshold
                if progress.reward_score >= adaptive_config.pass_threshold:
                    progress.status = TopicStatus.PASSED
                    logger.info(
                        f"Topic {topic_id} PASSED: "
                        f"reward {progress.reward_score:.4f} >= {adaptive_config.pass_threshold:.2f}"
                    )
                else:
                    progress.status = TopicStatus.FAILED
                    logger.info(
                        f"Topic {topic_id} FAILED: "
                        f"reward {progress.reward_score:.4f} < {adaptive_config.pass_threshold:.2f}"
                    )
            else:
                progress.status = TopicStatus.FAILED

            progress.training_time_seconds = time.time() - start_time

            logger.info(
                f"Topic {topic_id} complete: status={progress.status.value}, "
                f"reward={progress.reward_score:.2f}"
            )

            return progress

        except Exception as e:
            import traceback
            logger.error(f"Topic {topic_id} training failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            progress.status = TopicStatus.FAILED
            progress.training_time_seconds = time.time() - start_time
            return progress

    def _merge_unit_adapters(self, unit_progress: UnitProgress) -> None:
        """
        Merge all adapters from a unit into the base model.

        This performs incremental merging of all topic adapters from the unit,
        combining SFT + Reward + PPO adapters in sequence.

        Args:
            unit_progress: Unit progress containing all topic results
        """
        logger.info(f"Merging adapters for unit: {unit_progress.unit_id}")

        from .merge import LoRAMerger, MergeConfig
        from .adapter_utils import get_adapter_paths, get_merged_adapter_path, parse_topic_id

        # Collect all adapter paths from this unit's passed topics
        adapter_paths = []

        for chapter in unit_progress.chapters:
            for topic in chapter.topics:
                if topic.status == TopicStatus.PASSED:
                    try:
                        # Parse topic ID to get components
                        unit_id, chapter_id, topic_name = parse_topic_id(topic.topic_id)

                        # Get paths for this topic's adapters
                        sft_path, reward_path, ppo_path = get_adapter_paths(
                            self.config.output_dir,
                            unit_id,
                            chapter_id,
                            topic_name
                        )

                        # Add existing adapter paths (PPO includes all training)
                        if ppo_path.exists():
                            adapter_paths.append(str(ppo_path))
                            logger.debug(f"Added adapter: {ppo_path}")
                        elif sft_path.exists():
                            # Fallback to SFT if PPO doesn't exist
                            adapter_paths.append(str(sft_path))
                            logger.debug(f"Added adapter: {sft_path}")

                    except Exception as e:
                        logger.warning(f"Failed to get adapter paths for {topic.topic_id}: {e}")

        if not adapter_paths:
            logger.warning(f"No adapters found to merge for {unit_progress.unit_id}")
            return

        logger.info(f"Merging {len(adapter_paths)} adapters from {unit_progress.unit_id}")

        try:
            # Create merge output path
            merge_output = get_merged_adapter_path(
                self.config.output_dir,
                unit_progress.unit_id
            )

            # Create merge config
            merge_config = MergeConfig(
                base_model_path=self.model.config.name_or_path,
                adapter_path=adapter_paths[0],  # Required but overridden by merge_incremental
                output_path=str(merge_output),
                dtype="bfloat16",  # Match training dtype
                device="auto",
                verify_merge=True
            )

            # Perform incremental merge
            merger = LoRAMerger(merge_config)
            result = merger.merge_incremental(adapter_paths)

            if result.success:
                logger.info(
                    f"Successfully merged {len(adapter_paths)} adapters to {result.output_path} "
                    f"in {result.merge_time_seconds:.1f}s"
                )
                unit_progress.merged = True
            else:
                logger.error(f"Merge failed: {result.error_message}")

        except Exception as e:
            logger.error(f"Failed to merge unit adapters: {e}")
            import traceback
            logger.error(traceback.format_exc())

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
        """Group training data by topic ID."""
        if sft_data is None:
            sft_data = self._load_sft_data()
        if preference_data is None:
            preference_data = self._load_preference_data()

        grouped: Dict[str, Dict] = {}
        prompt_to_topic: Dict[str, str] = {}  # Map prompts to topic IDs

        # Group SFT data and build prompt-to-topic mapping
        for item in sft_data:
            # Support both 'topic_id' and 'source' field names
            topic_id = item.get("topic_id") or item.get("source", "unknown")
            if topic_id not in grouped:
                grouped[topic_id] = {
                    "sft_data": [],
                    "preference_pairs": [],
                    "prompts": [],
                }
            # Normalize field names: support both instruction/output and question/answer
            question = item.get("question") or item.get("instruction", "")
            normalized_item = {
                "question": question,
                "answer": item.get("answer") or item.get("output", ""),
                "topic_id": topic_id,
            }
            grouped[topic_id]["sft_data"].append(normalized_item)
            grouped[topic_id]["prompts"].append(question)
            # Build mapping for preference data lookup
            prompt_to_topic[question] = topic_id

        # Group preference data, using prompt mapping if no topic_id
        for item in preference_data:
            # Support both 'topic_id' and 'source' field names
            topic_id = item.get("topic_id") or item.get("source")
            # If no topic_id, try to match by prompt
            if not topic_id:
                prompt = item.get("prompt", "")
                topic_id = prompt_to_topic.get(prompt, "unknown")
            if topic_id not in grouped:
                grouped[topic_id] = {
                    "sft_data": [],
                    "preference_pairs": [],
                    "prompts": [],
                }
            grouped[topic_id]["preference_pairs"].append(item)

        return grouped

    def _organize_curriculum(self, grouped_data: Dict) -> List[Dict]:
        """Organize grouped data into unit/chapter/topic hierarchy."""
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

            if unit_id not in units:
                units[unit_id] = {"unit_id": unit_id, "chapters": {}}

            if chapter_id not in units[unit_id]["chapters"]:
                units[unit_id]["chapters"][chapter_id] = {
                    "chapter_id": chapter_id,
                    "topics": [],
                }

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
