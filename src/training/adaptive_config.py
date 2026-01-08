"""
Adaptive Configuration - Adjust hyperparameters based on data quality.

Dynamically tunes:
- Number of training epochs
- Batch size
- Learning rate
- Pass threshold

Based on:
- Example count
- Output length distribution
- Voice marker density
- Preference quality scores
"""

from dataclasses import dataclass
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Metrics about training data quality."""

    example_count: int
    avg_output_length: float
    voice_marker_density: float  # Percentage
    preference_quality_score: float  # 0-5 scale

    @property
    def is_high_quality(self) -> bool:
        """Check if data meets high quality thresholds."""
        return (
            15 <= self.example_count <= 30 and
            600 <= self.avg_output_length <= 1500 and
            self.voice_marker_density >= 20.0 and
            self.preference_quality_score >= 1.5
        )

    @property
    def is_trainable(self) -> bool:
        """Check if data is sufficient for training."""
        return (
            self.example_count >= 2 and
            self.avg_output_length >= 50 and
            self.voice_marker_density >= 0.0
        )


@dataclass
class AdaptiveTrainingConfig:
    """Training config adapted to data quality."""

    sft_epochs: int
    sft_learning_rate: float
    sft_batch_size: int

    reward_epochs: int
    reward_learning_rate: float
    reward_batch_size: int

    ppo_steps: int
    ppo_learning_rate: float

    pass_threshold: float

    # Rationale for chosen values
    rationale: str = ""


class AdaptiveConfigGenerator:
    """Generates training configs based on data quality."""

    def __init__(
        self,
        baseline_sft_epochs: int = 3,
        baseline_sft_lr: float = 3e-4,
        baseline_sft_batch_size: int = 4,
        baseline_reward_epochs: int = 3,
        baseline_reward_lr: float = 1e-5,
        baseline_reward_batch_size: int = 8,
        baseline_ppo_steps: int = 100,
        baseline_ppo_lr: float = 1e-5,
        baseline_pass_threshold: float = 0.85,
    ):
        """Initialize with baseline configs."""
        self.baseline_sft_epochs = baseline_sft_epochs
        self.baseline_sft_lr = baseline_sft_lr
        self.baseline_sft_batch_size = baseline_sft_batch_size
        self.baseline_reward_epochs = baseline_reward_epochs
        self.baseline_reward_lr = baseline_reward_lr
        self.baseline_reward_batch_size = baseline_reward_batch_size
        self.baseline_ppo_steps = baseline_ppo_steps
        self.baseline_ppo_lr = baseline_ppo_lr
        self.baseline_pass_threshold = baseline_pass_threshold

    def compute_data_quality(
        self,
        sft_data: List[Dict],
        preference_pairs: List[Dict]
    ) -> DataQualityMetrics:
        """
        Compute quality metrics for topic data.

        Args:
            sft_data: SFT training examples
            preference_pairs: Preference pairs

        Returns:
            DataQualityMetrics
        """
        # Example count
        example_count = len(preference_pairs)

        # Average output length
        outputs = [ex.get("output", ex.get("answer", "")) for ex in sft_data]
        avg_length = sum(len(o) for o in outputs) / len(outputs) if outputs else 0

        # Voice marker density
        voice_markers = self._compute_voice_density(outputs)

        # Preference quality
        pref_quality = self._compute_preference_quality(preference_pairs)

        return DataQualityMetrics(
            example_count=example_count,
            avg_output_length=avg_length,
            voice_marker_density=voice_markers,
            preference_quality_score=pref_quality,
        )

    def _compute_voice_density(self, outputs: List[str]) -> float:
        """Compute voice marker density (percentage)."""
        markers = [
            "Look,", "Okay,", "Here's the thing", "I've seen",
            "**", "systematic", "discipline", "conviction",
            "psychology", "thesis", "gambler", "The key"
        ]

        densities = []
        for output in outputs:
            marker_count = sum(output.lower().count(m.lower()) for m in markers)
            words = len(output.split())
            density = (marker_count / words * 100) if words > 0 else 0
            densities.append(density)

        return sum(densities) / len(densities) if densities else 0.0

    def _compute_preference_quality(self, pairs: List[Dict]) -> float:
        """Compute average preference quality score."""
        scores = []
        for pair in pairs:
            chosen = pair.get("chosen", "")
            rejected = pair.get("rejected", "")

            len_ratio = len(chosen) / (len(rejected) + 1)
            score = min(len_ratio, 3.0)  # Cap at 3x
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    def generate_config(
        self,
        metrics: DataQualityMetrics
    ) -> AdaptiveTrainingConfig:
        """
        Generate adaptive training config based on data quality.

        Strategy:
        - High quality data: Use baseline settings
        - Low example count: Increase epochs, reduce batch size
        - Poor quality: More conservative training, higher threshold
        """
        rationale_parts = []

        # SFT adjustments
        if metrics.example_count < 10:
            sft_epochs = self.baseline_sft_epochs + 2  # 5 epochs
            sft_batch_size = 2  # Smaller batches
            sft_lr = self.baseline_sft_lr * 0.5  # Lower LR
            rationale_parts.append(
                f"Low example count ({metrics.example_count}): "
                f"increased SFT epochs to {sft_epochs}, reduced batch to {sft_batch_size}"
            )
        elif metrics.example_count > 25:
            sft_epochs = self.baseline_sft_epochs  # Keep baseline
            sft_batch_size = 8  # Larger batches
            sft_lr = self.baseline_sft_lr
            rationale_parts.append(
                f"High example count ({metrics.example_count}): "
                f"using larger batch size {sft_batch_size}"
            )
        else:
            sft_epochs = self.baseline_sft_epochs
            sft_batch_size = self.baseline_sft_batch_size
            sft_lr = self.baseline_sft_lr
            rationale_parts.append(
                f"Example count in optimal range ({metrics.example_count}): using baseline SFT config"
            )

        # Reward model adjustments
        if metrics.preference_quality_score < 1.0:
            reward_epochs = self.baseline_reward_epochs + 2  # 5 epochs
            reward_lr = self.baseline_reward_lr * 0.5
            reward_batch_size = 4  # Smaller for noisy data
            rationale_parts.append(
                f"Low preference quality ({metrics.preference_quality_score:.2f}): "
                f"increased reward epochs to {reward_epochs}, smaller batch"
            )
        else:
            reward_epochs = self.baseline_reward_epochs
            reward_lr = self.baseline_reward_lr
            reward_batch_size = self.baseline_reward_batch_size
            rationale_parts.append(
                f"Good preference quality ({metrics.preference_quality_score:.2f}): baseline reward config"
            )

        # PPO adjustments
        if metrics.example_count < 10:
            ppo_steps = self.baseline_ppo_steps + 50  # 150 steps
            ppo_lr = self.baseline_ppo_lr * 0.5  # Lower LR
            rationale_parts.append(
                f"Low examples: increased PPO steps to {ppo_steps}"
            )
        else:
            ppo_steps = self.baseline_ppo_steps
            ppo_lr = self.baseline_ppo_lr

        # Pass threshold adjustments
        if metrics.is_high_quality:
            pass_threshold = self.baseline_pass_threshold  # 0.85
            rationale_parts.append(
                f"High quality data: standard threshold {pass_threshold}"
            )
        elif metrics.voice_marker_density < 15.0:
            pass_threshold = 0.75  # Lower threshold for weaker data
            rationale_parts.append(
                f"Low voice density ({metrics.voice_marker_density:.1f}%): "
                f"reduced threshold to {pass_threshold}"
            )
        else:
            pass_threshold = 0.80  # Middle ground
            rationale_parts.append(
                f"Moderate quality: threshold {pass_threshold}"
            )

        return AdaptiveTrainingConfig(
            sft_epochs=sft_epochs,
            sft_learning_rate=sft_lr,
            sft_batch_size=sft_batch_size,
            reward_epochs=reward_epochs,
            reward_learning_rate=reward_lr,
            reward_batch_size=reward_batch_size,
            ppo_steps=ppo_steps,
            ppo_learning_rate=ppo_lr,
            pass_threshold=pass_threshold,
            rationale=" | ".join(rationale_parts),
        )
