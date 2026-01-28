"""
Data quality metrics - shared across training approaches.

Provides metrics for assessing training data quality:
- Example count
- Output length distribution
- Voice marker density
- Preference quality scores
"""

from dataclasses import dataclass


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
            15 <= self.example_count <= 30
            and 600 <= self.avg_output_length <= 1500
            and self.voice_marker_density >= 20.0
            and self.preference_quality_score >= 1.5
        )

    @property
    def is_trainable(self) -> bool:
        """Check if data is sufficient for training."""
        return (
            self.example_count >= 2
            and self.avg_output_length >= 50
            and self.voice_marker_density >= 0.0
        )
