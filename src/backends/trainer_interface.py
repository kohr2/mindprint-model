"""
Trainer Interface - Abstract interface for trainers across backends.

Provides unified API for ORPO training that works
with both PyTorch and MLX implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingResult:
    """
    Unified training result across backends.

    Contains metrics and metadata from a training run.
    """

    success: bool
    final_loss: float
    training_time_seconds: float
    samples_trained: int
    adapter_path: str = ""
    error_message: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "final_loss": self.final_loss,
            "training_time_seconds": self.training_time_seconds,
            "samples_trained": self.samples_trained,
            "adapter_path": self.adapter_path,
            "error_message": self.error_message,
            "metrics": self.metrics,
        }


class TrainerInterface(ABC):
    """
    Abstract interface for trainers across backends.

    This interface abstracts away framework-specific training implementations,
    allowing the ORPO pipeline to work with PyTorch or MLX trainers.
    """

    @abstractmethod
    def train(
        self,
        train_data: List[Dict[str, Any]],
    ) -> TrainingResult:
        """
        Run training on data.

        Args:
            train_data: List of training examples
                For ORPO: [{"prompt": str, "chosen": str, "rejected": str}, ...]

        Returns:
            TrainingResult with metrics and status
        """
        ...

    @abstractmethod
    def train_on_topic(
        self,
        topic_data: List[Dict[str, Any]],
        topic_id: str,
    ) -> TrainingResult:
        """
        Train on a single topic.

        Args:
            topic_data: Training data for this topic
            topic_id: Unique identifier for the topic

        Returns:
            TrainingResult for this topic
        """
        ...

    @abstractmethod
    def save_adapter(self, path: Path) -> Path:
        """
        Save trained adapter.

        Args:
            path: Directory to save adapter

        Returns:
            Path where adapter was saved
        """
        ...

    @abstractmethod
    def get_model(self) -> "ModelInterface":
        """
        Get the trained model.

        Returns:
            ModelInterface instance with trained weights/adapter
        """
        ...

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get trainer configuration.

        Returns:
            Configuration dictionary
        """
        ...


class ORPOTrainerInterface(TrainerInterface):
    """
    Interface for ORPO (Odds Ratio Preference Optimization) trainers.

    Extends TrainerInterface with ORPO-specific methods.
    ORPO combines SFT and preference alignment in a single stage,
    eliminating the need for a reference model.
    """

    @abstractmethod
    def get_orpo_stats(self) -> Dict[str, Any]:
        """
        Get ORPO-specific statistics.

        Returns:
            Dictionary with ORPO loss components (NLL loss, odds ratio loss),
            accuracy, odds margins, etc.
        """
        ...
