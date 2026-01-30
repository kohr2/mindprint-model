"""
Weights & Biases integration for experiment tracking.

Tracks:
- Hyperparameters
- Training metrics (loss, accuracy, rewards)
- Evaluation metrics (voice fidelity, perplexity)
- Model artifacts
- System metrics (GPU memory, throughput)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class WandBConfig:
    """W&B configuration."""
    project: str = "mindprint"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""


class WandBTracker:
    """Weights & Biases experiment tracker."""
    
    def __init__(self, config: WandBConfig, run_config: Dict[str, Any]):
        """
        Initialize W&B tracker.
        
        Args:
            config: W&B configuration
            run_config: Run configuration to log
        """
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(
                project=config.project,
                entity=config.entity,
                config=run_config,
                tags=config.tags if config.tags else None,
                notes=config.notes if config.notes else None,
            )
            self._enabled = True
        except ImportError:
            self._enabled = False
            self.run = None
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log training metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Current training step
        """
        if not self._enabled:
            return
        
        metrics_with_step = {**metrics, "step": step}
        self.run.log(metrics_with_step)
    
    def log_artifact(self, path: Path, name: str, artifact_type: str) -> None:
        """
        Log model artifact.
        
        Args:
            path: Path to artifact file/directory
            name: Artifact name
            artifact_type: Type of artifact (e.g., "model", "checkpoint")
        """
        if not self._enabled:
            return
        
        artifact = self.wandb.Artifact(name, type=artifact_type)
        if path.is_file():
            artifact.add_file(str(path))
        elif path.is_dir():
            artifact.add_dir(str(path))
        self.run.log_artifact(artifact)
    
    def finish(self) -> None:
        """Finish run."""
        if self._enabled and self.run:
            self.run.finish()
