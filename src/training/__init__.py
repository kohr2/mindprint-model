"""
Training module for Bob Loukas mindprint RLHF.

Contains LoRA adapter merging, training utilities, and MPS support.
PPO-specific: includes reward model and PPO trainer.
"""

from .merge import MergeConfig, MergeResult, LoRAMerger
from .mps_utils import (
    MPSConfig,
    get_mps_device,
    mps_empty_cache,
    move_to_device,
    check_mps_operation_support,
    MPSTrainingContext,
)
from .sft_trainer import SFTConfig, SFTResult, SFTDataset, SFTTrainer
from .reward_model import (
    RewardConfig,
    RewardResult,
    RewardModel,
    RewardModelTrainer,
)
from .ppo_trainer import PPOConfig, PPOResult, PPOTrainer
from .ppo_pipeline import (
    TopicStatus,
    PipelineConfig,
    TopicProgress,
    ChapterProgress,
    UnitProgress,
    PipelineResult,
    PPOPipeline,
)

__all__ = [
    "MergeConfig",
    "MergeResult",
    "LoRAMerger",
    "MPSConfig",
    "get_mps_device",
    "mps_empty_cache",
    "move_to_device",
    "check_mps_operation_support",
    "MPSTrainingContext",
    "SFTConfig",
    "SFTResult",
    "SFTDataset",
    "SFTTrainer",
    "RewardConfig",
    "RewardResult",
    "RewardModel",
    "RewardModelTrainer",
    "PPOConfig",
    "PPOResult",
    "PPOTrainer",
    "TopicStatus",
    "PipelineConfig",
    "TopicProgress",
    "ChapterProgress",
    "UnitProgress",
    "PipelineResult",
    "PPOPipeline",
]
