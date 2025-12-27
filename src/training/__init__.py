"""
Training module for Bob Loukas mindprint RLHF.

Contains LoRA adapter merging, training utilities, and MPS support.
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
from .dpo_trainer import Rank1DPOConfig, DPOResult, Rank1DPOTrainer
from .dpo_pipeline import (
    TopicStatus,
    PipelineConfig,
    TopicProgress,
    ChapterProgress,
    UnitProgress,
    PipelineResult,
    DPOPipeline,
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
    "Rank1DPOConfig",
    "DPOResult",
    "Rank1DPOTrainer",
    "TopicStatus",
    "PipelineConfig",
    "TopicProgress",
    "ChapterProgress",
    "UnitProgress",
    "PipelineResult",
    "DPOPipeline",
]
