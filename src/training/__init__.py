"""
Training module for Bob Loukas mindprint RLHF.

Shared infrastructure from 'shared' branch + DPO-specific components.
Contains LoRA adapter merging, training utilities, MPS support, and DPO trainer.
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
from .adapter_utils import (
    get_adapter_paths,
    get_merged_adapter_path,
    parse_topic_id,
)
from .data_quality import DataQualityMetrics
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
    "RewardConfig",
    "RewardResult",
    "RewardModel",
    "RewardModelTrainer",
    "get_adapter_paths",
    "get_merged_adapter_path",
    "parse_topic_id",
    "DataQualityMetrics",
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
