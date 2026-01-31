"""
Training module for Bob Loukas mindprint ORPO.

Contains LoRA adapter merging, training utilities, MPS support, and ORPO pipeline.
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
from .orpo_pipeline import (
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
    "RewardConfig",
    "RewardResult",
    "RewardModel",
    "RewardModelTrainer",
    "get_adapter_paths",
    "get_merged_adapter_path",
    "parse_topic_id",
    "DataQualityMetrics",
    "TopicStatus",
    "PipelineConfig",
    "TopicProgress",
    "ChapterProgress",
    "UnitProgress",
    "PipelineResult",
    "DPOPipeline",
]
