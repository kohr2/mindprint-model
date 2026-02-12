"""
Training module for Bob Loukas mindprint ORPO.

Heavy dependencies (e.g. torch/transformers) are optional at import time so
lightweight unit tests can import specific modules without the full ML stack.
"""

from typing import Any

__all__ = []


def _export(name: str, value: Any) -> None:
    globals()[name] = value
    __all__.append(name)


try:
    from .merge import MergeConfig, MergeResult, LoRAMerger

    _export("MergeConfig", MergeConfig)
    _export("MergeResult", MergeResult)
    _export("LoRAMerger", LoRAMerger)
except Exception:
    pass

try:
    from .mps_utils import (
        MPSConfig,
        get_mps_device,
        mps_empty_cache,
        move_to_device,
        check_mps_operation_support,
        MPSTrainingContext,
    )

    _export("MPSConfig", MPSConfig)
    _export("get_mps_device", get_mps_device)
    _export("mps_empty_cache", mps_empty_cache)
    _export("move_to_device", move_to_device)
    _export("check_mps_operation_support", check_mps_operation_support)
    _export("MPSTrainingContext", MPSTrainingContext)
except Exception:
    pass

try:
    from .reward_model import (
        RewardConfig,
        RewardResult,
        RewardModel,
        RewardModelTrainer,
    )

    _export("RewardConfig", RewardConfig)
    _export("RewardResult", RewardResult)
    _export("RewardModel", RewardModel)
    _export("RewardModelTrainer", RewardModelTrainer)
except Exception:
    pass

try:
    from .adapter_utils import (
        get_adapter_paths,
        get_merged_adapter_path,
        parse_topic_id,
    )

    _export("get_adapter_paths", get_adapter_paths)
    _export("get_merged_adapter_path", get_merged_adapter_path)
    _export("parse_topic_id", parse_topic_id)
except Exception:
    pass

try:
    from .data_quality import DataQualityMetrics

    _export("DataQualityMetrics", DataQualityMetrics)
except Exception:
    pass

from .orpo_pipeline import (
    TopicStatus,
    PipelineConfig,
    TopicProgress,
    ChapterProgress,
    UnitProgress,
    PipelineResult,
    DPOPipeline,
)

_export("TopicStatus", TopicStatus)
_export("PipelineConfig", PipelineConfig)
_export("TopicProgress", TopicProgress)
_export("ChapterProgress", ChapterProgress)
_export("UnitProgress", UnitProgress)
_export("PipelineResult", PipelineResult)
_export("DPOPipeline", DPOPipeline)

