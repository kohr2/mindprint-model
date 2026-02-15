"""Regression tests for ORPO pipeline behavior."""

from types import SimpleNamespace
from unittest.mock import patch

from src.training.orpo_pipeline import (
    ChapterProgress,
    DPOPipeline,
    PipelineConfig,
    TopicProgress,
    TopicStatus,
    UnitProgress,
)


def _mock_tokenizer():
    return SimpleNamespace(pad_token="<pad>", eos_token="</s>")


def test_pipeline_config_exposes_orpo_lora_settings() -> None:
    """ORPO config includes LoRA settings used by backend adapter setup."""
    config = PipelineConfig()
    assert config.orpo_lora_rank == 8
    assert config.orpo_lora_alpha == 16.0
    assert config.orpo_lora_dropout == 0.05
    assert config.orpo_target_modules == [
        "q_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
    ]


def test_resume_failed_topics_do_not_trigger_early_stopping(tmp_path) -> None:
    """
    Early stopping must ignore failed-topic losses from resume state.

    Otherwise runs can stop early even when quality is poor.
    """
    config = PipelineConfig(
        early_stopping_enabled=True,
        early_stopping_patience=2,
        early_stopping_cv_threshold=15.0,
        early_stopping_min_topics=1,
        merge_after_unit=False,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "out"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    pipeline = DPOPipeline(model=object(), tokenizer=_mock_tokenizer(), config=config)

    resume_unit = UnitProgress(
        unit_id="unit-01",
        chapters=[
            ChapterProgress(
                chapter_id="unit-01/chapter-01",
                topics=[
                    TopicProgress("topic-1", TopicStatus.FAILED, orpo_loss=0.10),
                    TopicProgress("topic-2", TopicStatus.FAILED, orpo_loss=0.11),
                ],
            )
        ],
    )
    initial_progress = {"result": {"units": [resume_unit.to_dict()]}}

    curriculum = [
        {
            "unit_id": "unit-01",
            "chapters": [
                {
                    "chapter_id": "unit-01/chapter-01",
                    "topics": [
                        {"topic_id": "topic-1"},
                        {"topic_id": "topic-2"},
                        {"topic_id": "topic-3"},
                    ],
                }
            ],
        }
    ]

    with (
        patch.object(pipeline, "_load_preference_data", return_value=[]),
        patch.object(pipeline, "_group_data_by_topic", return_value={}),
        patch.object(pipeline, "_organize_curriculum", return_value=curriculum),
        patch.object(
            pipeline,
            "train_topic",
            return_value=TopicProgress("topic-3", TopicStatus.FAILED, orpo_loss=0.12),
        ) as train_topic_mock,
    ):
        result = pipeline.train_curriculum(initial_progress=initial_progress)

    assert train_topic_mock.call_count == 1
    assert result.total_topics == 3

