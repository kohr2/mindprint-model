"""Regression tests for ORPO pipeline behavior."""

from types import SimpleNamespace
from unittest.mock import patch

from src.training.orpo_pipeline import (
    ChapterProgress,
    ORPOPipeline,
    PipelineConfig,
    PipelineResult,
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
        require_holdout_for_gate=False,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "out"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    pipeline = ORPOPipeline(model=object(), tokenizer=_mock_tokenizer(), config=config)

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
        patch.object(
            pipeline,
            "_build_split_groups",
            return_value=(
                {},
                {
                    "summary": {
                        "total_records": 0,
                        "train_records": 0,
                        "holdout_records": 0,
                        "total_topics": 0,
                        "topics_without_holdout": [],
                    }
                },
            ),
        ),
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


def test_split_is_deterministic_and_leak_free(tmp_path) -> None:
    """Deterministic split must be stable and produce disjoint train/holdout sets."""
    config = PipelineConfig(
        split_seed=7,
        holdout_ratio=0.4,
        holdout_min_examples_per_topic=1,
        require_holdout_for_gate=False,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "out"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    pipeline = ORPOPipeline(model=object(), tokenizer=_mock_tokenizer(), config=config)

    topic_id = "episode-2026-02-01"
    preference_data = [
        {
            "source": topic_id,
            "prompt": f"Q{i}",
            "chosen": f"A{i}",
            "rejected": f"R{i}",
        }
        for i in range(6)
    ]

    grouped_a, manifest_a = pipeline._build_split_groups(preference_data)
    grouped_b, manifest_b = pipeline._build_split_groups(preference_data)

    assert manifest_a["summary"]["assignment_digest"] == manifest_b["summary"]["assignment_digest"]
    topic_manifest = manifest_a["topics"][topic_id]
    train_ids = set(topic_manifest["train_record_ids"])
    holdout_ids = set(topic_manifest["holdout_record_ids"])
    assert train_ids.isdisjoint(holdout_ids)
    assert len(grouped_a[topic_id]["holdout_pairs"]) >= 1
    assert len(grouped_a[topic_id]["preference_pairs"]) >= 1


def test_train_curriculum_fails_when_holdout_required_but_unavailable(tmp_path) -> None:
    """Holdout-required mode should fail fast when a topic cannot produce holdout records."""
    config = PipelineConfig(
        split_seed=42,
        holdout_ratio=0.5,
        holdout_min_examples_per_topic=1,
        require_holdout_for_gate=True,
        merge_after_unit=False,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "out"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    pipeline = ORPOPipeline(model=object(), tokenizer=_mock_tokenizer(), config=config)

    preference_data = [
        {
            "source": "episode-2026-02-10",
            "prompt": "Only sample",
            "chosen": "Answer",
            "rejected": "Bad",
        }
    ]

    result = pipeline.train_curriculum(preference_data=preference_data)
    assert result.success is False
    assert result.total_topics == 0


def test_promotion_gate_reports_threshold_reasons(tmp_path) -> None:
    """Promotion gate should provide explicit threshold failure reasons."""
    config = PipelineConfig(
        accuracy_threshold=0.7,
        topic_pass_threshold=0.9,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "out"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    pipeline = ORPOPipeline(model=object(), tokenizer=_mock_tokenizer(), config=config)

    pipeline.topic_gate_details = {
        "topic-1": {
            "topic_id": "topic-1",
            "status": "failed",
            "accuracy": 0.5,
            "voice_score": 0.6,
            "combined_score": 0.55,
            "train_examples": 10,
            "holdout_examples": 2,
            "gate_reason": "",
        }
    }
    result = PipelineResult(
        success=False,
        total_topics=1,
        passed_topics=0,
        failed_topics=["topic-1"],
        total_training_time_hours=0.1,
    )

    gate = pipeline.build_promotion_gate(result)
    assert gate["passed"] is False
    assert any("avg_accuracy_below_threshold" in reason for reason in gate["failed_conditions"])
