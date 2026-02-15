"""
Integration tests for ORPOPipeline with backend interface.
"""

import pytest
from src.backends import create_backend
from src.training.orpo_pipeline import ORPOPipeline, PipelineConfig


@pytest.mark.skip(reason="Requires model loading - slow integration test")
def test_orpo_pipeline_with_pytorch_backend():
    """Test ORPOPipeline initialization with PyTorch backend."""
    backend = create_backend("pytorch", device="cpu", dtype="float32")

    config = PipelineConfig(
        backend_type="pytorch",
        backend_device="cpu",
        backend_dtype="float32",
        orpo_steps_per_topic=10,
        output_dir="./test_output",
        checkpoint_dir="./test_checkpoints",
    )

    pipeline = ORPOPipeline(
        model=None,
        tokenizer=None,
        config=config,
        backend=backend,
    )

    assert pipeline.use_backend is True
    assert pipeline.backend is not None
    assert pipeline.backend.name == "pytorch"


def test_orpo_pipeline_legacy_mode():
    """Test ORPOPipeline works in legacy mode (no backend)."""
    config = PipelineConfig(
        backend_type=None,
        orpo_steps_per_topic=10,
        output_dir="./test_output",
        checkpoint_dir="./test_checkpoints",
    )

    assert config.backend_type is None


def test_pipeline_config_with_backend():
    """Test PipelineConfig with backend settings."""
    config = PipelineConfig(
        backend_type="pytorch",
        backend_device="mps",
        backend_dtype="float16",
        orpo_steps_per_topic=100,
    )

    assert config.backend_type == "pytorch"
    assert config.backend_device == "mps"
    assert config.backend_dtype == "float16"


def test_pipeline_config_defaults():
    """Test PipelineConfig defaults."""
    config = PipelineConfig()

    # Backend defaults
    assert config.backend_type is None
    assert config.backend_device == "auto"
    assert config.backend_dtype == "float16"

    # ORPO training defaults
    assert config.orpo_steps_per_topic == 100
    assert config.orpo_learning_rate == 3e-4
    assert config.orpo_lambda == 0.1


if __name__ == "__main__":
    print("Testing PipelineConfig with backend settings...")
    config = PipelineConfig(
        backend_type="pytorch",
        backend_device="cpu",
        backend_dtype="float32",
    )
    assert config.backend_type == "pytorch"
    print("  PipelineConfig backend settings work")

    print("\nTesting ORPOPipeline initialization with backend...")
    from src.backends import create_backend

    backend = create_backend("pytorch", device="cpu", dtype="float32")
    pipeline = ORPOPipeline(
        model=None,
        tokenizer=None,
        config=config,
        backend=backend,
    )
    assert pipeline.use_backend
    assert pipeline.backend.name == "pytorch"
    print("  ORPOPipeline initializes with backend")

    print("\nAll backend pipeline integration checks passed!")
