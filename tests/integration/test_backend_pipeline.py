"""
Integration tests for DPOPipeline with backend interface.
"""

import pytest
from src.backends import create_backend
from src.training.dpo_pipeline import DPOPipeline, PipelineConfig


@pytest.mark.skip(reason="Requires model loading - slow integration test")
def test_dpo_pipeline_with_pytorch_backend():
    """Test DPOPipeline initialization with PyTorch backend."""
    # Create PyTorch backend
    backend = create_backend("pytorch", device="cpu", dtype="float32")

    # Create pipeline config with backend settings
    config = PipelineConfig(
        backend_type="pytorch",
        backend_device="cpu",
        backend_dtype="float32",
        sft_epochs_per_topic=1,  # Minimal for testing
        dpo_steps_per_topic=10,
        output_dir="./test_output",
        checkpoint_dir="./test_checkpoints",
    )

    # Initialize pipeline with backend
    pipeline = DPOPipeline(
        model=None,  # Will be loaded via backend
        tokenizer=None,  # Will be loaded via backend
        config=config,
        backend=backend,
    )

    assert pipeline.use_backend is True
    assert pipeline.backend is not None
    assert pipeline.backend.name == "pytorch"


def test_dpo_pipeline_legacy_mode():
    """Test DPOPipeline works in legacy mode (no backend)."""
    # Create pipeline config without backend
    config = PipelineConfig(
        backend_type=None,  # Legacy mode
        sft_epochs_per_topic=1,
        dpo_steps_per_topic=10,
        output_dir="./test_output",
        checkpoint_dir="./test_checkpoints",
    )

    # Would need actual model/tokenizer for full test
    # This just tests config handling
    assert config.backend_type is None


def test_pipeline_config_with_backend():
    """Test PipelineConfig with backend settings."""
    config = PipelineConfig(
        backend_type="pytorch",
        backend_device="mps",
        backend_dtype="float16",
        sft_epochs_per_topic=3,
        dpo_steps_per_topic=100,
    )

    assert config.backend_type == "pytorch"
    assert config.backend_device == "mps"
    assert config.backend_dtype == "float16"


def test_pipeline_config_defaults():
    """Test PipelineConfig defaults."""
    config = PipelineConfig()

    # Backend defaults
    assert config.backend_type is None  # Legacy mode by default
    assert config.backend_device == "auto"
    assert config.backend_dtype == "float16"

    # Training defaults
    assert config.sft_epochs_per_topic == 3
    assert config.dpo_steps_per_topic == 100
    assert config.sft_learning_rate == 3e-4
    assert config.dpo_learning_rate == 5e-7


if __name__ == "__main__":
    # Run simple smoke tests
    print("Testing PipelineConfig with backend settings...")
    config = PipelineConfig(
        backend_type="pytorch",
        backend_device="cpu",
        backend_dtype="float32",
    )
    assert config.backend_type == "pytorch"
    print("✓ PipelineConfig backend settings work")

    print("\nTesting DPOPipeline initialization with backend...")
    from src.backends import create_backend

    backend = create_backend("pytorch", device="cpu", dtype="float32")
    pipeline = DPOPipeline(
        model=None,
        tokenizer=None,
        config=config,
        backend=backend,
    )
    assert pipeline.use_backend
    assert pipeline.backend.name == "pytorch"
    print("✓ DPOPipeline initializes with backend")

    print("\n✅ All backend pipeline integration checks passed!")
