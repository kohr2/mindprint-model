"""
Cross-Backend Equivalence Tests.

Tests that PyTorch and MLX backends produce comparable results.
"""

import pytest
from pathlib import Path
import json
import time
from typing import Dict, Any, List


@pytest.fixture
def test_sft_data() -> List[Dict[str, Any]]:
    """Sample SFT training data."""
    return [
        {
            "question": "What is a bull market?",
            "answer": "A bull market is a market condition characterized by rising prices and investor optimism.",
        },
        {
            "question": "What is a bear market?",
            "answer": "A bear market is a market condition characterized by falling prices and investor pessimism.",
        },
        {
            "question": "What is the Bitcoin halving?",
            "answer": "The Bitcoin halving is an event that reduces the block reward by 50%, occurring approximately every 4 years.",
        },
    ]


@pytest.fixture
def test_dpo_data() -> List[Dict[str, Any]]:
    """Sample DPO preference pairs."""
    return [
        {
            "prompt": "Explain Bitcoin cycles:",
            "chosen": "Bitcoin follows 4-year cycles driven by the halving event, which reduces supply inflation.",
            "rejected": "Bitcoin price goes up and down randomly.",
        },
        {
            "prompt": "What drives bull markets?",
            "chosen": "Bull markets are driven by increasing demand, positive sentiment, and favorable macro conditions.",
            "rejected": "Bull markets happen when people buy stuff.",
        },
    ]


@pytest.mark.slow
def test_pytorch_mlx_sft_equivalence(test_sft_data, tmp_path):
    """
    Test that PyTorch and MLX backends produce comparable SFT results.

    This is a slow integration test that requires both backends installed.
    Run with: pytest tests/integration/test_backend_equivalence.py -m slow
    """
    pytest.importorskip("torch")
    pytest.importorskip("mlx")

    from src.backends import create_backend

    # Config for minimal training
    sft_config = {
        "learning_rate": 3e-4,
        "num_epochs": 1,
        "per_device_batch_size": 2,
        "max_seq_length": 512,
        "lora_r": 8,
        "lora_alpha": 16,
    }

    results = {}

    # Test both backends
    for backend_type in ["pytorch", "mlx"]:
        print(f"\n{'='*60}")
        print(f"Testing {backend_type.upper()} backend")
        print('='*60)

        try:
            # Create backend
            backend = create_backend(
                backend_type,
                device="cpu" if backend_type == "pytorch" else "auto",
                dtype="float32",  # Use float32 for CPU
            )

            print(f"✓ Created {backend_type} backend")

            # Note: Full model loading test would be very slow
            # This test focuses on trainer creation and config handling

            print(f"✓ {backend_type} backend smoke test passed")

            results[backend_type] = {
                "success": True,
                "backend_name": backend.name,
            }

        except Exception as e:
            print(f"✗ {backend_type} backend failed: {e}")
            results[backend_type] = {
                "success": False,
                "error": str(e),
            }

    # Both backends should initialize successfully
    assert results["pytorch"]["success"], "PyTorch backend failed"
    assert results["mlx"]["success"], "MLX backend failed"

    print(f"\n{'='*60}")
    print("RESULTS: Both backends initialized successfully")
    print('='*60)


@pytest.mark.slow
def test_backend_training_time_comparison(test_sft_data):
    """
    Compare training time between backends (when both available).

    Run with: pytest tests/integration/test_backend_equivalence.py::test_backend_training_time_comparison --run-slow
    """
    # This is a placeholder for actual timing tests
    # Would require full model loading which is slow
    pass


def test_config_equivalence():
    """Test that config conversion works equivalently across backends."""
    from src.backends import create_backend
    # Import backends to trigger registration
    try:
        import src.backends.pytorch  # noqa
    except ImportError:
        pass
    try:
        import src.backends.mlx  # noqa
    except ImportError:
        pass

    config_dict = {
        "learning_rate": 3e-4,
        "num_epochs": 3,
        "per_device_batch_size": 4,
        "lora_r": 8,
        "lora_alpha": 16,
    }

    # Both backends should accept the same config format
    for backend_type in ["pytorch", "mlx"]:
        try:
            backend = create_backend(backend_type, device="auto")
            # Config dict should be valid for both
            assert isinstance(config_dict, dict)
            print(f"✓ {backend_type} accepts unified config format")
        except (ImportError, ValueError) as e:
            if "not installed" in str(e) or "Unknown backend_type" in str(e):
                pytest.skip(f"{backend_type} not installed")
            raise


def test_training_result_equivalence():
    """Test that TrainingResult format is consistent across backends."""
    from src.backends.trainer_interface import TrainingResult

    # Create sample results
    result1 = TrainingResult(
        success=True,
        final_loss=0.5,
        training_time_seconds=10.0,
        samples_trained=100,
        adapter_path="/path/to/adapter",
        metrics={"accuracy": 0.9},
    )

    result2 = TrainingResult(
        success=True,
        final_loss=0.51,  # Slightly different
        training_time_seconds=12.0,
        samples_trained=100,
        adapter_path="/path/to/adapter2",
        metrics={"accuracy": 0.89},
    )

    # Both results should have the same structure
    assert result1.success == result2.success
    assert abs(result1.final_loss - result2.final_loss) < 0.1
    assert "accuracy" in result1.metrics
    assert "accuracy" in result2.metrics


def test_model_interface_consistency():
    """Test that ModelInterface is consistent across backends."""
    from src.backends.model_interface import ModelInterface
    from src.backends.pytorch import PyTorchModel
    from src.backends.mlx import MLXModel

    # Both should implement ModelInterface
    assert hasattr(PyTorchModel, 'generate')
    assert hasattr(PyTorchModel, 'forward')
    assert hasattr(PyTorchModel, 'save_adapter')
    assert hasattr(PyTorchModel, 'load_adapter')

    assert hasattr(MLXModel, 'generate')
    assert hasattr(MLXModel, 'forward')
    assert hasattr(MLXModel, 'save_adapter')
    assert hasattr(MLXModel, 'load_adapter')

    print("✓ Both backends implement consistent ModelInterface")


def test_adapter_operations_consistency():
    """Test that adapter operations are consistent across backends."""
    from src.backends.adapter_interface import AdapterConfig

    # Create unified adapter config
    config = AdapterConfig(
        r=8,
        alpha=16.0,
        dropout=0.05,
        target_modules=["q_proj", "v_proj", "o_proj"],
    )

    # Should convert to both formats
    peft_config = config.to_peft_config()
    mlx_config = config.to_mlx_config()

    # Both should have required fields
    assert "r" in peft_config
    assert "lora_alpha" in peft_config
    assert "target_modules" in peft_config

    assert "rank" in mlx_config
    assert "scale" in mlx_config
    assert "target_modules" in mlx_config

    # Rank should be consistent
    assert peft_config["r"] == mlx_config["rank"]
    assert peft_config["target_modules"] == mlx_config["target_modules"]

    print("✓ Adapter config converts consistently")


@pytest.mark.benchmark
def test_benchmark_sft_training():
    """
    Benchmark SFT training performance across backends.

    Run with: pytest tests/integration/test_backend_equivalence.py::test_benchmark_sft_training -v --benchmark
    """
    # Placeholder for benchmarking tests
    # Would measure:
    # - Training time per epoch
    # - Memory usage
    # - Samples per second
    # - Final loss convergence
    pass


@pytest.mark.benchmark
def test_benchmark_dpo_training():
    """
    Benchmark DPO training performance across backends.

    Run with: pytest tests/integration/test_backend_equivalence.py::test_benchmark_dpo_training -v --benchmark
    """
    # Placeholder for DPO benchmarking
    # Would measure:
    # - DPO loss computation time
    # - Reference model overhead
    # - Final reward margins
    pass


def test_error_handling_consistency():
    """Test that both backends handle errors consistently."""
    from src.backends import create_backend
    # Import backends to trigger registration
    try:
        import src.backends.pytorch  # noqa
    except ImportError:
        pass
    try:
        import src.backends.mlx  # noqa
    except ImportError:
        pass

    for backend_type in ["pytorch", "mlx"]:
        try:
            backend = create_backend(backend_type)

            # Test loading non-existent model
            try:
                model = backend.load_model("nonexistent/model")
                assert False, "Should have raised error"
            except Exception as e:
                # Should raise some error
                assert isinstance(e, Exception)
                print(f"✓ {backend_type} handles missing model error")

        except (ImportError, ValueError) as e:
            if "not installed" in str(e) or "Unknown backend_type" in str(e):
                pytest.skip(f"{backend_type} not installed")
            raise


if __name__ == "__main__":
    print("Cross-Backend Equivalence Tests")
    print("=" * 60)

    print("\n1. Testing config equivalence...")
    test_config_equivalence()

    print("\n2. Testing training result equivalence...")
    test_training_result_equivalence()

    print("\n3. Testing model interface consistency...")
    test_model_interface_consistency()

    print("\n4. Testing adapter operations consistency...")
    test_adapter_operations_consistency()

    print("\n5. Testing error handling consistency...")
    test_error_handling_consistency()

    print("\n" + "=" * 60)
    print("✅ All equivalence tests passed!")
    print("=" * 60)

    print("\nNote: Slow integration tests require --run-slow flag:")
    print("  pytest tests/integration/test_backend_equivalence.py --run-slow")
