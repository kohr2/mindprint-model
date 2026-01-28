"""
Unit tests for MLX backend.
"""

import pytest
from src.backends import BackendRegistry, create_backend


def test_mlx_backend_registered():
    """Test that MLX backend is automatically registered."""
    # Import mlx backend module to trigger registration
    try:
        import src.backends.mlx  # noqa
        assert BackendRegistry.is_registered("mlx")
        assert "mlx" in BackendRegistry.list_backends()
    except ImportError:
        pytest.skip("mlx-lm not installed")


def test_create_mlx_backend():
    """Test creating MLX backend via factory."""
    try:
        from src.backends import create_backend

        # Create backend
        backend = create_backend("mlx", device="auto", dtype="float16", seed=42)

        assert backend is not None
        assert backend.name == "mlx"
        assert backend.config.backend_type == "mlx"
        assert backend.config.dtype == "float16"
        assert backend.config.seed == 42
    except ImportError:
        pytest.skip("mlx-lm not installed")


def test_mlx_backend_auto_registers():
    """Test that MLX backend auto-registers on import."""
    try:
        from src.backends.mlx import MLXBackend
        from src.backends import BackendRegistry

        assert BackendRegistry.is_registered("mlx")
        assert "mlx" in BackendRegistry.list_backends()
    except ImportError:
        pytest.skip("mlx-lm not installed")


def test_mlx_device_manager():
    """Test MLX device manager."""
    try:
        from src.backends.mlx import MLXDeviceManager

        device_manager = MLXDeviceManager("auto")

        assert device_manager.get_device() == "gpu"
        assert device_manager.is_gpu
        assert device_manager.is_available()

        # Test no-op methods
        device_manager.empty_cache()  # Should not raise
        device_manager.synchronize()  # Should not raise (may skip if mlx not installed)

    except ImportError:
        pytest.skip("mlx not installed")


def test_mlx_backend_supports_both_backends():
    """Test that both PyTorch and MLX backends can coexist."""
    # Import both backends
    try:
        import src.backends.pytorch  # noqa
        import src.backends.mlx  # noqa
    except ImportError:
        pytest.skip("mlx or pytorch not installed")

    # Both should be registered
    backends = BackendRegistry.list_backends()
    assert "pytorch" in backends
    assert "mlx" in backends


if __name__ == "__main__":
    # Simple smoke test
    try:
        from src.backends import BackendRegistry

        # Import mlx package to trigger registration
        import src.backends.mlx

        assert BackendRegistry.is_registered("mlx"), "MLX backend not registered!"
        print("✓ MLX backend registered successfully")

        # Try creating backend
        from src.backends import create_backend

        backend = create_backend("mlx", device="auto", dtype="float16")
        print(f"✓ Created MLX backend: {backend.name}")

        print("\n✅ All basic MLX backend checks passed!")
    except ImportError as e:
        print(f"⚠️  MLX dependencies not installed: {e}")
        print("Install with: pip install mlx mlx-lm")
